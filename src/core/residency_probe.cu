#include "core/residency_probe.h"

#include "core/device.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace infernix {
namespace {

__device__ unsigned g_residency_sink;

// A grid-stride streaming read of `count` 16-byte vectors with cache-volatile loads (PTX ld.cv:
// a cached system-memory line is treated as stale and fetched again), folded into one word that
// is stored only for a value the data never has, so the reads are not eliminated.
__global__ void residency_read_kernel(const uint4* data, std::size_t count) {
    unsigned acc = 0;
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < count;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const uint4 v = __ldcv(data + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9E3779B9U) { g_residency_sink = acc; }
}

struct TimingEvents {
    cudaEvent_t begin = nullptr, end = nullptr;
    TimingEvents() {
        CUDA_CHECK(cudaEventCreate(&begin));
        CUDA_CHECK(cudaEventCreate(&end));
    }
    ~TimingEvents() {
        if (begin != nullptr) { (void)cudaEventDestroy(begin); }
        if (end != nullptr) { (void)cudaEventDestroy(end); }
    }
    TimingEvents(const TimingEvents&)            = delete;
    TimingEvents& operator=(const TimingEvents&) = delete;
};

constexpr int kPasses  = 3;
constexpr int kThreads = 512;

} // namespace

ResidencyReading measure_residency(std::span<const DeviceSpan> ranges, cudaStream_t stream) {
    ResidencyReading out;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int multiprocessors = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
    const unsigned grid = static_cast<unsigned>(std::max(1, multiprocessors) * 4);
    TimingEvents events;
    for (const DeviceSpan& range : ranges) {
        const auto* bytes = static_cast<const std::uint8_t*>(range.data);
        for (std::size_t offset = 0; offset < range.bytes; offset += kResidencySliceBytes) {
            const std::size_t slice = std::min(kResidencySliceBytes, range.bytes - offset);
            if (slice < kResidencyMinimumBytes) { continue; }
            const std::size_t count = slice / sizeof(uint4);
            float best_ms           = std::numeric_limits<float>::infinity();
            for (int pass = 0; pass < kPasses; ++pass) {
                CUDA_CHECK(cudaEventRecord(events.begin, stream));
                residency_read_kernel<<<grid, kThreads, 0, stream>>>(reinterpret_cast<const uint4*>(bytes + offset),
                                                                    count);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventRecord(events.end, stream));
                CUDA_CHECK(cudaEventSynchronize(events.end));
                float ms = 0.0F;
                CUDA_CHECK(cudaEventElapsedTime(&ms, events.begin, events.end));
                best_ms = std::min(best_ms, ms);
            }
            const double read = static_cast<double>(count * sizeof(uint4));
            const double rate = best_ms > 0.0F ? read / (static_cast<double>(best_ms) / 1e3)
                                                : std::numeric_limits<double>::infinity();
            out.measured_bytes += slice;
            if (rate < kDeviceReadFloorBytesPerSecond) { out.slow_bytes += slice; }
            if (out.slowest_bytes_per_second == 0.0 || rate < out.slowest_bytes_per_second) {
                out.slowest_bytes_per_second = rate;
            }
        }
    }
    return out;
}

} // namespace infernix
