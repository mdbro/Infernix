// The spill guard's confirming read (core/residency_probe.h, core/vram_budget.h): device memory
// reads above the floor and mapped host memory (system memory over PCIe, the same path a WDDM
// sysmem-fallback allocation takes) below it; a mixed list is split slice by slice; a slice below
// the minimum is not measured; and SpillGuard::end confirms or clears a scripted free-memory
// shortfall from that read alone. Skips (77) without a CUDA device.

#include "core/arena.h"
#include "core/device.h"
#include "core/residency_probe.h"
#include "core/vram_budget.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

constexpr std::size_t kMiB = std::size_t{1} << 20;

int failures = 0;

void require(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// Free memory that never drops: every step looks spilled to the bookkeeping.
class StuckSource final : public infernix::VramBudgetSource {
public:
    infernix::VramSnapshot query() override {
        infernix::VramSnapshot s;
        s.device_free  = 8ULL << 30;
        s.device_total = 32ULL << 30;
        return s;
    }
};

// Free memory that drops by exactly what the step allocated.
class TrackingSource final : public infernix::VramBudgetSource {
public:
    explicit TrackingSource(std::uint64_t drop_after_begin) : drop_(drop_after_begin) {}
    infernix::VramSnapshot query() override {
        infernix::VramSnapshot s;
        s.device_free  = (8ULL << 30) - (calls_++ == 0 ? 0 : drop_);
        s.device_total = 32ULL << 30;
        return s;
    }

private:
    std::uint64_t drop_;
    int calls_ = 0;
};

double gbps(double bytes_per_second) { return bytes_per_second / 1e9; }

} // namespace

int main() {
    int count = 0;
    if (const cudaError_t error = cudaGetDeviceCount(&count);
        error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    try {
        infernix::DeviceContext device(0);
        const cudaStream_t stream = device.stream;

        const std::size_t device_bytes = 256 * kMiB;
        infernix::DeviceBuffer resident(device_bytes);
        resident.fill(0x5A);
        const infernix::DeviceSpan device_span{resident.p, resident.bytes};

        // Mapped pinned host memory: the GPU reads it over PCIe, as it reads a spilled allocation.
        const std::size_t host_bytes = 128 * kMiB;
        void* host                   = nullptr;
        CUDA_CHECK(cudaHostAlloc(&host, host_bytes, cudaHostAllocMapped | cudaHostAllocPortable));
        void* host_on_device = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(&host_on_device, host, 0));
        CUDA_CHECK(cudaMemset(host_on_device, 0x5A, host_bytes));
        CUDA_CHECK(cudaDeviceSynchronize());
        const infernix::DeviceSpan host_span{host_on_device, host_bytes};

        // 1. Device memory: every slice above the floor, repeatably.
        for (int repeat = 0; repeat < 5; ++repeat) {
            const auto r = infernix::measure_residency(std::span(&device_span, 1), stream);
            require(r.measured_bytes == device_bytes, "device: every byte measured");
            require(r.slow_bytes == 0, "device: no slow slice");
            require(r.slowest_bytes_per_second >= infernix::kDeviceReadFloorBytesPerSecond, "device: above the floor");
            if (repeat == 0) { std::printf("device memory: slowest slice %.0f GB/s\n", gbps(r.slowest_bytes_per_second)); }
        }

        // 2. System memory: every slice below the floor, in every pass (cache-volatile loads).
        {
            const auto r = infernix::measure_residency(std::span(&host_span, 1), stream);
            require(r.measured_bytes == host_bytes, "host: every byte measured");
            require(r.slow_bytes == host_bytes, "host: every slice slow");
            require(r.slowest_bytes_per_second < infernix::kDeviceReadFloorBytesPerSecond, "host: below the floor");
            std::printf("mapped host memory: slowest slice %.0f GB/s (floor %.0f)\n", gbps(r.slowest_bytes_per_second),
                        gbps(infernix::kDeviceReadFloorBytesPerSecond));
        }

        // 3. Mixed ranges: the slow bytes are the host range's.
        {
            const infernix::DeviceSpan both[] = {device_span, host_span};
            const auto r                      = infernix::measure_residency(both, stream);
            require(r.measured_bytes == device_bytes + host_bytes, "mixed: every byte measured");
            require(r.slow_bytes == host_bytes, "mixed: the host bytes are the slow ones");
        }

        // 4. A range below the minimum is not measured and counts as resident.
        {
            infernix::DeviceBuffer tiny(4 * kMiB);
            const infernix::DeviceSpan tiny_span{tiny.p, tiny.bytes};
            const auto r = infernix::measure_residency(std::span(&tiny_span, 1), stream);
            require(r.measured_bytes == 0 && r.slow_bytes == 0 && r.slowest_bytes_per_second == 0.0,
                    "tiny: not measured");
        }

        // 5. The guard: a shortfall over device memory is cleared, one over system memory confirmed,
        //    and no shortfall means no read.
        {
            StuckSource stuck;
            infernix::SpillGuard guard(stuck);
            guard.begin();
            const auto cleared = guard.end(device_bytes, std::span(&device_span, 1), stream);
            require(cleared.suspected == device_bytes, "guard: the bookkeeping suspects the whole step");
            require(cleared.spilled == 0, "guard: the read clears it over device memory");
            require(cleared.slowest_bytes_per_second > 0.0, "guard: the read ran");

            guard.begin();
            const auto confirmed = guard.end(host_bytes, std::span(&host_span, 1), stream);
            require(confirmed.suspected == host_bytes, "guard: suspected over host memory");
            require(confirmed.spilled == host_bytes, "guard: the read confirms the spill");

            TrackingSource tracking(device_bytes);
            infernix::SpillGuard quiet(tracking);
            quiet.begin();
            const auto none = quiet.end(device_bytes, std::span(&device_span, 1), stream);
            require(none.suspected == 0 && none.spilled == 0 && none.slowest_bytes_per_second == 0.0,
                    "guard: an accounted step is not read");
        }

        CUDA_CHECK(cudaFreeHost(host));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
