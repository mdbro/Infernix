#pragma once

// Whether device memory reads at device speed (docs/maintainer/qwen3_8-flash-next-design.md
// §19.3.7, the spill guard). Under Windows WDDM the driver can place a device allocation in shared
// system memory without an error. The OS bookkeeping (free memory, the process budget) only
// suggests where an allocation went; a timed read measures it: system memory reaches the GPU over
// PCIe at under 64 GB/s (5.0 x16), device memory on every supported card at several times that.

#include "core/arena.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace infernix {

// The floor between the two: twice PCIe 5.0 x16 (64 GB/s) and well below the slowest sm_120 card
// (GDDR7 on 128 bits, 320 GB/s).
inline constexpr double kDeviceReadFloorBytesPerSecond = 128e9;
// Ranges are read in slices of this size, so a partly spilled allocation is caught slice by slice.
inline constexpr std::size_t kResidencySliceBytes = 64ULL << 20;
// A slice below this size is launch-bound and is not measured (it counts as resident).
inline constexpr std::size_t kResidencyMinimumBytes = 8ULL << 20;

struct ResidencyReading {
    std::uint64_t measured_bytes          = 0;   // bytes read (slices of at least the minimum)
    std::uint64_t slow_bytes              = 0;   // of those, slices below the floor: system memory
    double slowest_bytes_per_second       = 0.0; // the slowest measured slice; 0 when nothing was measured
};

// Reads every range on `stream` and returns once the reads have completed: each slice three timed
// passes, the fastest counting, so a GPU time slice given to another process in one pass does not
// make a resident slice slow. The loads are cache-volatile, so a system-memory slice is fetched
// over PCIe in every pass whatever the L2 keeps. Ranges must be device-accessible and 16-byte
// aligned; the reads change nothing.
[[nodiscard]] ResidencyReading measure_residency(std::span<const DeviceSpan> ranges, cudaStream_t stream);

} // namespace infernix
