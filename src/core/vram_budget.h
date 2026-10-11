#pragma once

// Physical facts about one CUDA device's video memory: free memory for this process, its OS budget
// (Windows WDDM, through DXGI) and whether the device drives a display. No policy: sizing
// decisions live with their owners (docs/maintainer/qwen3_8-flash-next-design.md §19.3.7).

#include "core/arena.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>

namespace infernix {

enum class DisplayState : std::uint8_t {
    Headless, // no output attached and no display reported
    Attached, // an output is attached, or the driver reports an active display
    Unknown,  // neither source could tell; callers treat it as Attached
};

struct VramSnapshot {
    // Free device memory for this process. Under WDDM with a budget source, the OS budget less this
    // process's local usage (DXGI QueryVideoMemoryInfo); otherwise cudaMemGetInfo. The two agree in
    // a fresh process, but inside the engine cudaMemGetInfo has been seen 6.4 GiB low (the dense
    // weights counted twice) and unmoved by a mapped VMM chunk, which sized the expert cache a third
    // too small and made every chunk look spilled; the OS figures tracked both correctly.
    std::size_t device_free  = 0;
    std::size_t device_total = 0;
    std::size_t cuda_free    = 0; // cudaMemGetInfo's own figure, for the ledger
    // Windows WDDM: this process's local-memory budget and usage.
    bool has_budget            = false;
    std::uint64_t local_budget = 0;
    std::uint64_t local_usage  = 0;
    DisplayState display       = DisplayState::Unknown;
    int outputs                = -1; // DXGI outputs of the adapter; -1 unknown
    // A source opened past the budget: the bytes added to the OS budget (already included in
    // local_budget and device_free); 0 otherwise.
    std::uint64_t budget_allowance = 0;
};

// One device's source. query() may be called from any thread on which the device is current.
class VramBudgetSource {
public:
    virtual ~VramBudgetSource()             = default;
    [[nodiscard]] virtual VramSnapshot query() = 0;
    // A Windows event handle (HANDLE) signalled when the OS changes this process's video-memory
    // budget, or nullptr; it stays valid for the source's lifetime.
    [[nodiscard]] virtual void* change_event() { return nullptr; }
};

// The source for `device`: DXGI and NVML where present (both loaded or created lazily; their
// absence never fails, it only leaves fields unknown), else CUDA alone.
//
// `past_budget` (--vram-past-budget): WDDM's per-process budget withholds part of the physical
// VRAM that is allocatable at device speed (RTX 5090: budget 31,419 MiB against 32,187 MiB of
// physical memory less the driver's reserve; 640 MiB more allocated, design §19.3.8 VRAM item 2).
// The source then measures once, at open, allowance = NVML physical (total - reserved) - the
// budget - kPastBudgetMargin, and reports budget + allowance as the budget, from which free memory
// is derived as always (budget - usage). A later budget cut by the OS lowers the effective budget by
// the same amount. Memory past the budget may be demoted to system memory when another program
// needs VRAM. Without DXGI or NVML the allowance is 0.
inline constexpr std::uint64_t kPastBudgetMargin = 128ULL << 20;
[[nodiscard]] std::unique_ptr<VramBudgetSource> open_vram_budget_source(int device, bool past_budget = false);

// A device allocation step that the driver placed partly in system memory (the Windows WDDM sysmem
// fallback) lowers device free memory by less than it allocated. The bytes of a step of `bytes`
// missing from the drop in free memory beyond `tolerance`; 0: resident.
inline constexpr std::uint64_t kSpillToleranceBytes = 32ULL << 20;
[[nodiscard]] std::uint64_t spill_shortfall(std::uint64_t free_before, std::uint64_t free_after,
                                            std::uint64_t bytes,
                                            std::uint64_t tolerance = kSpillToleranceBytes) noexcept;

// Checks device allocation steps against a source's free memory, and confirms what that bookkeeping
// suspects by reading the memory: a shortfall in free memory can come from the OS budget moving
// with the allocation, or from a free-memory figure that does not reflect it, as well as from a
// real placement in system memory, and only the read tells them apart (core/residency_probe.h).
class SpillGuard {
public:
    explicit SpillGuard(VramBudgetSource& source) : source_(source) {}
    // Records free memory before a step.
    void begin() { before_ = source_.query().device_free; }
    // After a step that allocated `bytes`: the bytes missing from the drop in free memory (0: all
    // of it accounted for). Bookkeeping only; a shortfall is confirmed by a second reading, since
    // another program can change free memory during the step.
    [[nodiscard]] std::uint64_t end(std::uint64_t bytes);

    struct Check {
        std::uint64_t suspected = 0; // bytes missing from the drop in free memory
        std::uint64_t spilled   = 0; // of the step's memory, bytes a timed read found in system memory
        double slowest_bytes_per_second = 0.0; // the confirming read's slowest slice; 0 when none ran
    };
    // After a step that allocated `bytes` into `memory` (device-accessible; the step's work on
    // `stream` is waited for): the bookkeeping's suspicion, and the spill it confirms. The read runs
    // only on a suspicion, in slices (kResidencySliceBytes); slices below kResidencyMinimumBytes
    // count as resident. A suspicion the read clears is the bookkeeping's error, not a spill.
    [[nodiscard]] Check end(std::uint64_t bytes, std::span<const DeviceSpan> memory, cudaStream_t stream);

private:
    VramBudgetSource& source_;
    std::uint64_t before_ = 0;
};

namespace testing {
// Replaces every source opened afterwards with `fake` (tests of sizing and monitoring); an empty
// function restores the real sources.
void set_vram_budget_source(std::function<VramSnapshot()> fake);
} // namespace testing

} // namespace infernix
