#pragma once

#include "../internal.hpp"

#include <array>
#include <atomic>
#include <cstdint>

namespace aurora::gx::fifo {

struct ProcessResult {
  uint32_t bytesProcessed;
  bool drawDone;
};

// Process GX FIFO commands until the next draw done event or end of buffer
ProcessResult process(const uint8_t* data, uint32_t size) noexcept;
void clear_draw_cache() noexcept;
// The frame ended: its storage buffer, and so every array upload in it, is gone.
void forget_array_uploads() noexcept;
// aurora_gx_merge_small_draws.
void set_merge_small_draws(bool enabled) noexcept;
// Frame replay (replay.cpp): keep the draw cache in slot 0 or 1, as it stands between
// frames, and put it back.
void save_draw_cache(int slot) noexcept;
void restore_draw_cache(int slot) noexcept;
// Profiling: draws that found the pipeline, textures, uniforms, immediates dirty; a clean state
// but another vertex format; clean but no gx draw before it; mergeable; all draws.
extern std::array<std::atomic<uint64_t>, 8> g_mergeCounts;
// Of the draws with only the immediates dirty: the current matrix changed, an array was
// invalidated, an array's upload moved, nothing changed.
extern std::array<std::atomic<uint64_t>, 4> g_mergeImmCounts;

} // namespace aurora::gx::fifo
