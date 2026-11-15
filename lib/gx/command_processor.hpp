#pragma once

#include "../internal.hpp"

#include <cstdint>

namespace aurora::gx::fifo {

struct ProcessResult {
  uint32_t bytesProcessed;
  bool drawDone;
};

// Process GX FIFO commands until the next draw done event or end of buffer
ProcessResult process(const uint8_t* data, uint32_t size) noexcept;
void clear_draw_cache() noexcept;
// Frame replay (replay.cpp): keep the draw cache in slot 0 or 1, as it stands between
// frames, and put it back.
void save_draw_cache(int slot) noexcept;
void restore_draw_cache(int slot) noexcept;

} // namespace aurora::gx::fifo
