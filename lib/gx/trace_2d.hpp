#pragma once

#include "../internal.hpp"

#include <dolphin/gx/GXEnum.h>

namespace aurora::gx::trace_2d {
void set_path(const char* path) noexcept;
bool enabled() noexcept;
// Logs an orthographic draw's screen rectangle, texture and texture region (raw FIFO
// vertex bytes); a batch of quads quad by quad.
void draw(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, u32 vtxSize, const u8* data) noexcept;
} // namespace aurora::gx::trace_2d
