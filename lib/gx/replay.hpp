#pragma once

// Frame replay for high frame rates (aurora_replay_*, include/aurora/gfx.h).
//
// While enabled, the bytes of one frame's GX stream (a game tick) are kept as the FIFO
// drains them, and the command processor's state is copied when the frame opens and when
// it closes. aurora_replay_draw decodes the kept bytes again into another frame, starting
// from the opening state, with every draw's matrices blended between the previous frame
// and this one, and without the side effects of the first decode (draw-done, depth
// snapshots, evictions). Off by default; when off, none of this runs.

#include "../internal.hpp"

#include <dolphin/gx/GXEnum.h>

namespace aurora::gx::replay {

// The FIFO is keeping the bytes it drains.
bool capturing() noexcept;
// Called by the FIFO with the bytes a drain or clear consumed, in order.
void append(const u8* data, u32 size) noexcept;

// The command processor is decoding a replay (no side effects).
bool replaying() noexcept;
// A replay met bytes it cannot decode; it stops, and the picture is dropped.
void fail(const char* why) noexcept;
bool failed() noexcept;

// The first decode of a kept frame read `size` bytes of game memory at `data` (vertex
// arrays, indexed matrix loads, textures, TLUTs): a replay may read them again.
void note_read(const void* data, size_t size) noexcept;

// Around build_uniform: records the draw's matrices (first decode) or swaps in the
// blended ones (replay). after_uniform puts the state's matrices back.
void before_uniform(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount) noexcept;
void after_uniform() noexcept;

} // namespace aurora::gx::replay
