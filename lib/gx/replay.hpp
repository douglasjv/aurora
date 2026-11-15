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
#include "../gfx/texture.hpp"

#include <span>

#include <dolphin/gx/GXEnum.h>

namespace aurora::gx {
struct AttrArray;
}

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

// A draw's vertex bytes as the FIFO holds them: on the first decode, records the
// positions of a draw that has them in the stream (GX_DIRECT); on a replay, returns the
// bytes with those positions blended toward the previous tick's matching draw (in a buffer
// valid until the next call), or `data` itself.
const u8* vertices(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, u32 vtxSize, const u8* data) noexcept;

// A texture the previous tick's draw sampled, by texture map.
struct PreviousTexture {
  u32 map;
  gfx::TextureBind bind;
};
struct Crossfade {
  const u8* previous = nullptr; // the vertices with the previous tick's texture coordinates
  float weight = 1.f;           // this tick's share
  std::span<const PreviousTexture> textures; // to bind for the previous cell (a swapped texture)
};
// After vertices(), on a replay with crossfade=1: true when that draw is an additive
// flipbook that stepped a cell or swapped its texture since the previous tick. It is then
// drawn twice: with the bytes vertices() returned, its colour scaled by `weight`, and with
// `previous` and the previous tick's textures, scaled by 1 - weight. Valid until the next
// vertices().
bool crossfade(Crossfade& out) noexcept;
// After a draw is pushed (its textures resolved): on the first decode, notes which textures
// the draw vertices() last recorded samples.
void note_draw_textures() noexcept;
// Whether crossfade=1: additive draws then get shaders that can scale their colour.
bool crossfade_enabled() noexcept;

// A vertex attribute array's bytes for the GPU: on the first decode, keeps a copy of a
// texture coordinate array; on a replay, returns the array blended from the previous tick's
// copy when the game rewrote it (a scroll, taken the short way round a wrap), as floats
// (the game's fixed point moves a whole step a tick: a blend between rounds to one end),
// setting the array's replayType and replayStride and `size` to the bytes to push, in a
// buffer valid until the next call; or the array's own bytes.
const void* array_bytes(GXAttr attr, GXVtxFmt fmt, AttrArray& array, u32& size) noexcept;

// Around build_uniform: records the draw's matrices (first decode) or swaps in the
// blended ones (replay). after_uniform puts the state's matrices back.
void before_uniform(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount) noexcept;
void after_uniform() noexcept;

} // namespace aurora::gx::replay
