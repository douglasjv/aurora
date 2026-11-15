#ifndef AURORA_GFX_H
#define AURORA_GFX_H

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>

extern "C" {
#else
#include <stddef.h>
#include "stdint.h"
#endif

#if !defined(NDEBUG) && !defined(AURORA_GFX_DEBUG_GROUPS)
#define AURORA_GFX_DEBUG_GROUPS
#endif

void push_debug_group(const char* label);
void pop_debug_group();

typedef struct {
  uint32_t queuedPipelines;
  uint32_t createdPipelines;
  uint32_t drawCallCount;
  uint32_t mergedDrawCallCount;
  uint32_t lastVertSize;
  uint32_t lastUniformSize;
  uint32_t lastIndexSize;
  uint32_t lastStorageSize;
  uint32_t lastTextureUploadSize;
  // Pipelines a draw has asked for that are still compiling. Unlike queuedPipelines,
  // this leaves out the startup warmup of every pipeline in the pipeline cache.
  uint32_t requestedPipelines;
} AuroraStats;

const AuroraStats* aurora_get_stats();
float aurora_get_fps();

void aurora_enable_vsync(bool enabled);

// Frame replay, for frame rates above the game's own (skies-eternal, decision 0019 fork
// change; off by default, and nothing below runs until aurora_replay_enable(true)).
//
// aurora_replay_begin_tick, right after aurora_begin_frame, starts keeping the frame's GX
// stream and copies the decoder's state. aurora_replay_close_tick, after the frame's last
// GX command, drains the FIFO and stops keeping. aurora_end_frame_hidden then ends the
// frame without presenting it and pairs each draw with the previous tick's. Each
// aurora_replay_draw(blend), inside a frame of its own (aurora_begin_frame ...
// aurora_end_frame), decodes the kept stream again from the copied state with every draw's
// matrices blended from the previous tick (0) to this one (1), skipping draw-done,
// depth snapshots and evictions; false if it could not. aurora_replay_end_tick, after the
// last of them, puts back the state the tick ended with.
typedef struct {
  uint32_t records;  // distinct matrix sets the last closed tick drew with
  uint32_t matched;  // of those, paired with one from the tick before
  uint32_t rejected; // paired, but moved too far to blend
  bool cut;          // the tick looked like a camera cut and is not blended
  uint32_t bytes;    // size of the kept stream
  uint32_t blended;  // draws blended, all replays so far
  uint32_t unknown;  // replayed uniform builds with no record, all replays so far
  uint32_t replays;  // replays so far
  // Draws with positions in the stream (the game transformed them on the CPU): their
  // positions are blended the same way, pairing each with the previous tick's draw of the
  // same format, primitive, vertex count and texture coordinates, nearest first.
  uint32_t draws;         // such draws in the last closed tick
  uint32_t drawsMatched;  // of those, paired with one from the tick before
  uint32_t drawsRejected; // paired, but moved too far to blend
  uint32_t drawsBlended;  // draws whose positions were blended, all replays so far
  uint32_t ticksSkipped;  // ticks not replayed because something drew outside the stream, so far
  // Of the paired position draws of the last closed tick, those that turned as a rigid body
  // (a propeller): blended along the turn rather than straight across it.
  uint32_t drawsRigid;
  // Uniform builds whose colours, fog or texture matrices were blended, all replays so far
  // (each of those is also in blended).
  uint32_t looksBlended;
  // Position draws whose vertex colours were blended (the screen fade), all replays so far.
  uint32_t colorsBlended;
  // Of the paired position draws of the last closed tick, by how their texture coordinates in
  // the stream changed: moved a little (a scroll, blended), stepped by a cell or more (a
  // flipbook, not blended), or drew another texture (a flipbook of textures); and of the
  // stepped or swapped ones, those drawn additively (a glow, a flame).
  uint32_t drawsUvMoved;
  uint32_t drawsUvStepped;
  uint32_t drawsTexSwapped;
  uint32_t drawsStepAdditive;
  // Draws of the last closed tick with texture coordinates in the stream but indexed
  // positions: not paired, so their coordinates are never blended.
  uint32_t drawsUvUnpaired;
  // Position draws whose texture coordinates were blended, all replays so far.
  uint32_t uvsBlended;
  // Indexed texture coordinate arrays the game rewrote between ticks (Noxus's vortex) and
  // that were blended, all replays so far.
  uint32_t uvArraysBlended;
  // Flipbook draws drawn twice, the previous tick's cell and this one's cross-faded
  // (crossfade=1), all replays so far.
  uint32_t crossfaded;
  // Marks (aurora_replay_mark) whose callback a replay ran, all replays so far.
  uint32_t marksReplayed;
} AuroraReplayStats;

void aurora_replay_enable(bool enabled);
void aurora_replay_begin_tick();
void aurora_replay_close_tick();
void aurora_end_frame_hidden();
bool aurora_replay_draw(float blend);
void aurora_replay_end_tick();
// Whether the last closed tick's first decode read game memory overlapping [data, data +
// size) (vertex arrays, indexed matrix loads, textures, TLUTs), so its replays would read
// it again: a block the game freed during the tick matters only if this is true.
bool aurora_replay_reads(const void* data, size_t size);
// Something drew into the tick being kept by calling the renderer directly (aurora::gfx::
// create_pass, resolve_pass, push_custom_draw: the port's padded texture copy) rather than
// through the GX stream. A replay decodes only the stream, so it would miss that drawing
// and could rebuild a copied texture without it: the tick is not replayed, and is shown as
// it is.
void aurora_replay_skip_tick();
// A pass the game's side draws by calling the renderer directly (aurora::gfx::resolve_pass,
// push_encoder_task, push_custom_draw: the port's cel-shading outlines and bloom), at a
// point of the tick that the replays should repeat. Called just before that drawing, while
// a tick is kept, it drains the FIFO and marks the stream there with `callback` and a copy
// of `size` bytes at `data` (at most 256); each replay calls callback(copy, size) when its
// decode reaches the mark, to draw the pass again over the replayed picture. Outside a kept
// tick it does nothing. The callback must draw only from the renderer and its copy, not
// from game state, which has moved on by the time a replay runs.
typedef void (*AuroraReplayMarkCallback)(const void* data, size_t size);
void aurora_replay_mark(AuroraReplayMarkCallback callback, const void* data, size_t size);
// maxturn, jumprel, jumpabs, cutmin, cutfraction, nearestmax, blend (0 draws replays unblended),
// vertices (0 leaves positions in the stream unblended), rotation (0 blends matrices element by
// element), rigid (the residual a turning draw may have, 0 never turns), rigidmin (degrees),
// colors (0 leaves colours, fog and texture matrices unblended), colormax, texjump (also for
// texture coordinates in the stream), uvs (0 leaves texture coordinates in the stream
// unblended), crossfade (1 cross-fades additive flipbooks; off by default), debug.
bool aurora_replay_set_tunable(const char* name, float value);
const AuroraReplayStats* aurora_replay_stats();

#ifdef __cplusplus
}
#endif

#endif
