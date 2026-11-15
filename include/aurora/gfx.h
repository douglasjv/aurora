#ifndef AURORA_GFX_H
#define AURORA_GFX_H

#ifdef __cplusplus
#include <cstdint>

extern "C" {
#else
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
} AuroraReplayStats;

void aurora_replay_enable(bool enabled);
void aurora_replay_begin_tick();
void aurora_replay_close_tick();
void aurora_end_frame_hidden();
bool aurora_replay_draw(float blend);
void aurora_replay_end_tick();
// maxturn, jumprel, jumpabs, cutmin, cutfraction, nearestmax, blend (0 draws replays unblended).
bool aurora_replay_set_tunable(const char* name, float value);
const AuroraReplayStats* aurora_replay_stats();

#ifdef __cplusplus
}
#endif

#endif
