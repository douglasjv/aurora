// Frame replay for high frame rates. See replay.hpp and aurora_replay_* in
// include/aurora/gfx.h; the design is skies-eternal's docs/design/high-frame-rate.md.
#include "replay.hpp"

#include "command_processor.hpp"
#include "fifo.hpp"
#include "gx.hpp"

#include <aurora/aurora.h>
#include <aurora/gfx.h>

#include <absl/container/flat_hash_map.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <string_view>
#include <vector>

namespace aurora::gx::replay {
namespace {
constexpr Module Log{"aurora::gx::replay"};

// What a draw is, as opposed to where it is: the same model has the same key from one
// tick to the next.
struct Key {
  const void* posArray = nullptr; // indexed position data, null for direct positions
  u32 posDesc = 0;
  u32 fmt = 0;
  u32 prim = 0;
  u32 vtxCount = 0;
  u32 currentPnMtx = 0;

  bool operator==(const Key& rhs) const = default;

  template <typename H>
  friend H AbslHashValue(H h, const Key& key) {
    return H::combine(std::move(h), key.posArray, key.posDesc, key.fmt, key.prim, key.vtxCount, key.currentPnMtx);
  }
};

struct Matrices {
  std::array<Mat3x4<float>, MaxPnMtx> pos;
  std::array<Mat3x4<float>, MaxPnMtx> nrm;
  Mat4x4<float> proj;
  GXProjectionType projType;
};

// One uniform build of the first decode.
struct Record {
  Key key;
  Matrices m;
  s32 match = -1; // index of the previous tick's record it blends from
};

struct Tunables {
  // A match whose main matrix turns a basis vector by more than this many degrees is
  // drawn unblended.
  float maxTurn = 45.f;
  // ... or whose translation jumps by more than this fraction of its distance from the
  // camera, plus jumpAbs.
  float jumpRel = 0.5f;
  float jumpAbs = 0.f;
  // A tick where at least cutMin draws matched and more than cutFraction of them were
  // turned away is a camera cut: nothing in it is blended.
  float cutMin = 8.f;
  float cutFraction = 0.5f;
  // Key groups larger than this match in order instead of by nearest position.
  float nearestMax = 256.f;
  // 0 turns blending off (replays draw the tick as it is).
  float blend = 1.f;
};

bool sEnabled = false;
bool sCapturing = false;
bool sRecording = false;
bool sReplaying = false;
bool sFailed = false;
bool sReady = false; // a closed tick waits to be replayed
bool sCut = false;
bool sSwapped = false;
float sBlend = 1.f;
Tunables sTunables;
AuroraReplayStats sStats{};

std::vector<u8> sCapture;
std::vector<Record> sCurrent;
std::vector<Record> sPrevious;
// The first decode's records by the hash of their key and matrices: a replay recomputes
// the hash from the same state and finds its record without relying on build order.
absl::flat_hash_map<u64, u32> sCurrentByHash;
std::unique_ptr<GXState> sStart;
std::unique_ptr<GXState> sEnd;
Matrices sSaved;

u64 fnv(u64 h, const void* data, size_t size) noexcept {
  const auto* bytes = static_cast<const u8*>(data);
  for (size_t i = 0; i < size; ++i) {
    h ^= bytes[i];
    h *= 0x100000001B3ull;
  }
  return h;
}

Key current_key(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount) noexcept {
  const auto& state = g_gxState;
  const auto posDesc = state.vtxDesc[GX_VA_POS];
  const bool indexed = posDesc == GX_INDEX8 || posDesc == GX_INDEX16;
  return Key{
      .posArray = indexed ? state.arrays[GX_VA_POS].data : nullptr,
      .posDesc = static_cast<u32>(posDesc),
      .fmt = static_cast<u32>(fmt),
      .prim = static_cast<u32>(prim),
      .vtxCount = vtxCount,
      .currentPnMtx = state.currentPnMtx,
  };
}

u64 state_hash(const Key& key) noexcept {
  const auto& state = g_gxState;
  u64 h = 0xCBF29CE484222325ull;
  h = fnv(h, &key.posArray, sizeof(key.posArray));
  h = fnv(h, &key.posDesc, sizeof(u32) * 5);
  for (const auto& mtx : state.pnMtx) {
    h = fnv(h, &mtx.pos, sizeof(mtx.pos));
  }
  h = fnv(h, &state.proj, sizeof(state.proj));
  h = fnv(h, &state.projType, sizeof(state.projType));
  return h;
}

void save_matrices(Matrices& out) noexcept {
  const auto& state = g_gxState;
  for (u32 i = 0; i < MaxPnMtx; ++i) {
    out.pos[i] = state.pnMtx[i].pos;
    out.nrm[i] = state.pnMtx[i].nrm;
  }
  out.proj = state.proj;
  out.projType = state.projType;
}

void load_matrices(const Matrices& in) noexcept {
  auto& state = g_gxState;
  for (u32 i = 0; i < MaxPnMtx; ++i) {
    state.pnMtx[i].pos = in.pos[i];
    state.pnMtx[i].nrm = in.nrm[i];
  }
  state.proj = in.proj;
  state.projType = in.projType;
}

template <typename M>
M lerp(const M& a, const M& b, float t) noexcept {
  static_assert(sizeof(M) % sizeof(float) == 0);
  constexpr size_t N = sizeof(M) / sizeof(float);
  float fa[N];
  float fb[N];
  std::memcpy(fa, &a, sizeof(M));
  std::memcpy(fb, &b, sizeof(M));
  for (size_t i = 0; i < N; ++i) {
    fa[i] += (fb[i] - fa[i]) * t;
  }
  M out;
  std::memcpy(&out, fa, sizeof(M));
  return out;
}

// A 3x4 matrix as twelve floats, row major: rows x, y, z, each three basis components
// and a translation.
std::array<float, 12> flat(const Mat3x4<float>& m) noexcept {
  std::array<float, 12> f;
  static_assert(sizeof(Mat3x4<float>) == sizeof(float) * 12);
  std::memcpy(f.data(), &m, sizeof(f));
  return f;
}

const Mat3x4<float>& main_matrix(const Record& r) noexcept {
  return r.m.pos[std::min<u32>(r.key.currentPnMtx, MaxPnMtx - 1)];
}

float distance2(const Record& a, const Record& b) noexcept {
  const auto fa = flat(main_matrix(a));
  const auto fb = flat(main_matrix(b));
  const float dx = fa[3] - fb[3];
  const float dy = fa[7] - fb[7];
  const float dz = fa[11] - fb[11];
  return dx * dx + dy * dy + dz * dz;
}

// Whether a matched pair is the same thing having moved, not a jump.
bool close_enough(const Record& prev, const Record& cur) noexcept {
  if (prev.m.projType != cur.m.projType) {
    return false;
  }
  const auto a = flat(main_matrix(prev));
  const auto b = flat(main_matrix(cur));
  const float minCos = std::cos(sTunables.maxTurn * 3.14159265f / 180.f);
  for (int c = 0; c < 3; ++c) {
    const float ax = a[c], ay = a[4 + c], az = a[8 + c];
    const float bx = b[c], by = b[4 + c], bz = b[8 + c];
    const float la = std::sqrt(ax * ax + ay * ay + az * az);
    const float lb = std::sqrt(bx * bx + by * by + bz * bz);
    if (la < 1e-6f || lb < 1e-6f) {
      continue; // a collapsed axis (scaled to nothing) has no direction to compare
    }
    if ((ax * bx + ay * by + az * bz) / (la * lb) < minCos) {
      return false;
    }
  }
  const float da = std::sqrt(a[3] * a[3] + a[7] * a[7] + a[11] * a[11]);
  const float db = std::sqrt(b[3] * b[3] + b[7] * b[7] + b[11] * b[11]);
  const float jump = std::sqrt(distance2(prev, cur));
  return jump <= sTunables.jumpAbs + sTunables.jumpRel * std::max(da, db);
}

// Pairs this tick's records with the previous tick's, once, when the tick closes.
void match() noexcept {
  absl::flat_hash_map<Key, std::vector<u32>> byKey;
  for (u32 i = 0; i < sPrevious.size(); ++i) {
    byKey[sPrevious[i].key].push_back(i);
  }
  std::vector<bool> used(sPrevious.size(), false);
  absl::flat_hash_map<Key, u32> nextInOrder;
  u32 matched = 0;
  u32 rejected = 0;
  for (auto& r : sCurrent) {
    r.match = -1;
    const auto it = byKey.find(r.key);
    if (it == byKey.end()) {
      continue;
    }
    const auto& candidates = it->second;
    s32 best = -1;
    if (candidates.size() > static_cast<size_t>(sTunables.nearestMax)) {
      auto& next = nextInOrder[r.key];
      while (next < candidates.size() && used[candidates[next]]) {
        ++next;
      }
      if (next < candidates.size()) {
        best = static_cast<s32>(candidates[next]);
      }
    } else {
      float bestDistance = std::numeric_limits<float>::max();
      for (const u32 idx : candidates) {
        if (used[idx]) {
          continue;
        }
        const float d = distance2(sPrevious[idx], r);
        if (d < bestDistance) {
          bestDistance = d;
          best = static_cast<s32>(idx);
        }
      }
    }
    if (best < 0) {
      continue;
    }
    used[best] = true;
    if (!close_enough(sPrevious[best], r)) {
      ++rejected;
      continue;
    }
    r.match = best;
    ++matched;
  }
  const u32 paired = matched + rejected;
  sCut = paired >= static_cast<u32>(sTunables.cutMin) &&
         static_cast<float>(rejected) > sTunables.cutFraction * static_cast<float>(paired);
  sStats.records = static_cast<u32>(sCurrent.size());
  sStats.matched = matched;
  sStats.rejected = rejected;
  sStats.cut = sCut;
  sStats.bytes = static_cast<u32>(sCapture.size());
}

// A copy of the state as it stands between frames: nothing in it points into a frame's
// buffers.
std::unique_ptr<GXState> snapshot() {
  auto copy = std::make_unique<GXState>(g_gxState);
  for (auto& array : copy->arrays) {
    array.cachedRange = {};
  }
  return copy;
}

void decode(const std::vector<u8>& bytes) noexcept {
  size_t offset = 0;
  while (offset < bytes.size() && !sFailed) {
    const auto result = fifo::process(bytes.data() + offset, static_cast<u32>(bytes.size() - offset));
    if (result.bytesProcessed == 0) {
      fail("no progress");
      break;
    }
    offset += result.bytesProcessed; // a draw-done ends a run here and is otherwise ignored
  }
}
} // namespace

bool capturing() noexcept { return sCapturing; }

void append(const u8* data, u32 size) noexcept {
  if (sCapturing && size != 0) {
    sCapture.insert(sCapture.end(), data, data + size);
  }
}

bool replaying() noexcept { return sReplaying; }

void fail(const char* why) noexcept {
  if (!sFailed) {
    Log.warn("replay stopped: {}", why);
  }
  sFailed = true;
}

bool failed() noexcept { return sFailed; }

void before_uniform(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount) noexcept {
  if (sRecording) {
    const Key key = current_key(prim, fmt, vtxCount);
    const u64 hash = state_hash(key);
    if (!sCurrentByHash.contains(hash)) {
      Record r{.key = key};
      save_matrices(r.m);
      sCurrentByHash.emplace(hash, static_cast<u32>(sCurrent.size()));
      sCurrent.push_back(r);
    }
    return;
  }
  if (!sReplaying || sCut || sTunables.blend == 0.f) {
    return;
  }
  const Key key = current_key(prim, fmt, vtxCount);
  const auto it = sCurrentByHash.find(state_hash(key));
  if (it == sCurrentByHash.end()) {
    ++sStats.unknown;
    return;
  }
  const Record& cur = sCurrent[it->second];
  if (cur.key != key || cur.match < 0) {
    return;
  }
  const Record& prev = sPrevious[cur.match];
  save_matrices(sSaved);
  // Blend from where the draw was a tick ago (t = 0) to where it is now (t = 1).
  const float t = sBlend;
  auto& state = g_gxState;
  for (u32 i = 0; i < MaxPnMtx; ++i) {
    state.pnMtx[i].pos = lerp(prev.m.pos[i], cur.m.pos[i], t);
    state.pnMtx[i].nrm = lerp(prev.m.nrm[i], cur.m.nrm[i], t);
  }
  state.proj = lerp(prev.m.proj, cur.m.proj, t);
  sSwapped = true;
  ++sStats.blended;
}

void after_uniform() noexcept {
  if (sSwapped) {
    load_matrices(sSaved);
    sSwapped = false;
  }
}

} // namespace aurora::gx::replay

using namespace aurora::gx;
using namespace aurora::gx::replay;

void aurora_replay_enable(bool enabled) {
  sEnabled = enabled;
  if (!enabled) {
    sCapturing = sRecording = sReady = false;
    sCapture = {};
    sCurrent = {};
    sPrevious = {};
    sCurrentByHash = {};
    sStart.reset();
    sEnd.reset();
  }
}

void aurora_replay_begin_tick() {
  if (!sEnabled) {
    return;
  }
  sReady = false;
  sPrevious = std::move(sCurrent);
  sCurrent.clear();
  sCurrentByHash.clear();
  sCapture.clear();
  sStart = snapshot();
  fifo::save_draw_cache(0);
  sCapturing = true;
  sRecording = true;
}

void aurora_replay_close_tick() {
  if (!sCapturing) {
    return;
  }
  fifo::drain();
  sCapturing = false;
  sRecording = false;
}

void aurora_end_frame_hidden() {
  aurora_end_frame_present(false);
  if (!sEnabled || sStart == nullptr || sCapturing) {
    return;
  }
  sEnd = snapshot();
  fifo::save_draw_cache(1);
  match();
  sReady = true;
}

bool aurora_replay_draw(float blend) {
  if (!sReady) {
    return false;
  }
  // Commands the game wrote after the tick closed (between ticks) belong after its end
  // state: decode them there, in this frame, and take the end state again.
  if (fifo::get_buffer_size() != 0) {
    g_gxState = *sEnd;
    fifo::restore_draw_cache(1);
    fifo::drain();
    sEnd = snapshot();
    fifo::save_draw_cache(1);
  }
  g_gxState = *sStart;
  fifo::restore_draw_cache(0);
  sFailed = false;
  sBlend = std::clamp(blend, 0.f, 1.f);
  sReplaying = true;
  decode(sCapture);
  sReplaying = false;
  after_uniform();
  ++sStats.replays;
  return !sFailed;
}

void aurora_replay_end_tick() {
  if (!sReady) {
    return;
  }
  g_gxState = *sEnd;
  fifo::restore_draw_cache(1);
  sReady = false;
}

bool aurora_replay_set_tunable(const char* name, float value) {
  const std::string_view n{name};
  if (n == "maxturn") {
    sTunables.maxTurn = value;
  } else if (n == "jumprel") {
    sTunables.jumpRel = value;
  } else if (n == "jumpabs") {
    sTunables.jumpAbs = value;
  } else if (n == "cutmin") {
    sTunables.cutMin = value;
  } else if (n == "cutfraction") {
    sTunables.cutFraction = value;
  } else if (n == "nearestmax") {
    sTunables.nearestMax = value;
  } else if (n == "blend") {
    sTunables.blend = value;
  } else {
    return false;
  }
  return true;
}

const AuroraReplayStats* aurora_replay_stats() { return &sStats; }
