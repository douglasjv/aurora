// Frame replay for high frame rates. See replay.hpp and aurora_replay_* in
// include/aurora/gfx.h; the design is skies-eternal's docs/design/high-frame-rate.md.
#include "replay.hpp"

#include "command_processor.hpp"
#include "fifo.hpp"
#include "gx.hpp"
#include "texture.hpp"

#include <aurora/aurora.h>
#include <aurora/gfx.h>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>

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

// A draw with its positions in the stream (CPU-transformed geometry), as the first decode
// saw it.
struct DrawKey {
  u32 fmt = 0;
  u32 prim = 0;
  u32 vtxCount = 0;
  u32 vtxSize = 0;
  u64 uvHash = 0; // the texture coordinates' bytes (or their indices): same model, same strip

  bool operator==(const DrawKey& rhs) const = default;

  template <typename H>
  friend H AbslHashValue(H h, const DrawKey& key) {
    return H::combine(std::move(h), key.fmt, key.prim, key.vtxCount, key.vtxSize, key.uvHash);
  }
};

struct DrawRecord {
  DrawKey key;
  u32 posOffset = 0; // within a vertex
  u8 posType = 0;    // GXCompType
  u8 posFrac = 0;
  u8 posCount = 0;   // 2 or 3
  u32 first = 0;     // first position in sPositions (or sPrevPositions), 3 floats a vertex
  float centroid[3] = {};
  float radius = 0.f;
  s32 match = -1;
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
  // A draw looks at most this many unpaired candidates for the nearest.
  float nearestMax = 2048.f;
  // 0 turns blending off (replays draw the tick as it is).
  float blend = 1.f;
  // Non-zero logs each tick's matching.
  float debug = 0.f;
  // 0 leaves positions in the stream (CPU-transformed draws) unblended.
  float vertices = 1.f;
};

bool sEnabled = false;
bool sCapturing = false;
bool sRecording = false;
bool sReplaying = false;
bool sFailed = false;
bool sReady = false; // a closed tick waits to be replayed
bool sSkip = false;  // the kept tick drew outside the stream (aurora_replay_skip_tick)
bool sCut = false;
bool sSwapped = false;
float sBlend = 1.f;
Tunables sTunables;
AuroraReplayStats sStats{};

std::vector<u8> sCapture;
std::vector<Record> sCurrent;
std::vector<Record> sPrevious;
std::vector<DrawRecord> sDraws;
std::vector<DrawRecord> sPrevDraws;
std::vector<float> sPositions;
std::vector<float> sPrevPositions;
u32 sDrawIndex = 0; // the next direct-position draw of the decode
std::vector<u8> sVertexScratch;
// The first decode's records by the hash of their key and matrices: a replay recomputes
// the hash from the same state and finds its record without relying on build order.
absl::flat_hash_map<u64, u32> sCurrentByHash;
// Game memory the tick's first decode read, as [begin, end) ranges: noted while
// recording (deduplicated), sorted and merged when the tick closes.
std::vector<std::pair<uintptr_t, uintptr_t>> sReads;
absl::flat_hash_set<std::pair<uintptr_t, uintptr_t>> sReadSet;
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
  std::memcpy(static_cast<void*>(&out), fa, sizeof(M));
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

float read_comp(const u8* p, u8 type, u8 frac) noexcept {
  switch (type) {
  case GX_U8:
    return std::ldexp(static_cast<float>(*p), -frac);
  case GX_S8:
    return std::ldexp(static_cast<float>(static_cast<s8>(*p)), -frac);
  case GX_U16:
    return std::ldexp(static_cast<float>(static_cast<u16>(p[0] << 8 | p[1])), -frac);
  case GX_S16:
    return std::ldexp(static_cast<float>(static_cast<s16>(p[0] << 8 | p[1])), -frac);
  case GX_F32: {
    const u32 bits = static_cast<u32>(p[0]) << 24 | static_cast<u32>(p[1]) << 16 | static_cast<u32>(p[2]) << 8 | p[3];
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
  }
  default:
    return 0.f;
  }
}

void write_comp(u8* p, u8 type, u8 frac, float value) noexcept {
  const auto fixed = [&](float lo, float hi) {
    return static_cast<long>(std::lround(std::clamp(std::ldexp(value, frac), lo, hi)));
  };
  switch (type) {
  case GX_U8:
    *p = static_cast<u8>(fixed(0.f, 255.f));
    break;
  case GX_S8:
    *p = static_cast<u8>(static_cast<s8>(fixed(-128.f, 127.f)));
    break;
  case GX_U16: {
    const auto v = static_cast<u16>(fixed(0.f, 65535.f));
    p[0] = static_cast<u8>(v >> 8);
    p[1] = static_cast<u8>(v);
    break;
  }
  case GX_S16: {
    const auto v = static_cast<u16>(static_cast<s16>(fixed(-32768.f, 32767.f)));
    p[0] = static_cast<u8>(v >> 8);
    p[1] = static_cast<u8>(v);
    break;
  }
  case GX_F32: {
    u32 bits;
    std::memcpy(&bits, &value, 4);
    p[0] = static_cast<u8>(bits >> 24);
    p[1] = static_cast<u8>(bits >> 16);
    p[2] = static_cast<u8>(bits >> 8);
    p[3] = static_cast<u8>(bits);
    break;
  }
  default:
    break;
  }
}

// Where a format's position sits in a vertex, when the stream holds it, and where its
// texture coordinates start. False when the position is indexed (or absent).
bool position_layout(GXVtxFmt fmt, DrawRecord& r, u32& uvOffset) noexcept {
  const auto& vtxFmt = g_gxState.vtxFmts[fmt];
  u32 offset = 0;
  bool direct = false;
  uvOffset = UINT32_MAX;
  for (int i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7; ++i) {
    const auto attr = static_cast<GXAttr>(i);
    const auto& attrFmt = vtxFmt.attrs[i];
    const auto desc = g_gxState.vtxDesc[i];
    if (desc == GX_NONE) {
      continue;
    }
    if (i >= GX_VA_TEX0 && uvOffset == UINT32_MAX) {
      uvOffset = offset;
    }
    u32 size;
    if (desc == GX_DIRECT) {
      size = comp_type_size(attr, attrFmt.type) * comp_cnt_count(attr, attrFmt.cnt);
    } else {
      size = desc == GX_INDEX8 ? 1 : 2;
      if (i == GX_VA_NRM && attrFmt.cnt == GX_NRM_NBT3) {
        size *= 3;
      }
    }
    if (i == GX_VA_POS && desc == GX_DIRECT) {
      direct = true;
      r.posOffset = offset;
      r.posType = static_cast<u8>(attrFmt.type);
      r.posFrac = attrFmt.frac;
      r.posCount = comp_cnt_count(attr, attrFmt.cnt);
    }
    offset += size;
  }
  return direct && (r.posCount == 2 || r.posCount == 3);
}

float draw_distance2(const DrawRecord& a, const DrawRecord& b) noexcept {
  const float dx = a.centroid[0] - b.centroid[0];
  const float dy = a.centroid[1] - b.centroid[1];
  const float dz = a.centroid[2] - b.centroid[2];
  return dx * dx + dy * dy + dz * dz;
}

// Whether a matched pair of draws is the same thing having moved, not a jump: its centre
// moved less than jumprel of its distance from the origin (the camera, for the view-space
// positions the CPU transform makes) plus jumpabs, and its size changed by less than half.
bool draw_close_enough(const DrawRecord& prev, const DrawRecord& cur) noexcept {
  const auto length = [](const float* c) { return std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]); };
  const float jump = std::sqrt(draw_distance2(prev, cur));
  if (jump > sTunables.jumpAbs + sTunables.jumpRel * std::max(length(prev.centroid), length(cur.centroid))) {
    return false;
  }
  const float big = std::max(prev.radius, cur.radius);
  return big < 1e-6f || std::min(prev.radius, cur.radius) >= 0.5f * big;
}

// What two draws must share to be paired: everything in the key but the texture
// coordinates, which scroll on some meshes (the world map's vortex) from tick to tick.
struct ShapeKey {
  u32 fmt, prim, vtxCount, vtxSize;
  bool operator==(const ShapeKey& rhs) const = default;
  template <typename H>
  friend H AbslHashValue(H h, const ShapeKey& key) {
    return H::combine(std::move(h), key.fmt, key.prim, key.vtxCount, key.vtxSize);
  }
};

ShapeKey shape(const DrawKey& k) noexcept { return {k.fmt, k.prim, k.vtxCount, k.vtxSize}; }

// Pairs this tick's position draws with the previous tick's. The CPU draws a mesh's strips
// in the same order every tick, and many strips look alike (same shape, same texture
// coordinates: a wall's panels), sitting closer together than they move in a tick, so
// nearness alone pairs them wrongly. The two ticks' draws are aligned as a diff aligns two
// texts: draws whose key appears once in each tick are anchors, and pairs grow forward and
// backward from them while the keys agree; then again allowing other texture coordinates
// (meshes whose coordinates scroll); what is left pairs with the nearest unpaired draw of the
// same shape, preferring the same coordinates.
void match_draws(u32& matched, u32& rejected) noexcept {
  const size_t n = sDraws.size();
  const size_t m = sPrevDraws.size();
  std::vector<s32> pair(n, -1);
  std::vector<bool> used(m, false);
  for (auto& r : sDraws) {
    r.match = -1;
  }
  const auto take = [&](size_t i, size_t j) {
    pair[i] = static_cast<s32>(j);
    used[j] = true;
  };
  // Anchors: keys found exactly once in each tick.
  absl::flat_hash_map<DrawKey, std::pair<s32, s32>> counts; // key -> (index or -2 if repeated) in cur, prev
  for (size_t i = 0; i < n; ++i) {
    auto [it, inserted] = counts.try_emplace(sDraws[i].key, static_cast<s32>(i), -1);
    if (!inserted) {
      it->second.first = -2;
    }
  }
  for (size_t j = 0; j < m; ++j) {
    const auto it = counts.find(sPrevDraws[j].key);
    if (it != counts.end()) {
      it->second.second = it->second.second == -1 ? static_cast<s32>(j) : -2;
    }
  }
  for (const auto& [key, at] : counts) {
    if (at.first >= 0 && at.second >= 0 && draw_close_enough(sPrevDraws[at.second], sDraws[at.first])) {
      take(static_cast<size_t>(at.first), static_cast<size_t>(at.second));
    }
  }
  // Grow runs from every pair, forward and backward, while the draws agree.
  const auto grow = [&](bool sameUv) {
    const auto agrees = [&](size_t i, size_t j) {
      const auto& a = sDraws[i];
      const auto& b = sPrevDraws[j];
      return pair[i] < 0 && !used[j] && shape(a.key) == shape(b.key) && (!sameUv || a.key.uvHash == b.key.uvHash) &&
             draw_close_enough(b, a);
    };
    for (size_t i = 0; i < n; ++i) {
      if (pair[i] < 0) {
        continue;
      }
      for (size_t a = i + 1, b = static_cast<size_t>(pair[i]) + 1; a < n && b < m && agrees(a, b); ++a, ++b) {
        take(a, b);
      }
    }
    for (size_t i = n; i-- > 0;) {
      if (pair[i] < 0) {
        continue;
      }
      for (size_t a = i, b = static_cast<size_t>(pair[i]); a > 0 && b > 0 && agrees(a - 1, b - 1); --a, --b) {
        take(a - 1, b - 1);
      }
    }
  };
  grow(true);
  grow(false);
  // The rest: the nearest unpaired draw of the same shape, the same coordinates first.
  absl::flat_hash_map<ShapeKey, std::vector<u32>> byShape;
  for (u32 j = 0; j < m; ++j) {
    if (!used[j]) {
      byShape[shape(sPrevDraws[j].key)].push_back(j);
    }
  }
  u32 drawsRejected = 0;
  for (size_t i = 0; i < n; ++i) {
    if (pair[i] >= 0) {
      continue;
    }
    const auto& r = sDraws[i];
    const auto it = byShape.find(shape(r.key));
    if (it == byShape.end()) {
      continue;
    }
    float bestSame = std::numeric_limits<float>::max();
    float bestAny = std::numeric_limits<float>::max();
    s32 same = -1;
    s32 any = -1;
    u32 looked = 0;
    for (const u32 idx : it->second) {
      if (used[idx]) {
        continue;
      }
      if (++looked > static_cast<u32>(sTunables.nearestMax)) {
        break;
      }
      const float d = draw_distance2(sPrevDraws[idx], r);
      if (sPrevDraws[idx].key.uvHash == r.key.uvHash && d < bestSame) {
        bestSame = d;
        same = static_cast<s32>(idx);
      }
      if (d < bestAny) {
        bestAny = d;
        any = static_cast<s32>(idx);
      }
    }
    const s32 best = same >= 0 && draw_close_enough(sPrevDraws[same], r) ? same : any;
    if (best < 0) {
      continue;
    }
    used[best] = true;
    if (!draw_close_enough(sPrevDraws[best], r)) {
      ++drawsRejected;
      continue;
    }
    pair[i] = best;
  }
  u32 drawsMatched = 0;
  for (size_t i = 0; i < n; ++i) {
    sDraws[i].match = pair[i];
    drawsMatched += pair[i] >= 0;
  }
  sStats.draws = static_cast<u32>(n);
  sStats.drawsMatched = drawsMatched;
  sStats.drawsRejected = drawsRejected;
  matched += drawsMatched;
  rejected += drawsRejected;
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
  if (sTunables.debug != 0.f) {
    float moved = 0.f;
    float projMoved = 0.f;
    for (const auto& r : sCurrent) {
      if (r.match >= 0) {
        moved += std::sqrt(distance2(sPrevious[r.match], r));
        const auto pa = flat(sPrevious[r.match].m.pos[0]);
        (void)pa;
        float d = 0.f;
        const float* a = reinterpret_cast<const float*>(&sPrevious[r.match].m.proj);
        const float* b = reinterpret_cast<const float*>(&r.m.proj);
        for (int i = 0; i < 16; ++i) d += std::abs(a[i] - b[i]);
        projMoved += d;
      }
    }
    Log.info("match: {} records, {} previous, {} matched, {} rejected; matched translations moved {:.3f} in all, projections {:.4f}",
             sCurrent.size(), sPrevious.size(), matched, rejected, moved, projMoved);
    static u32 detailTicks = 0;
    if (sTunables.debug >= 2.f && ++detailTicks % static_cast<u32>(sTunables.debug) == 0) {
      for (size_t i = 0; i < sCurrent.size(); ++i) {
        const auto& r = sCurrent[i];
        const auto f = flat(main_matrix(r));
        const float dist = std::sqrt(f[3] * f[3] + f[7] * f[7] + f[11] * f[11]);
        float moved1 = -1.f;
        float turn = 0.f;
        if (r.match >= 0) {
          moved1 = std::sqrt(distance2(sPrevious[r.match], r));
          const auto g = flat(main_matrix(sPrevious[r.match]));
          for (int c = 0; c < 3; ++c) {
            const float d = f[c] * g[c] + f[4 + c] * g[4 + c] + f[8 + c] * g[8 + c];
            const float la = std::sqrt(f[c] * f[c] + f[4 + c] * f[4 + c] + f[8 + c] * f[8 + c]);
            const float lb = std::sqrt(g[c] * g[c] + g[4 + c] * g[4 + c] + g[8 + c] * g[8 + c]);
            if (la > 1e-6f && lb > 1e-6f) {
              turn = std::max(turn, std::acos(std::clamp(d / (la * lb), -1.f, 1.f)) * 57.29578f);
            }
          }
        }
        Log.info("  rec {}: pos {} desc {} fmt {} prim {} n {} mtx {} proj {}; dist {:.2f} moved {:.3f} turn {:.2f}deg",
                 i, r.key.posArray, r.key.posDesc, r.key.fmt, r.key.prim, r.key.vtxCount, r.key.currentPnMtx,
                 static_cast<int>(r.m.projType), dist, moved1, turn);
      }
    }
  }
  match_draws(matched, rejected);
  const u32 paired = matched + rejected;
  sCut = paired >= static_cast<u32>(sTunables.cutMin) &&
         static_cast<float>(rejected) > sTunables.cutFraction * static_cast<float>(paired);
  if (sTunables.debug != 0.f) {
    Log.info("tick: {} position draws, {} paired, {} too far; {} matrix sets; {} paired in all, {} too far{}",
             sDraws.size(), sStats.drawsMatched, sStats.drawsRejected, sCurrent.size(), matched, rejected,
             sCut ? "; CUT" : "");
  }
  sStats.records = static_cast<u32>(sCurrent.size());
  sStats.matched = matched - sStats.drawsMatched;
  sStats.rejected = rejected - sStats.drawsRejected;
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

void note_texture(const GXTexObj_& obj) noexcept {
  if (obj.has_data()) {
    note_read(obj.data, texture::texture_source_size(obj.format(), obj.width(), obj.height(), obj.mip_count()));
  }
}

void note_tlut(const GXTlutObj_& tlut) noexcept {
  if (reinterpret_cast<uintptr_t>(tlut.data) >= 0x10000) {
    note_read(tlut.data, texture::tlut_source_size(tlut.numEntries));
  }
}

// What the tick can read without loading it first: the arrays, textures and TLUTs in
// the state it starts from.
void note_start_state() noexcept {
  for (const auto& array : g_gxState.arrays) {
    note_read(array.data, array.size);
  }
  for (const auto& obj : g_gxState.loadedTextures) {
    note_texture(obj);
  }
  for (const auto& tlut : g_gxState.loadedTluts) {
    note_tlut(tlut);
  }
}

void merge_reads() noexcept {
  std::sort(sReads.begin(), sReads.end());
  size_t out = 0;
  for (const auto& range : sReads) {
    if (out != 0 && range.first <= sReads[out - 1].second) {
      sReads[out - 1].second = std::max(sReads[out - 1].second, range.second);
    } else {
      sReads[out++] = range;
    }
  }
  sReads.resize(out);
  sReadSet.clear();
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

void note_read(const void* data, size_t size) noexcept {
  if (!sRecording || data == nullptr || size == 0) {
    return;
  }
  const auto begin = reinterpret_cast<uintptr_t>(data);
  const std::pair<uintptr_t, uintptr_t> range{begin, begin + size};
  if (sReadSet.insert(range).second) {
    sReads.push_back(range);
  }
}

void fail(const char* why) noexcept {
  if (!sFailed) {
    Log.warn("replay stopped: {}", why);
  }
  sFailed = true;
}

bool failed() noexcept { return sFailed; }

const u8* vertices(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, u32 vtxSize, const u8* data) noexcept {
  if (!sRecording && !sReplaying) {
    return data;
  }
  DrawRecord probe;
  u32 uvOffset;
  if (vtxCount == 0 || vtxSize == 0 || !position_layout(fmt, probe, uvOffset)) {
    return data;
  }
  const u32 index = sDrawIndex++;
  if (sRecording) {
    probe.key = DrawKey{
        .fmt = static_cast<u32>(fmt),
        .prim = static_cast<u32>(prim),
        .vtxCount = vtxCount,
        .vtxSize = vtxSize,
    };
    u64 h = 0xCBF29CE484222325ull;
    if (uvOffset < vtxSize) {
      for (u32 v = 0; v < vtxCount; ++v) {
        h = fnv(h, data + v * vtxSize + uvOffset, vtxSize - uvOffset);
      }
    }
    probe.key.uvHash = h;
    probe.first = static_cast<u32>(sPositions.size());
    float lo[3] = {INFINITY, INFINITY, INFINITY};
    float hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    const u8 size = probe.posType == GX_F32 ? 4 : probe.posType >= GX_U16 ? 2 : 1;
    for (u32 v = 0; v < vtxCount; ++v) {
      const u8* p = data + v * vtxSize + probe.posOffset;
      for (u32 c = 0; c < 3; ++c) {
        const float value = c < probe.posCount ? read_comp(p + c * size, probe.posType, probe.posFrac) : 0.f;
        sPositions.push_back(value);
        lo[c] = std::min(lo[c], value);
        hi[c] = std::max(hi[c], value);
      }
    }
    float r2 = 0.f;
    for (u32 c = 0; c < 3; ++c) {
      probe.centroid[c] = 0.5f * (lo[c] + hi[c]);
      r2 += 0.25f * (hi[c] - lo[c]) * (hi[c] - lo[c]);
    }
    probe.radius = std::sqrt(r2);
    sDraws.push_back(probe);
    return data;
  }
  // Replay: the same stream from the same state, so the same draw at the same index.
  if (sCut || sTunables.blend == 0.f || sTunables.vertices == 0.f || index >= sDraws.size()) {
    return data;
  }
  const DrawRecord& cur = sDraws[index];
  if (cur.key.fmt != static_cast<u32>(fmt) || cur.key.prim != static_cast<u32>(prim) ||
      cur.key.vtxCount != vtxCount || cur.key.vtxSize != vtxSize || cur.match < 0) {
    return data;
  }
  const DrawRecord& prev = sPrevDraws[cur.match];
  sVertexScratch.assign(data, data + static_cast<size_t>(vtxCount) * vtxSize);
  const float t = sBlend;
  const u8 size = cur.posType == GX_F32 ? 4 : cur.posType >= GX_U16 ? 2 : 1;
  for (u32 v = 0; v < vtxCount; ++v) {
    u8* p = sVertexScratch.data() + v * vtxSize + cur.posOffset;
    const float* a = &sPrevPositions[prev.first + v * 3];
    const float* b = &sPositions[cur.first + v * 3];
    for (u32 c = 0; c < cur.posCount; ++c) {
      write_comp(p + c * size, cur.posType, cur.posFrac, a[c] + (b[c] - a[c]) * t);
    }
  }
  ++sStats.drawsBlended;
  return sVertexScratch.data();
}

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
    sDraws = {};
    sPrevDraws = {};
    sPositions = {};
    sPrevPositions = {};
    sReads = {};
    sReadSet = {};
    sStart.reset();
    sEnd.reset();
  }
}

void aurora_replay_begin_tick() {
  if (!sEnabled) {
    return;
  }
  sReady = false;
  sSkip = false;
  sPrevious = std::move(sCurrent);
  sCurrent.clear();
  sPrevDraws = std::move(sDraws);
  sDraws.clear();
  sPrevPositions = std::move(sPositions);
  sPositions.clear();
  sDrawIndex = 0;
  sCurrentByHash.clear();
  sCapture.clear();
  sReads.clear();
  sReadSet.clear();
  sStart = snapshot();
  fifo::save_draw_cache(0);
  sCapturing = true;
  sRecording = true;
  note_start_state();
}

void aurora_replay_close_tick() {
  if (!sCapturing) {
    return;
  }
  fifo::drain();
  sCapturing = false;
  sRecording = false;
  merge_reads();
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
  if (!sReady || sSkip) {
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
  sDrawIndex = 0;
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

void aurora_replay_skip_tick() {
  if (sRecording && !sSkip) {
    sSkip = true;
    ++sStats.ticksSkipped;
  }
}

bool aurora_replay_reads(const void* data, size_t size) {
  const auto begin = reinterpret_cast<uintptr_t>(data);
  const auto end = begin + size;
  // The first merged range ending after begin: it overlaps when it starts before end.
  const auto it = std::upper_bound(sReads.begin(), sReads.end(), begin,
                                   [](uintptr_t value, const auto& range) { return value < range.second; });
  return it != sReads.end() && it->first < end;
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
  } else if (n == "vertices") {
    sTunables.vertices = value;
  } else if (n == "debug") {
    sTunables.debug = value;
  } else if (n == "blend") {
    sTunables.blend = value;
  } else {
    return false;
  }
  return true;
}

const AuroraReplayStats* aurora_replay_stats() { return &sStats; }
