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

// The colour state a draw's uniforms carry, with the texture matrices: what a fade, a
// colour animation or a scrolling texture changes from tick to tick.
struct Look {
  std::array<Vec4<float>, MaxTevRegs> colorRegs;
  std::array<Vec4<float>, GX_MAX_KCOLOR> kcolors;
  std::array<Vec4<float>, MaxColorChannels> mat;
  std::array<Vec4<float>, MaxColorChannels> amb;
  Vec4<float> fogColor;
  float fogA = 0.f;
  float fogB = 0.f;
  float fogC = 0.f;
  GXFogType fogType = GX_FOG_NONE;
  std::array<Mat3x4<float>, MaxTexMtx> texMtxs;
  std::array<Mat3x4<float>, MaxPTTexMtx> ptTexMtxs;
};

// One uniform build of the first decode.
struct Record {
  Key key;
  Matrices m;
  Look look;
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
  u64 posHash = 0;   // its positions' values: a draw that did not move has the same
  u32 raw = 0;       // first byte of its vertices in sRaw (or sPrevRaw)
  u8 clrCount = 0;   // colours in the stream (GX_DIRECT), 0 to 2
  u8 clrType[2] = {};
  u32 clrOffset[2] = {};
  // Set when the tick closes, for a paired draw that turned as a rigid body from its match:
  // the turn's axis (unit), angle (radians), a point on the axis and the slide along it.
  bool rigid = false;
  float axis[3] = {};
  float angle = 0.f;
  float pivot[3] = {};
  float slide = 0.f;
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
  // turned away is a camera cut: nothing in it is blended. Through the opening and the
  // s35 sail and battles, cuts turned away 35 to 98% (two at 47 and 48%), and the fastest
  // battle-start sweep 26%: a missed cut shows the old scene smeared into the new one, a
  // false one only shows a tick at 30.
  float cutMin = 8.f;
  float cutFraction = 0.3f;
  // A draw looks at most this many unpaired candidates for the nearest.
  float nearestMax = 2048.f;
  // 0 turns blending off (replays draw the tick as it is).
  float blend = 1.f;
  // Non-zero logs each tick's matching.
  float debug = 0.f;
  // 0 leaves positions in the stream (CPU-transformed draws) unblended.
  float vertices = 1.f;
  // 0 blends matrices element by element (a turning thing shrinks halfway); otherwise the
  // 3x3 part is blended as a rotation and a stretch.
  float rotation = 1.f;
  // A paired position draw whose positions fit a rigid turn from its match, leaving a residual
  // of at most this fraction of its spread, turns along the arc; 0 never does.
  float rigid = 0.1f;
  // ... when it turned at least this many degrees (below, a straight blend is as good).
  float rigidMin = 3.f;
  // 0 leaves colours, fog and texture matrices unblended.
  float colors = 1.f;
  // A colour channel that changes by more than this (1 is full intensity) between paired draws
  // is a switch, not a fade, and is not blended.
  float colorMax = 0.25f;
  // A texture matrix whose translation moves by more than this (in texture widths, after
  // taking a wrap the short way) jumped (a flipbook) and is not blended.
  float texJump = 0.25f;
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
u32 sStillPaired = 0; // position draws of the last closed tick paired because they did not move
std::vector<u8> sRaw;     // the vertices of this tick's position draws, as the stream held them
std::vector<u8> sPrevRaw; // ... and the previous tick's
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
Look sSavedLook;

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
  // The colours too: two builds with the same matrices and other colours are two records.
  h = fnv(h, state.colorRegs.data(), sizeof(state.colorRegs));
  h = fnv(h, state.kcolors.data(), sizeof(state.kcolors));
  for (const auto& chan : state.colorChannelState) {
    h = fnv(h, &chan.matColor, sizeof(chan.matColor));
    h = fnv(h, &chan.ambColor, sizeof(chan.ambColor));
  }
  h = fnv(h, &state.fog.color, sizeof(state.fog.color));
  h = fnv(h, &state.fog.a, sizeof(float) * 3);
  h = fnv(h, state.texMtxs.data(), sizeof(state.texMtxs));
  h = fnv(h, state.ptTexMtxs.data(), sizeof(state.ptTexMtxs));
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

void save_look(Look& out) noexcept {
  const auto& state = g_gxState;
  out.colorRegs = state.colorRegs;
  out.kcolors = state.kcolors;
  for (u32 i = 0; i < MaxColorChannels; ++i) {
    out.mat[i] = state.colorChannelState[i].matColor;
    out.amb[i] = state.colorChannelState[i].ambColor;
  }
  out.fogColor = state.fog.color;
  out.fogA = state.fog.a;
  out.fogB = state.fog.b;
  out.fogC = state.fog.c;
  out.fogType = state.fog.type;
  out.texMtxs = state.texMtxs;
  out.ptTexMtxs = state.ptTexMtxs;
}

void load_look(const Look& in) noexcept {
  auto& state = g_gxState;
  state.colorRegs = in.colorRegs;
  state.kcolors = in.kcolors;
  for (u32 i = 0; i < MaxColorChannels; ++i) {
    state.colorChannelState[i].matColor = in.mat[i];
    state.colorChannelState[i].ambColor = in.amb[i];
  }
  state.fog.color = in.fogColor;
  state.fog.a = in.fogA;
  state.fog.b = in.fogB;
  state.fog.c = in.fogC;
  state.texMtxs = in.texMtxs;
  state.ptTexMtxs = in.ptTexMtxs;
}

float color_distance(const Vec4<float>& a, const Vec4<float>& b) noexcept {
  float d = 0.f;
  for (int i = 0; i < 4; ++i) {
    d += std::abs(a.m[i] - b.m[i]);
  }
  return d;
}

// How far apart two looks' colours are: only to choose between records whose matrices are
// the same (one model's materials).
float look_distance(const Look& a, const Look& b) noexcept {
  float d = 0.f;
  for (u32 i = 0; i < MaxTevRegs; ++i) {
    d += color_distance(a.colorRegs[i], b.colorRegs[i]);
  }
  for (u32 i = 0; i < GX_MAX_KCOLOR; ++i) {
    d += color_distance(a.kcolors[i], b.kcolors[i]);
  }
  for (u32 i = 0; i < MaxColorChannels; ++i) {
    d += color_distance(a.mat[i], b.mat[i]) + color_distance(a.amb[i], b.amb[i]);
  }
  return d;
}

// A colour blended from a to b, or b when a channel jumps by more than colormax: a switch
// (a flash, a value used as a flag), not a fade.
Vec4<float> blend_color(const Vec4<float>& a, const Vec4<float>& b, float t) noexcept {
  Vec4<float> out = b;
  for (int i = 0; i < 4; ++i) {
    if (std::abs(b.m[i] - a.m[i]) > sTunables.colorMax) {
      return b;
    }
    out.m[i] = a.m[i] + (b.m[i] - a.m[i]) * t;
  }
  return out;
}

// A texture matrix blended from a to b. Its translation is taken the short way round a wrap
// (a scroll from 0.95 to 0.05 moves 0.1, not 0.9), which is what a repeating texture shows;
// a move of more than texjump after that is a flipbook's jump and is not blended.
Mat3x4<float> blend_tex_mtx(const Mat3x4<float>& a, const Mat3x4<float>& b, float t) noexcept {
  if (std::memcmp(&a, &b, sizeof(a)) == 0) {
    return b;
  }
  float fa[12];
  float fb[12];
  std::memcpy(fa, &a, sizeof(fa));
  std::memcpy(fb, &b, sizeof(fb));
  float out[12];
  for (int i = 0; i < 12; ++i) {
    if (i == 3 || i == 7) {
      float delta = fb[i] - fa[i];
      delta -= std::round(delta);
      if (std::abs(delta) > sTunables.texJump) {
        return b;
      }
      out[i] = fb[i] - (1.f - t) * delta;
    } else {
      out[i] = fa[i] + (fb[i] - fa[i]) * t;
    }
  }
  Mat3x4<float> m;
  std::memcpy(static_cast<void*>(&m), out, sizeof(out));
  return m;
}

using M3 = std::array<float, 9>; // row major

float det3(const M3& m) noexcept {
  return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

// The rotation of a polar decomposition (m = R S, R orthogonal, S symmetric), by Higham's
// iteration R <- (R + R^-T) / 2. False when m is (nearly) singular.
bool polar_rotation(const M3& m, M3& r) noexcept {
  r = m;
  for (int iter = 0; iter < 30; ++iter) {
    const float d = det3(r);
    if (!std::isfinite(d) || std::abs(d) < 1e-12f) {
      return false;
    }
    // R^-T = cofactor matrix / det.
    M3 c{
        r[4] * r[8] - r[5] * r[7], r[5] * r[6] - r[3] * r[8], r[3] * r[7] - r[4] * r[6],
        r[2] * r[7] - r[1] * r[8], r[0] * r[8] - r[2] * r[6], r[1] * r[6] - r[0] * r[7],
        r[1] * r[5] - r[2] * r[4], r[2] * r[3] - r[0] * r[5], r[0] * r[4] - r[1] * r[3],
    };
    float change = 0.f;
    for (int i = 0; i < 9; ++i) {
      const float next = 0.5f * (r[i] + c[i] / d);
      change = std::max(change, std::abs(next - r[i]));
      r[i] = next;
    }
    if (change < 1e-6f) {
      break;
    }
  }
  return true;
}

// Quaternion (w, x, y, z) of a proper rotation matrix.
std::array<float, 4> to_quat(const M3& m) noexcept {
  std::array<float, 4> q;
  const float trace = m[0] + m[4] + m[8];
  if (trace > 0.f) {
    const float s = std::sqrt(trace + 1.f) * 2.f;
    q = {0.25f * s, (m[7] - m[5]) / s, (m[2] - m[6]) / s, (m[3] - m[1]) / s};
  } else if (m[0] > m[4] && m[0] > m[8]) {
    const float s = std::sqrt(1.f + m[0] - m[4] - m[8]) * 2.f;
    q = {(m[7] - m[5]) / s, 0.25f * s, (m[1] + m[3]) / s, (m[2] + m[6]) / s};
  } else if (m[4] > m[8]) {
    const float s = std::sqrt(1.f + m[4] - m[0] - m[8]) * 2.f;
    q = {(m[2] - m[6]) / s, (m[1] + m[3]) / s, 0.25f * s, (m[5] + m[7]) / s};
  } else {
    const float s = std::sqrt(1.f + m[8] - m[0] - m[4]) * 2.f;
    q = {(m[3] - m[1]) / s, (m[2] + m[6]) / s, (m[5] + m[7]) / s, 0.25f * s};
  }
  return q;
}

M3 from_quat(const std::array<float, 4>& q) noexcept {
  const float w = q[0], x = q[1], y = q[2], z = q[3];
  return {
      1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
      2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
      2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
  };
}

std::array<float, 4> slerp(std::array<float, 4> a, const std::array<float, 4>& b, float t) noexcept {
  float dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  if (dot < 0.f) {
    for (auto& v : a) {
      v = -v;
    }
    dot = -dot;
  }
  std::array<float, 4> out;
  if (dot > 0.9995f) {
    float len = 0.f;
    for (int i = 0; i < 4; ++i) {
      out[i] = a[i] + (b[i] - a[i]) * t;
      len += out[i] * out[i];
    }
    len = std::sqrt(len);
    for (auto& v : out) {
      v /= len;
    }
    return out;
  }
  const float theta = std::acos(std::min(dot, 1.f));
  const float sa = std::sin((1.f - t) * theta) / std::sin(theta);
  const float sb = std::sin(t * theta) / std::sin(theta);
  for (int i = 0; i < 4; ++i) {
    out[i] = sa * a[i] + sb * b[i];
  }
  return out;
}

// The rotation by angle about a unit axis (Rodrigues).
M3 axis_angle(const float* u, float angle) noexcept {
  const float c = std::cos(angle);
  const float s = std::sin(angle);
  const float k = 1.f - c;
  return {
      c + u[0] * u[0] * k,        u[0] * u[1] * k - u[2] * s, u[0] * u[2] * k + u[1] * s,
      u[1] * u[0] * k + u[2] * s, c + u[1] * u[1] * k,        u[1] * u[2] * k - u[0] * s,
      u[2] * u[0] * k - u[1] * s, u[2] * u[1] * k + u[0] * s, c + u[2] * u[2] * k,
  };
}

void apply(const M3& m, const float* v, float* out) noexcept {
  out[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
  out[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
  out[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
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

// A 3x4 matrix blended from a to b: the translation straight across, the 3x3 part as a
// rotation (slerp) times a stretch (lerp) from their polar decompositions, so a turning
// thing keeps its size halfway. Element by element when either part cannot be decomposed
// (singular), when one mirrors and the other does not, or with rotation=0.
Mat3x4<float> blend_matrix(const Mat3x4<float>& a, const Mat3x4<float>& b, float t) noexcept {
  if (std::memcmp(&a, &b, sizeof(a)) == 0) {
    return b;
  }
  if (sTunables.rotation == 0.f) {
    return lerp(a, b, t);
  }
  float fa[12];
  float fb[12];
  std::memcpy(fa, &a, sizeof(fa));
  std::memcpy(fb, &b, sizeof(fb));
  const M3 ma{fa[0], fa[1], fa[2], fa[4], fa[5], fa[6], fa[8], fa[9], fa[10]};
  const M3 mb{fb[0], fb[1], fb[2], fb[4], fb[5], fb[6], fb[8], fb[9], fb[10]};
  M3 ra;
  M3 rb;
  if (!polar_rotation(ma, ra) || !polar_rotation(mb, rb)) {
    return lerp(a, b, t);
  }
  const bool mirrorA = det3(ra) < 0.f;
  if (mirrorA != (det3(rb) < 0.f)) {
    return lerp(a, b, t);
  }
  if (mirrorA) { // m = (-R)(-S), -R a proper rotation
    for (int i = 0; i < 9; ++i) {
      ra[i] = -ra[i];
      rb[i] = -rb[i];
    }
  }
  // S = R^T m for each; the result is R(t) lerp(Sa, Sb, t), exact at both ends.
  M3 s;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      float sa = 0.f;
      float sb = 0.f;
      for (int k = 0; k < 3; ++k) {
        sa += ra[k * 3 + r] * ma[k * 3 + c];
        sb += rb[k * 3 + r] * mb[k * 3 + c];
      }
      s[r * 3 + c] = sa + (sb - sa) * t;
    }
  }
  const M3 rt = from_quat(slerp(to_quat(ra), to_quat(rb), t));
  float out[12];
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      float v = 0.f;
      for (int k = 0; k < 3; ++k) {
        v += rt[r * 3 + k] * s[k * 3 + c];
      }
      out[r * 4 + c] = v;
    }
    out[r * 4 + 3] = fa[r * 4 + 3] + (fb[r * 4 + 3] - fa[r * 4 + 3]) * t;
  }
  Mat3x4<float> m;
  std::memcpy(static_cast<void*>(&m), out, sizeof(out));
  return m;
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
    if ((i == GX_VA_CLR0 || i == GX_VA_CLR1) && desc == GX_DIRECT && r.clrCount < 2) {
      r.clrOffset[r.clrCount] = offset;
      r.clrType[r.clrCount] = static_cast<u8>(attrFmt.type);
      ++r.clrCount;
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

// A vertex colour in the stream (GXCompType of a colour attribute) as four channels and
// their maxima, and back.
void read_color(const u8* p, u8 type, int c[4], int max[4]) noexcept {
  switch (type) {
  case GX_RGB565: {
    const int v = p[0] << 8 | p[1];
    c[0] = v >> 11, c[1] = (v >> 5) & 63, c[2] = v & 31, c[3] = 1;
    max[0] = 31, max[1] = 63, max[2] = 31, max[3] = 1;
    break;
  }
  case GX_RGB8:
  case GX_RGBX8:
    c[0] = p[0], c[1] = p[1], c[2] = p[2], c[3] = 1;
    max[0] = max[1] = max[2] = 255, max[3] = 1;
    break;
  case GX_RGBA4: {
    const int v = p[0] << 8 | p[1];
    c[0] = v >> 12, c[1] = (v >> 8) & 15, c[2] = (v >> 4) & 15, c[3] = v & 15;
    max[0] = max[1] = max[2] = max[3] = 15;
    break;
  }
  case GX_RGBA6: {
    const int v = p[0] << 16 | p[1] << 8 | p[2];
    c[0] = v >> 18, c[1] = (v >> 12) & 63, c[2] = (v >> 6) & 63, c[3] = v & 63;
    max[0] = max[1] = max[2] = max[3] = 63;
    break;
  }
  default: // GX_RGBA8
    c[0] = p[0], c[1] = p[1], c[2] = p[2], c[3] = p[3];
    max[0] = max[1] = max[2] = max[3] = 255;
    break;
  }
}

void write_color(u8* p, u8 type, const int c[4]) noexcept {
  switch (type) {
  case GX_RGB565: {
    const int v = c[0] << 11 | c[1] << 5 | c[2];
    p[0] = static_cast<u8>(v >> 8), p[1] = static_cast<u8>(v);
    break;
  }
  case GX_RGB8:
  case GX_RGBX8: // the fourth byte is padding: left as the stream has it
    p[0] = static_cast<u8>(c[0]), p[1] = static_cast<u8>(c[1]), p[2] = static_cast<u8>(c[2]);
    break;
  case GX_RGBA4: {
    const int v = c[0] << 12 | c[1] << 8 | c[2] << 4 | c[3];
    p[0] = static_cast<u8>(v >> 8), p[1] = static_cast<u8>(v);
    break;
  }
  case GX_RGBA6: {
    const int v = c[0] << 18 | c[1] << 12 | c[2] << 6 | c[3];
    p[0] = static_cast<u8>(v >> 16), p[1] = static_cast<u8>(v >> 8), p[2] = static_cast<u8>(v);
    break;
  }
  default:
    for (int i = 0; i < 4; ++i) {
      p[i] = static_cast<u8>(c[i]);
    }
    break;
  }
}

// The eigenvector of a symmetric 4x4 matrix with the largest eigenvalue, by Jacobi
// rotations; false when the two largest are too close to tell apart.
bool largest_eigenvector(std::array<std::array<double, 4>, 4> a, std::array<double, 4>& out) noexcept {
  std::array<std::array<double, 4>, 4> v{};
  for (int i = 0; i < 4; ++i) {
    v[i][i] = 1.0;
  }
  for (int sweep = 0; sweep < 32; ++sweep) {
    double off = 0.0;
    for (int p = 0; p < 4; ++p) {
      for (int q = p + 1; q < 4; ++q) {
        off += a[p][q] * a[p][q];
      }
    }
    if (off < 1e-24) {
      break;
    }
    for (int p = 0; p < 4; ++p) {
      for (int q = p + 1; q < 4; ++q) {
        if (std::abs(a[p][q]) < 1e-30) {
          continue;
        }
        const double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        const double t = (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0);
        const double s = t * c;
        for (int k = 0; k < 4; ++k) {
          const double akp = a[k][p];
          const double akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k < 4; ++k) {
          const double apk = a[p][k];
          const double aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (int k = 0; k < 4; ++k) {
          const double vkp = v[k][p];
          const double vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  int best = 0;
  for (int i = 1; i < 4; ++i) {
    if (a[i][i] > a[best][best]) {
      best = i;
    }
  }
  double second = -INFINITY;
  for (int i = 0; i < 4; ++i) {
    if (i != best) {
      second = std::max(second, a[i][i]);
    }
  }
  const double scale = std::max(std::abs(a[best][best]), 1e-30);
  if (a[best][best] - second < 1e-4 * scale) {
    return false;
  }
  for (int i = 0; i < 4; ++i) {
    out[i] = v[i][best];
  }
  return true;
}

// Whether a paired draw turned as a rigid body from its match, and if so about what: the
// best rotation of its positions about their centre (Horn's quaternion method), accepted
// when what it leaves unexplained is small against the draw's spread and the turn is big
// enough to matter. The turn and the move together are a screw: a rotation about an axis
// through `pivot`, and a slide along it, which a picture blends by angle.
void fit_rigid(DrawRecord& cur, const DrawRecord& prev) noexcept {
  cur.rigid = false;
  const u32 n = cur.key.vtxCount;
  if (sTunables.rigid <= 0.f || n < 3) {
    return;
  }
  const float* pa = &sPrevPositions[prev.first];
  const float* pb = &sPositions[cur.first];
  double ca[3] = {};
  double cb[3] = {};
  for (u32 v = 0; v < n; ++v) {
    for (int c = 0; c < 3; ++c) {
      ca[c] += pa[v * 3 + c];
      cb[c] += pb[v * 3 + c];
    }
  }
  for (int c = 0; c < 3; ++c) {
    ca[c] /= n;
    cb[c] /= n;
  }
  double sxy[3][3] = {};
  double spread = 0.0;
  for (u32 v = 0; v < n; ++v) {
    double a[3];
    double b[3];
    for (int c = 0; c < 3; ++c) {
      a[c] = pa[v * 3 + c] - ca[c];
      b[c] = pb[v * 3 + c] - cb[c];
      spread += a[c] * a[c];
    }
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        sxy[i][j] += a[i] * b[j];
      }
    }
  }
  if (spread < 1e-12) {
    return;
  }
  const double xx = sxy[0][0], xy = sxy[0][1], xz = sxy[0][2];
  const double yx = sxy[1][0], yy = sxy[1][1], yz = sxy[1][2];
  const double zx = sxy[2][0], zy = sxy[2][1], zz = sxy[2][2];
  const std::array<std::array<double, 4>, 4> nm{{
      {xx + yy + zz, yz - zy, zx - xz, xy - yx},
      {yz - zy, xx - yy - zz, xy + yx, zx + xz},
      {zx - xz, xy + yx, -xx + yy - zz, yz + zy},
      {xy - yx, zx + xz, yz + zy, -xx - yy + zz},
  }};
  std::array<double, 4> q;
  if (!largest_eigenvector(nm, q)) {
    return; // flat or symmetric: the turn is not determined
  }
  if (q[0] < 0) {
    for (auto& value : q) {
      value = -value;
    }
  }
  const double len = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  for (auto& value : q) {
    value /= len;
  }
  const double half = std::acos(std::min(1.0, q[0]));
  const double angle = 2.0 * half;
  if (angle < sTunables.rigidMin * 3.14159265358979 / 180.0 || angle > 175.0 * 3.14159265358979 / 180.0) {
    return;
  }
  const double sinHalf = std::sin(half);
  const float axis[3] = {static_cast<float>(q[1] / sinHalf), static_cast<float>(q[2] / sinHalf),
                         static_cast<float>(q[3] / sinHalf)};
  const M3 r = from_quat({static_cast<float>(q[0]), static_cast<float>(q[1]), static_cast<float>(q[2]),
                          static_cast<float>(q[3])});
  double residual = 0.0;
  for (u32 v = 0; v < n; ++v) {
    const float a[3] = {static_cast<float>(pa[v * 3] - ca[0]), static_cast<float>(pa[v * 3 + 1] - ca[1]),
                        static_cast<float>(pa[v * 3 + 2] - ca[2])};
    float ra[3];
    apply(r, a, ra);
    for (int c = 0; c < 3; ++c) {
      const double d = ra[c] - (pb[v * 3 + c] - cb[c]);
      residual += d * d;
    }
  }
  if (residual > static_cast<double>(sTunables.rigid) * sTunables.rigid * spread) {
    return;
  }
  // The move T = cb - R ca splits into a slide along the axis and a part across it, which
  // the rotation about a point p makes: (I - R) p = T_across, p = (T_across + cot(angle/2)
  // axis x T_across) / 2.
  const float fca[3] = {static_cast<float>(ca[0]), static_cast<float>(ca[1]), static_cast<float>(ca[2])};
  float rca[3];
  apply(r, fca, rca);
  float t[3];
  for (int c = 0; c < 3; ++c) {
    t[c] = static_cast<float>(cb[c]) - rca[c];
  }
  const float slide = t[0] * axis[0] + t[1] * axis[1] + t[2] * axis[2];
  float across[3];
  for (int c = 0; c < 3; ++c) {
    across[c] = t[c] - slide * axis[c];
  }
  const float cross[3] = {axis[1] * across[2] - axis[2] * across[1], axis[2] * across[0] - axis[0] * across[2],
                          axis[0] * across[1] - axis[1] * across[0]};
  const float cotHalf = static_cast<float>(std::cos(half) / sinHalf);
  for (int c = 0; c < 3; ++c) {
    cur.pivot[c] = 0.5f * (across[c] + cotHalf * cross[c]);
    cur.axis[c] = axis[c];
  }
  cur.angle = static_cast<float>(angle);
  cur.slide = slide;
  cur.rigid = true;
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

// A draw that did not move: its key and its positions' hash.
struct Still {
  DrawKey key;
  u64 posHash;
  bool operator==(const Still& rhs) const = default;
  template <typename H>
  friend H AbslHashValue(H h, const Still& s) {
    return H::combine(std::move(h), s.key, s.posHash);
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
  u32 stillPaired = 0;
  for (auto& r : sDraws) {
    r.match = -1;
  }
  const auto take = [&](size_t i, size_t j) {
    pair[i] = static_cast<s32>(j);
    used[j] = true;
  };
  // First, draws that did not move: the same shape, coordinates and positions in both
  // ticks. Text and HUD quads are alike in everything else (each glyph a texture of its own,
  // drawn with the same coordinates), so a letter typed in the middle would otherwise shift
  // every later letter's pair by one.
  {
    absl::flat_hash_map<Still, std::vector<u32>> still;
    stillPaired = 0;
    for (u32 j = 0; j < m; ++j) {
      still[Still{sPrevDraws[j].key, sPrevDraws[j].posHash}].push_back(j);
    }
    for (size_t i = 0; i < n; ++i) {
      const auto it = still.find(Still{sDraws[i].key, sDraws[i].posHash});
      if (it == still.end()) {
        continue;
      }
      for (const u32 j : it->second) {
        if (!used[j]) {
          take(i, j);
          ++stillPaired;
          break;
        }
      }
    }
  }
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
    if (at.first >= 0 && at.second >= 0 && pair[at.first] < 0 && !used[at.second] &&
        draw_close_enough(sPrevDraws[at.second], sDraws[at.first])) {
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
  u32 drawsRigid = 0;
  for (size_t i = 0; i < n; ++i) {
    sDraws[i].match = pair[i];
    drawsMatched += pair[i] >= 0;
    if (pair[i] >= 0) {
      fit_rigid(sDraws[i], sPrevDraws[pair[i]]);
      drawsRigid += sDraws[i].rigid;
    } else {
      sDraws[i].rigid = false;
    }
  }
  sStats.drawsRigid = drawsRigid;
  sStillPaired = stillPaired;
  sStats.draws = static_cast<u32>(n);
  sStats.drawsMatched = drawsMatched;
  sStats.drawsRejected = drawsRejected;
  // Draws that did not move say nothing about a cut (the HUD stays put across one): the cut
  // test counts the others.
  matched += drawsMatched - stillPaired;
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
      float bestLook = std::numeric_limits<float>::max();
      for (const u32 idx : candidates) {
        if (used[idx]) {
          continue;
        }
        const float d = distance2(sPrevious[idx], r);
        // Records with the same matrices (one model's materials) pair by their colours.
        if (d < bestDistance || (d == bestDistance && look_distance(sPrevious[idx].look, r.look) < bestLook)) {
          bestDistance = d;
          bestLook = look_distance(sPrevious[idx].look, r.look);
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
  sStats.matched = matched - (sStats.drawsMatched - sStillPaired);
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
    probe.posHash = fnv(0xCBF29CE484222325ull, &sPositions[probe.first], sizeof(float) * 3 * vtxCount);
    float r2 = 0.f;
    for (u32 c = 0; c < 3; ++c) {
      probe.centroid[c] = 0.5f * (lo[c] + hi[c]);
      r2 += 0.25f * (hi[c] - lo[c]) * (hi[c] - lo[c]);
    }
    probe.radius = std::sqrt(r2);
    probe.raw = static_cast<u32>(sRaw.size());
    sRaw.insert(sRaw.end(), data, data + static_cast<size_t>(vtxCount) * vtxSize);
    sDraws.push_back(probe);
    return data;
  }
  // Replay: the same stream from the same state, so the same draw at the same index. At
  // blend 1 the draw is the tick's own (exactly: a blend there would add rounding).
  if (sCut || sTunables.blend == 0.f || sTunables.vertices == 0.f || sBlend >= 1.f || index >= sDraws.size()) {
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
  if (cur.rigid) {
    // Along the screw: each position turns by t of the angle about the axis through the
    // pivot and slides t of the way along it; what the rigid turn leaves unexplained (the
    // residual) is added in proportion, so t = 0 and t = 1 give the two ticks exactly.
    const M3 full = axis_angle(cur.axis, cur.angle);
    const M3 part = axis_angle(cur.axis, cur.angle * t);
    for (u32 v = 0; v < vtxCount; ++v) {
      u8* p = sVertexScratch.data() + v * vtxSize + cur.posOffset;
      const float* a = &sPrevPositions[prev.first + v * 3];
      const float* b = &sPositions[cur.first + v * 3];
      const float rel[3] = {a[0] - cur.pivot[0], a[1] - cur.pivot[1], a[2] - cur.pivot[2]};
      float turned[3];
      float partly[3];
      apply(full, rel, turned);
      apply(part, rel, partly);
      for (u32 c = 0; c < cur.posCount; ++c) {
        const float fit = turned[c] + cur.pivot[c] + cur.slide * cur.axis[c];
        const float value = partly[c] + cur.pivot[c] + t * cur.slide * cur.axis[c] + t * (b[c] - fit);
        write_comp(p + c * size, cur.posType, cur.posFrac, value);
      }
    }
  } else {
    for (u32 v = 0; v < vtxCount; ++v) {
      u8* p = sVertexScratch.data() + v * vtxSize + cur.posOffset;
      const float* a = &sPrevPositions[prev.first + v * 3];
      const float* b = &sPositions[cur.first + v * 3];
      for (u32 c = 0; c < cur.posCount; ++c) {
        write_comp(p + c * size, cur.posType, cur.posFrac, a[c] + (b[c] - a[c]) * t);
      }
    }
  }
  // Vertex colours (the screen fade's quad), unless a channel jumps (a switch, not a fade).
  if (sTunables.colors != 0.f && cur.clrCount != 0 && prev.clrCount == cur.clrCount) {
    bool blendable = true;
    for (u32 k = 0; k < cur.clrCount && blendable; ++k) {
      blendable = prev.clrType[k] == cur.clrType[k] && prev.clrOffset[k] == cur.clrOffset[k];
      for (u32 v = 0; v < vtxCount && blendable; ++v) {
        int ca[4], cb[4], max[4];
        read_color(&sPrevRaw[prev.raw + v * vtxSize + cur.clrOffset[k]], cur.clrType[k], ca, max);
        read_color(data + v * vtxSize + cur.clrOffset[k], cur.clrType[k], cb, max);
        for (int c = 0; c < 4; ++c) {
          if (std::abs(cb[c] - ca[c]) > sTunables.colorMax * max[c]) {
            blendable = false;
          }
        }
      }
    }
    bool changed = false;
    for (u32 k = 0; k < cur.clrCount && blendable; ++k) {
      for (u32 v = 0; v < vtxCount; ++v) {
        int ca[4], cb[4], max[4], out[4];
        read_color(&sPrevRaw[prev.raw + v * vtxSize + cur.clrOffset[k]], cur.clrType[k], ca, max);
        read_color(data + v * vtxSize + cur.clrOffset[k], cur.clrType[k], cb, max);
        for (int c = 0; c < 4; ++c) {
          out[c] = static_cast<int>(std::lround(ca[c] + (cb[c] - ca[c]) * t));
          changed |= ca[c] != cb[c];
        }
        write_color(sVertexScratch.data() + v * vtxSize + cur.clrOffset[k], cur.clrType[k], out);
      }
    }
    sStats.colorsBlended += changed;
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
      save_look(r.look);
      sCurrentByHash.emplace(hash, static_cast<u32>(sCurrent.size()));
      sCurrent.push_back(r);
    }
    return;
  }
  if (!sReplaying || sCut || sTunables.blend == 0.f || sBlend >= 1.f) {
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
  save_look(sSavedLook);
  // Blend from where the draw was a tick ago (t = 0) to where it is now (t = 1).
  const float t = sBlend;
  auto& state = g_gxState;
  for (u32 i = 0; i < MaxPnMtx; ++i) {
    state.pnMtx[i].pos = blend_matrix(prev.m.pos[i], cur.m.pos[i], t);
    state.pnMtx[i].nrm = blend_matrix(prev.m.nrm[i], cur.m.nrm[i], t);
  }
  state.proj = lerp(prev.m.proj, cur.m.proj, t);
  if (sTunables.colors != 0.f) {
    const Look& a = prev.look;
    const Look& b = cur.look;
    for (u32 i = 0; i < MaxTevRegs; ++i) {
      state.colorRegs[i] = blend_color(a.colorRegs[i], b.colorRegs[i], t);
    }
    for (u32 i = 0; i < GX_MAX_KCOLOR; ++i) {
      state.kcolors[i] = blend_color(a.kcolors[i], b.kcolors[i], t);
    }
    for (u32 i = 0; i < MaxColorChannels; ++i) {
      state.colorChannelState[i].matColor = blend_color(a.mat[i], b.mat[i], t);
      state.colorChannelState[i].ambColor = blend_color(a.amb[i], b.amb[i], t);
    }
    if (a.fogType == b.fogType) {
      state.fog.color = blend_color(a.fogColor, b.fogColor, t);
      state.fog.a = a.fogA + (b.fogA - a.fogA) * t;
      state.fog.b = a.fogB + (b.fogB - a.fogB) * t;
      state.fog.c = a.fogC + (b.fogC - a.fogC) * t;
    }
    for (u32 i = 0; i < MaxTexMtx; ++i) {
      state.texMtxs[i] = blend_tex_mtx(a.texMtxs[i], b.texMtxs[i], t);
    }
    for (u32 i = 0; i < MaxPTTexMtx; ++i) {
      state.ptTexMtxs[i] = blend_tex_mtx(a.ptTexMtxs[i], b.ptTexMtxs[i], t);
    }
    if (look_distance(a, b) != 0.f || color_distance(a.fogColor, b.fogColor) != 0.f ||
        std::memcmp(a.texMtxs.data(), b.texMtxs.data(), sizeof(a.texMtxs)) != 0 ||
        std::memcmp(a.ptTexMtxs.data(), b.ptTexMtxs.data(), sizeof(a.ptTexMtxs)) != 0) {
      ++sStats.looksBlended;
    }
  }
  sSwapped = true;
  ++sStats.blended;
}

void after_uniform() noexcept {
  if (sSwapped) {
    load_matrices(sSaved);
    load_look(sSavedLook);
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
    sRaw = {};
    sPrevRaw = {};
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
  sPrevRaw = std::move(sRaw);
  sRaw.clear();
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
  } else if (n == "rotation") {
    sTunables.rotation = value;
  } else if (n == "rigid") {
    sTunables.rigid = value;
  } else if (n == "rigidmin") {
    sTunables.rigidMin = value;
  } else if (n == "colors") {
    sTunables.colors = value;
  } else if (n == "colormax") {
    sTunables.colorMax = value;
  } else if (n == "texjump") {
    sTunables.texJump = value;
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
