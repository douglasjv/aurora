// Debug trace of 2D (orthographic) draws, for planning widescreen layouts. Off unless
// aurora::gfx::set_trace_2d is given a path; when off, nothing here runs.
#include "trace_2d.hpp"

#include "../gfx/frame.hpp"
#include "../gfx/texture.hpp"
#include "../gfx/texture_replacement.hpp"
#include "../internal.hpp"
#include "gx.hpp"

#include <absl/container/flat_hash_set.h>
#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace aurora::gx::trace_2d {
namespace {
FILE* s_file = nullptr;
absl::flat_hash_set<std::string> s_seen;

float read_comp(const u8* p, GXCompType type, u8 frac, std::endian e) noexcept {
  switch (type) {
  case GX_U8:
    return std::ldexp(static_cast<float>(*p), -frac);
  case GX_S8:
    return std::ldexp(static_cast<float>(static_cast<s8>(*p)), -frac);
  case GX_U16:
    return std::ldexp(static_cast<float>(read_bits<u16>(p, e)), -frac);
  case GX_S16:
    return std::ldexp(static_cast<float>(read_bits<s16>(p, e)), -frac);
  case GX_F32:
    return read_bits<f32>(p, e);
  default:
    return 0.f;
  }
}

// Reads one attribute's components (direct or through its array) into out.
bool read_attr(const u8* vtx, u32 offset, int i, GXAttrType desc, const VtxAttrFmt& attrFmt, float* out,
               u8& count) noexcept {
  const auto attr = static_cast<GXAttr>(i);
  const u8 size = comp_type_size(attr, attrFmt.type);
  count = comp_cnt_count(attr, attrFmt.cnt);
  const u8* src = vtx + offset;
  std::endian e = std::endian::big;
  if (desc != GX_DIRECT) {
    const auto& array = g_gxState.arrays[i];
    if (array.data == nullptr) {
      return false;
    }
    const u32 index = desc == GX_INDEX8 ? vtx[offset] : read_bits<u16>(vtx + offset);
    src = static_cast<const u8*>(array.data) + index * array.stride;
    e = array.le ? std::endian::little : std::endian::big;
  }
  for (u8 c = 0; c < count && c < 3; ++c) {
    out[c] = read_comp(src + size * c, attrFmt.type, attrFmt.frac, e);
  }
  return true;
}

// The vertex's position, position-matrix index and first texture coordinate (when it has
// one), from the raw FIFO bytes of one vertex.
bool decode_vertex(const u8* vtx, GXVtxFmt fmt, Vec3<float>& pos, u32& mtx, float* uv, bool& hasUv) noexcept {
  const auto& vtxFmt = g_gxState.vtxFmts[fmt];
  mtx = g_gxState.currentPnMtx;
  hasUv = false;
  bool hasPos = false;
  u32 offset = 0;
  for (int i = GX_VA_PNMTXIDX; i <= GX_VA_TEX0; ++i) {
    const auto attr = static_cast<GXAttr>(i);
    const auto& attrFmt = vtxFmt.attrs[i];
    const auto desc = g_gxState.vtxDesc[i];
    if (desc == GX_NONE) {
      continue;
    }
    if (i == GX_VA_PNMTXIDX && desc == GX_DIRECT) {
      mtx = vtx[offset] / 3;
    }
    if (i == GX_VA_NRM && attrFmt.cnt == GX_NRM_NBT3 && desc != GX_DIRECT) {
      return hasPos; // three indices; nothing after it is needed badly enough to decode
    }
    u8 count = 0;
    if (i == GX_VA_POS) {
      float p[3] = {0.f, 0.f, 0.f};
      if (!read_attr(vtx, offset, i, desc, attrFmt, p, count)) {
        return false;
      }
      pos = {p[0], p[1], count > 2 ? p[2] : 0.f};
      hasPos = true;
    } else if (i == GX_VA_TEX0) {
      hasUv = read_attr(vtx, offset, i, desc, attrFmt, uv, count);
    }
    if (desc == GX_DIRECT) {
      offset += comp_type_size(attr, attrFmt.type) * comp_cnt_count(attr, attrFmt.cnt);
    } else {
      offset += desc == GX_INDEX8 ? 1 : 2;
    }
  }
  return hasPos;
}

float dot_row(const Vec4<float>& row, float x, float y, float z) noexcept {
  return row[0] * x + row[1] * y + row[2] * z + row[3];
}
} // namespace

void set_path(const char* path) noexcept {
  if (s_file != nullptr) {
    fclose(s_file);
    s_file = nullptr;
  }
  s_seen.clear();
  if (path != nullptr && path[0] != '\0') {
    s_file = fopen(path, "w");
    if (s_file != nullptr) {
      fputs("# frame\tx0\ty0\tx1\ty1\twidth\tfull_width\ttexture\ttex_size\tvertices\tscissor\tviewport\tuv\tprim\n",
            s_file);
    }
  }
}

bool enabled() noexcept { return s_file != nullptr; }

namespace {
struct Corner {
  float sx, sy, u, v;
  bool hasUv;
};

void emit(GXPrimitive prim, const Corner* corners, u32 count, const std::string& texture, const std::string& texSize,
          u32 texW, u32 texH, const std::string& scissor, const std::string& viewport) noexcept {
  float x0 = INFINITY, y0 = INFINITY, x1 = -INFINITY, y1 = -INFINITY;
  float u0 = INFINITY, v0 = INFINITY, u1 = -INFINITY, v1 = -INFINITY;
  bool hasUv = false;
  for (u32 i = 0; i < count; ++i) {
    const auto& c = corners[i];
    x0 = std::min(x0, c.sx);
    y0 = std::min(y0, c.sy);
    x1 = std::max(x1, c.sx);
    y1 = std::max(y1, c.sy);
    if (c.hasUv) {
      hasUv = true;
      u0 = std::min(u0, c.u);
      v0 = std::min(v0, c.v);
      u1 = std::max(u1, c.u);
      v1 = std::max(v1, c.v);
    }
  }
  // Texture coordinates in texels when the texture's size is known (texture matrices are
  // not applied), otherwise as given.
  std::string uv = "-";
  if (hasUv) {
    const float sw = texW != 0 ? static_cast<float>(texW) : 1.f;
    const float sh = texH != 0 ? static_cast<float>(texH) : 1.f;
    uv = fmt::format("{:.1f},{:.1f}-{:.1f},{:.1f}", u0 * sw, v0 * sh, u1 * sw, v1 * sh);
  }
  const bool fullWidth = x0 <= 1.f && x1 >= 639.f;
  // One line per texture, rounded rectangle and texture region, the first time it is drawn.
  const auto key = fmt::format("{}|{}|{}|{}|{}|{}|{}", texture, std::lround(x0 / 4), std::lround(y0 / 4),
                               std::lround(x1 / 4), std::lround(y1 / 4), scissor, uv);
  if (!s_seen.insert(key).second) {
    return;
  }
  fmt::print(s_file, "{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
             gfx::current_frame(), x0, y0, x1, y1, x1 - x0, fullWidth ? "full" : "-", texture, texSize, count, scissor,
             viewport, uv, static_cast<u32>(prim));
  fflush(s_file);
}
} // namespace

void draw(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, u32 vtxSize, const u8* data) noexcept {
  if (g_gxState.projType != GX_ORTHOGRAPHIC || vtxCount == 0 || vtxSize == 0) {
    return;
  }
  const auto& vp = g_gxState.logicalViewport;
  const auto& proj = g_gxState.proj;
  std::vector<Corner> corners(vtxCount);
  for (u16 v = 0; v < vtxCount; ++v) {
    Vec3<float> pos{};
    u32 mtx = 0;
    float uv[3] = {0.f, 0.f, 0.f};
    bool hasUv = false;
    if (!decode_vertex(data + v * vtxSize, fmt, pos, mtx, uv, hasUv) || mtx >= MaxPnMtx) {
      return;
    }
    const auto& m = g_gxState.pnMtx[mtx].pos;
    const float vx = dot_row(m.m0, pos.x, pos.y, pos.z);
    const float vy = dot_row(m.m1, pos.x, pos.y, pos.z);
    const float vz = dot_row(m.m2, pos.x, pos.y, pos.z);
    const float nx = dot_row(proj.m0, vx, vy, vz);
    const float ny = dot_row(proj.m1, vx, vy, vz);
    corners[v] = {vp.left + (nx + 1.f) * 0.5f * vp.width, vp.top + (1.f - ny) * 0.5f * vp.height, uv[0], uv[1], hasUv};
  }

  std::string texture = "-";
  std::string texSize = "-";
  u32 texW = 0, texH = 0;
  if (g_gxState.numTevStages > 0) {
    const u32 map = g_gxState.tevStages[0].texMapId;
    if (map != GX_TEXMAP_NULL && (map & GX_TEX_DISABLE) == 0) {
      const auto& obj = g_gxState.loadedTextures[map & 7];
      if (obj.has_data()) {
        texW = obj.width();
        texH = obj.height();
        texSize = fmt::format("{}x{}/fmt{}", texW, texH, obj.format());
        if (g_gxState.copyTextures.contains(obj.data)) {
          // A copy of the frame (GXCopyTex): screen content, not art from the disc.
          texture = "efb-copy";
        } else {
          texture = gfx::texture_replacement::build_texture_replacement_name(obj);
          if (const auto dot = texture.rfind('.'); dot != std::string::npos) {
            texture = texture.substr(0, dot) + ".png";
          }
        }
      }
    }
  }
  const auto& sc = g_gxState.logicalScissor;
  const std::string scissor = sc.x == 0 && sc.y == 0 && sc.width == 640 && sc.height == 480
                                  ? "-"
                                  : fmt::format("{},{},{},{}", sc.x, sc.y, sc.width, sc.height);
  const std::string viewport = vp.left == 0.f && vp.top == 0.f && vp.width == 640.f && vp.height == 480.f
                                   ? "-"
                                   : fmt::format("{:g},{:g},{:g},{:g}", vp.left, vp.top, vp.width, vp.height);
  // A batch of quads is logged quad by quad: a menu drawn as one batch is many pieces.
  if (prim == GX_QUADS && vtxCount % 4 == 0) {
    for (u32 q = 0; q < vtxCount; q += 4) {
      emit(prim, &corners[q], 4, texture, texSize, texW, texH, scissor, viewport);
    }
  } else {
    emit(prim, corners.data(), vtxCount, texture, texSize, texW, texH, scissor, viewport);
  }
}
} // namespace aurora::gx::trace_2d

namespace aurora::gfx {
void set_trace_2d(const char* path) noexcept { gx::trace_2d::set_path(path); }
} // namespace aurora::gfx
