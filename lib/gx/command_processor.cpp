#include "command_processor.hpp"

#include <array>
#include <string_view>
#include <cstdlib>
#include <cstring>
#include <atomic>

#include <aurora/gfx.h>
#include <aurora/gfx.hpp>

#include "../gfx/depth_peek.hpp"
#include "../gfx/recording.hpp"
#include "../internal.hpp"
#include "dolphin/gd/GDGeometry.h"
#include "dolphin/gx/GXAurora.h"
#include "gx.hpp"
#include "pipeline.hpp"
#include "regs.hpp"
#include "replay.hpp"
#include "shader_info.hpp"
#include "texture.hpp"
#include "trace_2d.hpp"

#include <tracy/Tracy.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace aurora::gx::fifo {
std::array<std::atomic<uint64_t>, 8> g_mergeCounts{};
std::array<std::atomic<uint64_t>, 4> g_mergeImmCounts{};
namespace {
constexpr Module Log{"aurora::gx::fifo"};

u16 prepare_idx_buffer(ByteBuffer& buf, GXPrimitive prim, u16 vtxStart, u16 vtxCount) noexcept {
  u16 numIndices = 0;
  if (prim == GX_QUADS) {
    buf.reserve_extra((vtxCount / 4) * 6 * sizeof(u16));

    for (u16 v = 0; v < vtxCount; v += 4) {
      u16 idx0 = vtxStart + v;
      u16 idx1 = vtxStart + v + 1;
      u16 idx2 = vtxStart + v + 2;
      u16 idx3 = vtxStart + v + 3;

      buf.append(idx0);
      buf.append(idx1);
      buf.append(idx2);
      numIndices += 3;

      buf.append(idx2);
      buf.append(idx3);
      buf.append(idx0);
      numIndices += 3;
    }
  } else if (prim == GX_TRIANGLES) {
    buf.reserve_extra(vtxCount * sizeof(u16));
    for (u16 v = 0; v < vtxCount; ++v) {
      const u16 idx = vtxStart + v;
      buf.append(idx);
      ++numIndices;
    }
  } else if (prim == GX_TRIANGLEFAN) {
    buf.reserve_extra(((u32(vtxCount) - 3) * 3 + 3) * sizeof(u16));
    for (u16 v = 0; v < vtxCount; ++v) {
      const u16 idx = vtxStart + v;
      if (v < 3) {
        buf.append(idx);
        ++numIndices;
        continue;
      }
      buf.append(std::array{vtxStart, static_cast<u16>(idx - 1), idx});
      numIndices += 3;
    }
  } else if (prim == GX_TRIANGLESTRIP) {
    buf.reserve_extra(((static_cast<u32>(vtxCount) - 3) * 3 + 3) * sizeof(u16));
    for (u16 v = 0; v < vtxCount; ++v) {
      const u16 idx = vtxStart + v;
      if (v < 3) {
        buf.append(idx);
        ++numIndices;
        continue;
      }
      if ((v & 1) == 0) {
        buf.append(std::array{static_cast<u16>(idx - 2), static_cast<u16>(idx - 1), idx});
      } else {
        buf.append(std::array{static_cast<u16>(idx - 1), static_cast<u16>(idx - 2), idx});
      }
      numIndices += 3;
    }
  } else if (prim == GX_LINES || prim == GX_LINESTRIP || prim == GX_POINTS) {
    buf.reserve_extra(6 * sizeof(u16));
    buf.append<u16>(0);
    buf.append<u16>(1);
    buf.append<u16>(3);
    buf.append<u16>(3);
    buf.append<u16>(2);
    buf.append<u16>(0);
    numIndices = 6;
  } else
    UNLIKELY FATAL("unsupported primitive type {}", static_cast<u32>(prim));
  return numIndices;
}

// GX FIFO opcodes - use CP_ prefix to avoid clashing with GXCommandList.h macros
constexpr u8 CP_CMD_NOP = GX_NOP;
constexpr u8 CP_CMD_LOAD_CP_REG = GX_LOAD_CP_REG;
constexpr u8 CP_CMD_LOAD_XF_REG = GX_LOAD_XF_REG;
constexpr u8 CP_CMD_LOAD_INDX_A = GX_LOAD_INDX_A;
constexpr u8 CP_CMD_LOAD_INDX_B = GX_LOAD_INDX_B;
constexpr u8 CP_CMD_LOAD_INDX_C = GX_LOAD_INDX_C;
constexpr u8 CP_CMD_LOAD_INDX_D = GX_LOAD_INDX_D;
constexpr u8 CP_CMD_CALL_DL = GX_CMD_CALL_DL;
constexpr u8 CP_CMD_INVAL_VTX = GX_CMD_INVL_VC;
constexpr u8 CP_CMD_LOAD_BP_REG = GX_LOAD_BP_REG & GX_OPCODE_MASK;

// Primitive type mask
constexpr u8 CP_OPCODE_MASK = GX_OPCODE_MASK;
constexpr u8 CP_VAT_MASK = GX_VAT_MASK;

struct FogRangeLutKey {
  std::array<u16, 10> rangeK;
  f32 rangeCenter;
  f32 renderWidth;
  u32 targetWidth;

  bool operator==(const FogRangeLutKey&) const = default;
};

struct FogRangeLutEntry {
  FogRangeLutKey key;
  std::vector<f32> factors;
};

constexpr size_t MaxFogRangeLuts = 32;
std::vector<FogRangeLutEntry> sFogRangeLuts;

struct DrawCache {
  uint64_t targetLayoutKey = 0;
  PipelineConfig config{};
  ShaderInfo shaderInfo{};
  gfx::PipelineRef pipelineRef{};
  GXBindGroups bindGroups{};
  uint64_t bindGeneration = 0;
  GXVtxFmt fmt = GX_MAX_VTXFMT;
  u8 lineMode = 0;
  bool hasPipeline = false;
  gfx::Range uniformRange{};
  gfx::Range fogRange{};
  FogRangeLutKey fogRangeKey{};
  bool hasFogRange = false;
  GXVtxFmt lastDrawFmt = GX_MAX_VTXFMT;
  // The texture coordinates the cached pipeline reads from the vertex stream although the
  // game indexes them (expand_small_arrays), one bit for each of TEX0..TEX7.
  u8 expandMask = 0;
};
DrawCache sDrawCache;

FogRangeLutKey fog_range_lut_key() noexcept {
  const auto& state = g_gxState.fog;
  const f32 logicalWidth = std::max(g_gxState.logicalViewport.width, 1.f);
  const f32 renderWidth = std::max(g_gxState.renderViewport.width, 1.f);
  return {
      .rangeK = state.rangeK,
      .rangeCenter = ((static_cast<f32>(state.rangeCenter) - g_gxState.logicalViewport.left) / logicalWidth) * 2.f -
                     1.f + (g_gxState.renderViewport.left / renderWidth) * 2.f,
      .renderWidth = renderWidth,
      .targetWidth = gfx::get_render_target_size().x,
  };
}

std::vector<f32> build_fog_range_lut(const FogRangeLutKey& key) {
  std::array<f32, 10> rangeK;
  for (u32 i = 0; i < rangeK.size(); ++i) {
    const u32 source = (i & ~1u) | (1u - (i & 1u));
    rangeK[i] = static_cast<f32>(key.rangeK[source]) / 64.f;
  }

  std::vector<f32> lut(key.targetWidth);
  for (u32 x = 0; x < key.targetWidth; ++x) {
    const f32 screenX = ((static_cast<f32>(x) + 0.5f) / key.renderWidth) * 2.f - 1.f;
    const f32 offset = screenX - key.rangeCenter;
    const f32 rangeIndex = std::clamp(9.f - std::abs(offset) * 9.f, 0.f, 9.f);
    const u32 lower = static_cast<u32>(rangeIndex);
    const u32 upper = std::min(lower + 1, 9u);
    const f32 fraction = rangeIndex - static_cast<f32>(lower);
    const f32 k = std::max(rangeK[lower] * (1.f - fraction) + rangeK[upper] * fraction, 0.000001f);
    lut[x] = std::sqrt(offset * offset + k * k) / k;
  }
  return lut;
}

const std::vector<f32>& resolve_fog_range_lut(const FogRangeLutKey& key) {
  for (const auto& entry : sFogRangeLuts) {
    if (entry.key == key) {
      return entry.factors;
    }
  }
  if (sFogRangeLuts.size() == MaxFogRangeLuts) {
    sFogRangeLuts.erase(sFogRangeLuts.begin());
  }
  sFogRangeLuts.emplace_back(FogRangeLutEntry{key, build_fog_range_lut(key)});
  return sFogRangeLuts.back().factors;
}

gfx::Range push_fog_range_lut(const FogRangeLutKey& key) {
  const auto& lut = resolve_fog_range_lut(key);
  return gfx::push_storage(reinterpret_cast<const u8*>(lut.data()), lut.size() * sizeof(f32));
}

u8 line_mode_for_prim(GXPrimitive prim) noexcept {
  switch (prim) {
  case GX_LINES:
    return 1;
  case GX_LINESTRIP:
    return 2;
  case GX_POINTS:
    return 3;
  default:
    return 0;
  }
}
} // namespace

static void handle_draw(u8 cmd, ByteReader& reader) noexcept;
static void handle_aurora(ByteReader& reader) noexcept;

ProcessResult process(const u8* data, u32 size) noexcept {
  ZoneScoped;
  ByteReader reader{{data, size}};

  while (!reader.empty()) {
    const u8 cmd = reader.read<u8>();
    u8 opcode = cmd & CP_OPCODE_MASK;

    switch (opcode) {
    case CP_CMD_NOP:
      continue;

    case CP_CMD_LOAD_BP_REG: {
      const u32 value = reader.read<u32>();
      handle_bp(value);
      if (reg_get(value, 8, 24) == GX_BP_REG_DRAWDONE) {
        return {static_cast<u32>(reader.offset()), true};
      }
      break;
    }

    case CP_CMD_LOAD_CP_REG: {
      const u8 addr = reader.read<u8>();
      handle_cp(addr, reader.read<u32>());
      break;
    }

    case CP_CMD_LOAD_XF_REG: {
      const u32 header = reader.read<u32>();
      const u32 count = ((header >> 16) & 0xFFFF) + 1;
      const u16 addr = header & 0xFFFF;
      handle_xf(addr, reader.take(count * sizeof(u32)));
      break;
    }

    case CP_CMD_LOAD_INDX_A:
    case CP_CMD_LOAD_INDX_B:
    case CP_CMD_LOAD_INDX_C:
    case CP_CMD_LOAD_INDX_D: {
      ZoneScopedN("LOAD_INDX");
      const u32 arrayType = GX_POS_MTX_ARRAY + (opcode - CP_CMD_LOAD_INDX_A) / 0x08;
      const u16 srcArrayIdx = reader.read<u16>();
      const u16 addrLen = reader.read<u16>();

      const u16 len = (addrLen >> 12) + 1;
      const u16 dstAddr = addrLen & 0x0FFF;
      auto const& array = g_gxState.arrays[arrayType];
      const u32 srcOffset = static_cast<u32>(srcArrayIdx) * array.stride;
      const u32 srcSize = static_cast<u32>(len) * sizeof(u32);
      AURORA_ASSERT(array.data != nullptr, "indexed XF load from unmapped array {}", arrayType);
      AURORA_ASSERT(srcOffset <= array.size && srcSize <= array.size - srcOffset,
                    "indexed XF load outside array {}: offset={}, size={}, array size={}", arrayType, srcOffset,
                    srcSize, array.size);
      auto const* srcData = static_cast<const u8*>(array.data) + srcOffset;
      replay::note_read(array.data, array.size);
      if (!copy_xf_data(dstAddr, srcData, len, array.le ? std::endian::little : std::endian::big)) {
#ifndef NDEBUG
        Log.debug("Unimplemented indexed XF load (opcode 0x{:02X}, dstAddr=%04x)", opcode, dstAddr);
#endif
      }
      break;
    }

    case CP_CMD_CALL_DL: {
      // Call display list: 8 bytes (address + size)
      Log.warn("Ignoring nested GX_CMD_CALL_DL");
      reader.skip(8);
      break;
    }

    case CP_CMD_INVAL_VTX: {
      for (auto& array : g_gxState.arrays) {
        array.cachedRange = {};
      }
      g_gxState.dirty |= DirtyImmediates;
      break;
    }

    case GX_AURORA: {
      handle_aurora(reader);
      break;
    }

    // Draw commands: 0x80-0xBF
    case GX_DRAW_QUADS:
    case GX_DRAW_TRIANGLES:
    case GX_DRAW_TRIANGLE_STRIP:
    case GX_DRAW_TRIANGLE_FAN:
    case GX_DRAW_LINES:
    case GX_DRAW_LINE_STRIP:
    case GX_DRAW_POINTS: {
      handle_draw(cmd, reader);
      break;
    }

    default:
      // Check if it's a draw command (0x80-0xBF range)
      if (cmd >= 0x80) {
        handle_draw(cmd, reader);
      } else if (replay::replaying()) {
        replay::fail("unknown opcode");
        return {size, false};
      } else {
        // Hex dump surrounding bytes for debugging
        {
          const size_t pos = reader.offset();
          size_t dumpStart = (pos > 17) ? pos - 17 : 0;
          size_t dumpEnd = (pos + 16 < size) ? pos + 16 : size;
          std::string hex;
          for (size_t i = dumpStart; i < dumpEnd; i++) {
            if (i == pos - 1)
              hex += fmt::format("[{:02x}]", data[i]);
            else
              hex += fmt::format(" {:02x}", data[i]);
          }
          Log.error("  hex dump (pos {}-{}):{}", dumpStart, dumpEnd - 1, hex);
        }
        FATAL("command_processor: unknown opcode 0x{:02X} at pos {}", cmd, reader.offset() - 1);
      }
      break;
    }
  }
  return {size, false};
}

[[noreturn]] static void handle_draw_overrun(size_t totalVtxBytes, const ByteReader& reader) noexcept {
  // Hex dump around the draw command for debugging
  const size_t pos = reader.offset();
  const size_t size = reader.size();
  const u8* data = reader.data();
  size_t cmdPos = pos - 2 - 1; // opcode byte position (before vtxCount and pos++)
  size_t dumpStart = (cmdPos > 16) ? cmdPos - 16 : 0;
  size_t dumpEnd = (cmdPos + 32 < size) ? cmdPos + 32 : size;
  std::string hex;
  for (size_t i = dumpStart; i < dumpEnd; i++) {
    if (i == cmdPos)
      hex += fmt::format("[{:02x}]", data[i]);
    else
      hex += fmt::format(" {:02x}", data[i]);
  }
  Log.error("  hex dump around draw cmd (pos {}-{}):{}", dumpStart, dumpEnd - 1, hex);
  FATAL("draw vertex data overrun: need {} bytes at pos {}, have {}", totalVtxBytes, pos, reader.remaining());
}

static u32 calc_vtx_size(GXVtxFmt fmt) noexcept {
  u32 vtxSize = 0;
  const auto& vtxFmt = g_gxState.vtxFmts[fmt];
  for (int i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7; ++i) {
    const auto& attrFmt = vtxFmt.attrs[i];
    switch (g_gxState.vtxDesc[i]) {
    case GX_NONE:
      break;
    case GX_DIRECT: {
      const auto attr = static_cast<GXAttr>(i);
      vtxSize += comp_type_size(attr, attrFmt.type) * comp_cnt_count(attr, attrFmt.cnt);
      break;
    }
    case GX_INDEX8:
      vtxSize += i == GX_VA_NRM && attrFmt.cnt == GX_NRM_NBT3 ? 3 : 1;
      break;
    case GX_INDEX16:
      vtxSize += i == GX_VA_NRM && attrFmt.cnt == GX_NRM_NBT3 ? 6 : 2;
      break;
    }
  }
  g_gxState.lastVtxFmt = fmt;
  g_gxState.lastVtxSize = vtxSize;
  return vtxSize;
}

// The fade of the draw being pushed (DrawImmediateData::fade): set only around a replay's
// cross-faded flipbook draws.
static u32 sDrawFade = 0;
// The cached pipeline reads a replay's blended arrays as floats (AttrArray::replayType).
static bool sPipelineFloatArrays = false;
// The texture coordinates the draw being pushed carries in its vertex stream (DrawCache::expandMask).
static u8 sDrawExpandMask = 0;

// Merging small draws (aurora_gx_merge_small_draws, or AURORA_MERGE_SMALL_DRAWS=1; off by
// default). A game that invalidates the vertex cache before every primitive and indexes a
// small texture coordinate array of each primitive's own makes every draw upload its arrays
// again, and none merge. With this on, an array whose bytes are those of its last upload
// reuses it, small texture coordinate arrays are carried in the vertex stream
// (expand_small_arrays), and a draw whose state comes out the same as the last draw's merges
// with it. The pictures differ only where a merged triangle shares an edge with an earlier
// one: a GPU may shade fragments of both in one 2x2 quad, so the texture's level of detail
// along the edge comes from screen derivatives across the two (Apple M-series: tens of
// pixels a picture, a few levels at most). Aurora's merging of draws with no state change at
// all does the same.
static std::atomic<bool> sMergeSmallDrawsSetting = [] {
  const char* env = std::getenv("AURORA_MERGE_SMALL_DRAWS");
  return env != nullptr && env[0] == '1';
}();
// The setting as it stood at the start of the frame being recorded (forget_array_uploads).
static bool sMergeSmallDraws = sMergeSmallDrawsSetting.load();

// The last upload of each indexed attribute's array this frame (with sMergeSmallDraws).
struct ArrayUpload {
  gfx::Range range;
  u8 replayType = 0xFF;
  u8 replayStride = 0;
  std::vector<u8> bytes;
};
static std::array<ArrayUpload, GX_VA_TEX7 + 1> sArrayUploads;
static void upload_array(int attr, GXVtxFmt fmt) noexcept {
  auto& array = g_gxState.arrays[attr];
  replay::note_read(array.data, array.size);
  if (array.cachedRange.size != 0) {
    return;
  }
  u32 size = array.size;
  const void* bytes = replay::array_bytes(static_cast<GXAttr>(attr), fmt, array, size);
  if (!sMergeSmallDraws || bytes == nullptr) {
    array.cachedRange = gfx::push_storage(static_cast<const uint8_t*>(bytes), size);
    return;
  }
  auto& last = sArrayUploads[attr];
  if (last.range.size != 0 && last.replayType == array.replayType && last.replayStride == array.replayStride &&
      last.bytes.size() == size && std::memcmp(last.bytes.data(), bytes, size) == 0) {
    array.cachedRange = last.range;
    return;
  }
  array.cachedRange = gfx::push_storage(static_cast<const uint8_t*>(bytes), size);
  last.range = array.cachedRange;
  last.replayType = array.replayType;
  last.replayStride = array.replayStride;
  const auto* b = static_cast<const u8*>(bytes);
  last.bytes.assign(b, b + size);
}

// A draw whose only change since the draw before is the immediates being marked dirty (the
// vertex cache invalidated, the current matrix set again) and whose immediates come out the
// same apart from where its vertices start: the state is the last draw's, so it is clean.
static bool immediates_unchanged(GXVtxFmt fmt) noexcept {
  const auto* last = gfx::get_last_draw_command<DrawData>();
  if (last == nullptr || last->immediateData.currentPnMtx != g_gxState.currentPnMtx ||
      last->immediateData.fade != 0 || sPipelineFloatArrays) {
    return false;
  }
  for (int i = GX_VA_POS; i <= GX_VA_TEX7; ++i) {
    if (g_gxState.vtxDesc[i] != GX_INDEX8 && g_gxState.vtxDesc[i] != GX_INDEX16) {
      continue;
    }
    upload_array(i, fmt);
    const auto& array = g_gxState.arrays[i];
    if (array.replayType != 0xFF || array.cachedRange.offset != last->immediateData.arrayStart[i - GX_VA_POS]) {
      return false;
    }
  }
  const auto& cache = sDrawCache;
  if (cache.config.shaderConfig.fogRangeEnabled &&
      (!cache.hasFogRange || cache.fogRangeKey != fog_range_lut_key() ||
       last->immediateData.fogRangeBase != cache.fogRange.offset / sizeof(u32))) {
    return false;
  }
  return true;
}
// Textures a replay's cross-faded draw binds instead of the state's (its previous cell's).
static std::span<const replay::PreviousTexture> sTextureOverride;

static void push_gx_draw(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, gfx::Range vertRange, gfx::Range idxRange,
                         u32 numIndices) noexcept {
  auto& state = g_gxState;
  auto& cache = sDrawCache;

  DrawImmediateData immediates{.vtxStart = vertRange.offset, .currentPnMtx = state.currentPnMtx, .fade = sDrawFade};
  bool floatArrays = false;
  for (int i = GX_VA_POS; i <= GX_VA_TEX7; ++i) {
    if (state.vtxDesc[i] != GX_INDEX8 && state.vtxDesc[i] != GX_INDEX16) {
      continue;
    }
    upload_array(i, fmt);
    auto& array = state.arrays[i];
    floatArrays |= array.replayType != 0xFF;
    immediates.arrayStart[i - GX_VA_POS] = array.cachedRange.offset;
  }
  // A replay's blended arrays read as floats need their own pipeline, and so does the next
  // draw after one.
  if (floatArrays || sPipelineFloatArrays) {
    state.dirty |= DirtyPipeline;
  }

  const u8 lineMode = line_mode_for_prim(prim);
  const auto targetLayoutKey = gfx::get_render_target_layout().key;
  const bool pipelineValid = cache.hasPipeline && (state.dirty & DirtyPipeline) == 0 && cache.fmt == fmt &&
                             cache.lineMode == lineMode && cache.targetLayoutKey == targetLayoutKey;
  if (!pipelineValid) {
    const bool hadPipeline = cache.hasPipeline;
    const auto prevSampledTextures = cache.shaderInfo.sampledTextures;
    const auto prevSampledIndTextures = cache.shaderInfo.sampledIndTextures;
    populate_pipeline_config(cache.config, prim, fmt);
    cache.expandMask = sDrawExpandMask;
    sPipelineFloatArrays = floatArrays;
    cache.shaderInfo = build_shader_info(cache.config.shaderConfig);
    cache.pipelineRef = gfx::pipeline_ref(cache.config);
    cache.targetLayoutKey = targetLayoutKey;
    cache.fmt = fmt;
    cache.lineMode = lineMode;
    cache.hasPipeline = true;
    state.dirty = (state.dirty & ~DirtyPipeline) | DirtyUniform;
    if (!hadPipeline || prevSampledTextures != cache.shaderInfo.sampledTextures ||
        prevSampledIndTextures != cache.shaderInfo.sampledIndTextures) {
      cache.bindGeneration = 0;
    }
  }

  const bool bindGroupsValid =
      (state.dirty & DirtyTextures) == 0 && cache.bindGeneration == texture::current_bind_generation();
  if (!bindGroupsValid) {
    const auto prevBindGroup = cache.bindGroups.textureBindGroup;
    resolve_sampled_textures(cache.shaderInfo);
    for (const auto& t : sTextureOverride) {
      g_gxState.textures[t.map] = t.bind;
    }
    cache.bindGroups = build_bind_groups(cache.shaderInfo);
    cache.bindGeneration = texture::current_bind_generation();
    state.dirty &= ~DirtyTextures;
    // For texture_size_bias uniform
    if (cache.bindGroups.textureBindGroup != prevBindGroup) {
      state.dirty |= DirtyUniform;
    }
  }

  const bool uniformValid = (state.dirty & DirtyUniform) == 0 && cache.uniformRange.size != 0;
  if (!uniformValid) {
    replay::before_uniform(prim, fmt, vtxCount);
    cache.uniformRange = build_uniform(cache.shaderInfo);
    replay::after_uniform();
    state.dirty &= ~DirtyUniform;
  }
  if (cache.config.shaderConfig.fogRangeEnabled) {
    const auto key = fog_range_lut_key();
    if (!cache.hasFogRange || cache.fogRangeKey != key) {
      cache.fogRange = push_fog_range_lut(key);
      cache.fogRangeKey = key;
      cache.hasFogRange = true;
    }
  }
  immediates.fogRangeBase = cache.fogRange.offset / sizeof(u32);

  state.dirty &= ~DirtyImmediates;

  uint32_t instanceCount = 1;
  if (prim == GX_LINES) {
    instanceCount = vtxCount / 2;
  } else if (prim == GX_LINESTRIP) {
    instanceCount = vtxCount - 1;
  } else if (prim == GX_POINTS) {
    instanceCount = vtxCount;
  }
  cache.lastDrawFmt = fmt;
  gfx::push_draw_command(DrawData{
      .pipeline = cache.pipelineRef,
      .bindGroups = cache.bindGroups,
      .vertRange = vertRange,
      .idxRange = idxRange,
      .uniformRange = cache.uniformRange,
      .immediateData = immediates,
      .vtxCount = vtxCount,
      .indexCount = numIndices,
      .instanceCount = instanceCount,
      .dstAlpha = state.dstAlpha,
  });
}

static void handle_draw_unmerged(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, gfx::Range vertRange) noexcept {
  ZoneScoped;
  u32 numIndices = 0;
  gfx::Range idxRange;

  if (prim != GX_TRIANGLES) {
    ZoneScopedN("build idx buffer");
    static ByteBuffer idxBuf;
    numIndices = prepare_idx_buffer(idxBuf, prim, 0, vtxCount);
    idxRange = gfx::push_indices(idxBuf.data(), idxBuf.size(), 4);
    idxBuf.clear();
  }

  push_gx_draw(prim, fmt, vtxCount, vertRange, idxRange, numIndices);
}

// Small indexed texture coordinate arrays, carried in the vertex stream. Skies of Arcadia
// points GX_VA_TEX0 at a few coordinates of each primitive's own (18 to 66 bytes) and
// invalidates the vertex cache before every primitive, so every draw had arrays of its own
// and none merged: a busy village made 3,085 draws of about ten vertices, and on a tile-based
// GPU each draw's set-up is paid in the binning pass and again in every bin. An expanded
// coordinate is the array's bytes (big-endian, as the stream's are) in place of its index,
// read by the shader as a direct attribute with the same conversion, so the values are the
// same. Not for a replay's blended arrays (read as floats), an index past its array, or
// NBT normals. Only with sMergeSmallDraws.
constexpr u32 MaxExpandedArray = 512;

struct ExpandedDraw {
  u8 mask = 0;
};

// Sets the expanded coordinates' vertex descriptors to direct while the draw is pushed.
struct VtxDescOverride {
  u8 mask;
  std::array<GXAttrType, GX_VA_TEX7 + 1> saved{};
  explicit VtxDescOverride(u8 m) : mask(m) {
    for (int t = 0; t < 8; ++t) {
      if (mask & (1u << t)) {
        saved[GX_VA_TEX0 + t] = g_gxState.vtxDesc[GX_VA_TEX0 + t];
        g_gxState.vtxDesc[GX_VA_TEX0 + t] = GX_DIRECT;
      }
    }
  }
  ~VtxDescOverride() {
    for (int t = 0; t < 8; ++t) {
      if (mask & (1u << t)) {
        g_gxState.vtxDesc[GX_VA_TEX0 + t] = saved[GX_VA_TEX0 + t];
      }
    }
  }
  VtxDescOverride(const VtxDescOverride&) = delete;
  VtxDescOverride& operator=(const VtxDescOverride&) = delete;
};

static ExpandedDraw expand_small_arrays(GXVtxFmt fmt, u16 vtxCount, u32 vtxSize, const u8* vertices,
                                        ByteBuffer& out) noexcept {
  ExpandedDraw result;
  if (!sMergeSmallDraws || vtxCount == 0) {
    return result;
  }
  const auto& vtxFmt = g_gxState.vtxFmts[fmt];
  struct Source {
    const u8* bytes = nullptr;
    u32 size = 0;
    u32 elemBytes = 0;
    u8 compSize = 0;
    bool swap = false;
  };
  std::array<Source, 8> sources{};
  for (int t = 0; t < 8; ++t) {
    const int i = GX_VA_TEX0 + t;
    const auto type = g_gxState.vtxDesc[i];
    if (type != GX_INDEX8 && type != GX_INDEX16) {
      continue;
    }
    auto& array = g_gxState.arrays[i];
    if (array.data == nullptr || array.size == 0 || array.size > MaxExpandedArray || array.stride == 0) {
      continue;
    }
    const auto attr = static_cast<GXAttr>(i);
    const auto& attrFmt = vtxFmt.attrs[i];
    const u8 compSize = comp_type_size(attr, attrFmt.type);
    const u32 elemBytes = compSize * comp_cnt_count(attr, attrFmt.cnt);
    if (elemBytes > array.stride) {
      continue;
    }
    replay::note_read(array.data, array.size);
    u32 size = array.size;
    const void* bytes = replay::array_bytes(attr, fmt, array, size);
    if (array.replayType != 0xFF || bytes == nullptr) {
      continue;
    }
    sources[t] = Source{static_cast<const u8*>(bytes), size, elemBytes, compSize, array.le && compSize > 1};
    result.mask |= 1u << t;
  }
  if (result.mask == 0) {
    return result;
  }
  // The stream's layout, attribute by attribute, as populate_pipeline_config lays it out.
  u32 newSize = vtxSize;
  for (int t = 0; t < 8; ++t) {
    if (result.mask & (1u << t)) {
      newSize += sources[t].elemBytes - (g_gxState.vtxDesc[GX_VA_TEX0 + t] == GX_INDEX8 ? 1 : 2);
    }
  }
  out.clear();
  out.append_zeroes(size_t(newSize) * vtxCount);
  u8* dst = out.data();
  const u8* src = vertices;
  for (u16 v = 0; v < vtxCount; ++v) {
    for (int i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7; ++i) {
      const auto type = g_gxState.vtxDesc[i];
      if (type == GX_NONE) {
        continue;
      }
      const auto attr = static_cast<GXAttr>(i);
      const auto& attrFmt = vtxFmt.attrs[i];
      u32 len;
      if (type == GX_DIRECT) {
        len = comp_type_size(attr, attrFmt.type) * comp_cnt_count(attr, attrFmt.cnt);
      } else {
        const bool nbt3 = i == GX_VA_NRM && attrFmt.cnt == GX_NRM_NBT3;
        len = type == GX_INDEX8 ? (nbt3 ? 3 : 1) : (nbt3 ? 6 : 2);
      }
      const int t = i - GX_VA_TEX0;
      if (t >= 0 && (result.mask & (1u << t))) {
        const auto& source = sources[t];
        const u32 index = type == GX_INDEX8 ? src[0] : (u32(src[0]) << 8 | src[1]);
        const u32 at = index * g_gxState.arrays[i].stride;
        if (at + source.elemBytes > source.size) {
          return {}; // an index past its array: leave the draw as the game made it
        }
        const u8* elem = source.bytes + at;
        if (source.swap) {
          for (u32 c = 0; c < source.elemBytes; c += source.compSize) {
            for (u32 b = 0; b < source.compSize; ++b) {
              dst[c + b] = elem[c + source.compSize - 1 - b];
            }
          }
        } else {
          std::memcpy(dst, elem, source.elemBytes);
        }
        dst += source.elemBytes;
      } else {
        std::memcpy(dst, src, len);
        dst += len;
      }
      src += len;
    }
  }
  return result;
}

static void draw_prim(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, ByteReader& reader) noexcept {
  ZoneScoped;
  u32 vtxSize;
  if (g_gxState.lastVtxFmt == fmt)
    LIKELY { vtxSize = g_gxState.lastVtxSize; }
  else
    UNLIKELY { vtxSize = calc_vtx_size(fmt); }

  u32 totalVtxBytes = vtxCount * vtxSize;
  if (totalVtxBytes > reader.remaining())
    UNLIKELY {
      if (replay::replaying()) {
        replay::fail("draw overrun");
        reader.skip(reader.remaining());
        return;
      }
      handle_draw_overrun(totalVtxBytes, reader);
    }

  // Push raw vertex data to buffer. Merged draws must remain contiguous with the previous range.
  const auto vertexData = reader.take(totalVtxBytes);
  if (trace_2d::enabled() && !replay::replaying())
    UNLIKELY { trace_2d::draw(prim, fmt, vtxCount, vtxSize, vertexData.data()); }
  const u8* vertices = replay::vertices(prim, fmt, vtxCount, vtxSize, vertexData.data());
  replay::Crossfade fade;
  if (replay::crossfade(fade))
    UNLIKELY {
      // A replay's cross-faded flipbook: this tick's cell scaled by weight, the previous
      // tick's by 1 - weight (with its own textures, when it swapped them), each a draw of
      // its own (a draw's fade is per draw).
      if (sDrawCache.expandMask != 0) {
        g_gxState.dirty |= DirtyPipeline;
      }
      gfx::Range vertRange = gfx::push_verts(vertices, vertexData.size(), 4);
      sDrawFade = std::bit_cast<u32>(1.f - fade.weight);
      handle_draw_unmerged(prim, fmt, vtxCount, vertRange);
      vertRange = gfx::push_verts(fade.previous, vertexData.size(), 4);
      sDrawFade = std::bit_cast<u32>(fade.weight);
      std::array<gfx::TextureBind, MaxTextures> saved;
      if (!fade.textures.empty()) {
        for (const auto& t : fade.textures) {
          saved[t.map] = g_gxState.textures[t.map];
        }
        sTextureOverride = fade.textures;
        g_gxState.dirty |= DirtyTextures;
      }
      handle_draw_unmerged(prim, fmt, vtxCount, vertRange);
      if (!fade.textures.empty()) {
        for (const auto& t : fade.textures) {
          g_gxState.textures[t.map] = saved[t.map];
        }
        sTextureOverride = {};
        g_gxState.dirty |= DirtyTextures;
      }
      sDrawFade = 0;
      return;
    }
  // Texture coordinates the game indexes from small arrays of their own are carried in the
  // vertex stream instead, so the draws merge (expand_small_arrays).
  static ByteBuffer expanded;
  const ExpandedDraw exp = expand_small_arrays(fmt, vtxCount, vtxSize, vertices, expanded);
  if (exp.mask != 0) {
    vertices = expanded.data();
  }
  const size_t vertexBytes = exp.mask != 0 ? expanded.size() : vertexData.size();
  sDrawExpandMask = exp.mask;
  if (exp.mask != sDrawCache.expandMask) {
    g_gxState.dirty |= DirtyPipeline;
  }
  const VtxDescOverride vtxDescOverride{exp.mask};

  if (sMergeSmallDraws && g_gxState.dirty == DirtyImmediates && fmt == sDrawCache.lastDrawFmt &&
      sDrawCache.lineMode == 0 && prim != GX_LINES && prim != GX_LINESTRIP && prim != GX_POINTS &&
      immediates_unchanged(fmt)) {
    g_gxState.dirty = 0;
  }
  const bool cleanState = g_gxState.dirty == 0 && fmt == sDrawCache.lastDrawFmt && sDrawCache.lineMode == 0 &&
                          prim != GX_LINES && prim != GX_LINESTRIP && prim != GX_POINTS;
  auto* lastDraw = cleanState ? gfx::get_last_draw_command<DrawData>() : nullptr;
  {
    // Profiling counts (AURORA_GPU_TIMES line): why a draw did not merge with the one before.
    auto& c = g_mergeCounts;
    auto& c2 = g_mergeImmCounts;
    const u32 d = g_gxState.dirty;
    if (d & DirtyPipeline) c[0].fetch_add(1, std::memory_order_relaxed);
    if (d & DirtyTextures) c[1].fetch_add(1, std::memory_order_relaxed);
    if (d & DirtyUniform) c[2].fetch_add(1, std::memory_order_relaxed);
    if (d & DirtyImmediates) c[3].fetch_add(1, std::memory_order_relaxed);
    if (d == 0 && fmt != sDrawCache.lastDrawFmt) c[4].fetch_add(1, std::memory_order_relaxed);
    if (cleanState && lastDraw == nullptr) c[5].fetch_add(1, std::memory_order_relaxed);
    if (lastDraw != nullptr) c[6].fetch_add(1, std::memory_order_relaxed);
    c[7].fetch_add(1, std::memory_order_relaxed);
    if (d == DirtyImmediates && fmt == sDrawCache.lastDrawFmt && sDrawCache.lineMode == 0 && prim != GX_LINES &&
        prim != GX_LINESTRIP && prim != GX_POINTS) {
      if (const auto* last = gfx::get_last_draw_command<DrawData>()) {
        bool pnmtx = last->immediateData.currentPnMtx != g_gxState.currentPnMtx;
        bool invalidated = false;
        bool moved = false;
        for (int i = GX_VA_POS; i <= GX_VA_TEX7; ++i) {
          if (g_gxState.vtxDesc[i] != GX_INDEX8 && g_gxState.vtxDesc[i] != GX_INDEX16) {
            continue;
          }
          const auto& array = g_gxState.arrays[i];
          if (array.cachedRange.size == 0) {
            invalidated = true;
          } else if (array.cachedRange.offset != last->immediateData.arrayStart[i - GX_VA_POS]) {
            moved = true;
          }
        }
        if (pnmtx) c2[0].fetch_add(1, std::memory_order_relaxed);
        if (invalidated) c2[1].fetch_add(1, std::memory_order_relaxed);
        if (moved) c2[2].fetch_add(1, std::memory_order_relaxed);
        if (!pnmtx && !invalidated && !moved) c2[3].fetch_add(1, std::memory_order_relaxed);
      }
    }
  }

  const bool canMerge = lastDraw != nullptr && lastDraw->instanceCount == 1 && lastDraw->immediateData.fade == 0;
  gfx::Range vertRange = gfx::push_verts(vertices, vertexBytes, canMerge ? 0 : 4);

  // Try to merge with previous draw call
  if (canMerge) {
    u32 numIndices = 0;
    gfx::Range idxRange;
    static ByteBuffer idxBuf;
    const bool hadIndexRange = lastDraw->idxRange.size != 0;
    if (lastDraw->indexCount == 0 && prim != GX_TRIANGLES) {
      // Generate triangle index buffer for previous draw
      lastDraw->indexCount = prepare_idx_buffer(idxBuf, GX_TRIANGLES, 0, lastDraw->vtxCount);
    }
    if (lastDraw->indexCount != 0) {
      numIndices += prepare_idx_buffer(idxBuf, prim, lastDraw->vtxCount, vtxCount);
      idxRange = gfx::push_indices(idxBuf.data(), idxBuf.size(), hadIndexRange ? 0 : 4);
      idxBuf.clear();
    }
    CHECK(lastDraw->vertRange.offset + lastDraw->vertRange.size == vertRange.offset,
          "Non-consecutive vertex ranges ({} < {})", lastDraw->vertRange.offset + lastDraw->vertRange.size,
          vertRange.offset);
    if (hadIndexRange) {
      CHECK(lastDraw->idxRange.offset + lastDraw->idxRange.size == idxRange.offset,
            "Non-consecutive index ranges ({} < {})", lastDraw->idxRange.offset + lastDraw->idxRange.size,
            idxRange.offset);
    }
    lastDraw->vertRange.size += vertRange.size;
    if (lastDraw->idxRange.size == 0) {
      lastDraw->idxRange = idxRange;
    } else {
      lastDraw->idxRange.size += idxRange.size;
    }
    lastDraw->vtxCount += vtxCount;
    lastDraw->indexCount += numIndices;
    gfx::detail::increment_merged_draw_count();
    replay::note_draw_textures();
    return;
  }

  handle_draw_unmerged(prim, fmt, vtxCount, vertRange);
  replay::note_draw_textures();
}

static void handle_draw(u8 cmd, ByteReader& reader) noexcept {
  const auto fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
  const auto prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
  draw_prim(prim, fmt, reader.read<u16>(), reader);
}

void handle_aurora(ByteReader& reader) noexcept {
  ZoneScoped;
  const u16 subCmd = reader.read<u16>();

  if (subCmd == GX_AURORA_LOAD_VIEWPORT_RENDER) {
    const f32 left = reader.read<f32>();
    const f32 top = reader.read<f32>();
    const f32 width = reader.read<f32>();
    const f32 height = reader.read<f32>();
    const f32 nearZ = reader.read<f32>();
    const f32 farZ = reader.read<f32>();
    set_render_viewport({
        .left = left,
        .top = top,
        .width = width,
        .height = height,
        .znear = nearZ,
        .zfar = farZ,
    });
  } else if (subCmd == GX_AURORA_LOAD_SCISSOR_RENDER) {
    const s32 left = reader.read<s32>();
    const s32 top = reader.read<s32>();
    const s32 width = reader.read<s32>();
    const s32 height = reader.read<s32>();
    set_render_scissor({left, top, width, height});
  } else if (subCmd == GX_AURORA_LOAD_PROJECTION_FULL) {
    auto& proj = g_gxState.proj;
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        proj[r][c] = reader.read<f32>();
      }
    }
    // Invalidate projection XF regs
    for (u32 reg = 0x20; reg <= 0x26; ++reg) {
      g_gxState.xfRegValid.reset(reg);
    }
    g_gxState.dirty |= DirtyUniform;
  } else if (subCmd >= GX_AURORA_LOAD_ARRAYBASE && subCmd <= (GX_AURORA_LOAD_ARRAYBASE | 0x0f)) {
    const u32 attrIdx = subCmd - GX_AURORA_LOAD_ARRAYBASE + GX_VA_POS;
    const u64 arrayAddr = reader.read<u64>();
    const u32 arraySize = reader.read<u32>();
    const bool le = reader.read<u8>() == 1;

    auto& array = g_gxState.arrays[attrIdx];
    const auto newData = reinterpret_cast<void*>(arrayAddr);
    if (array.data != newData || array.size != arraySize || array.le != le) {
      if (array.le != le) {
        // Endianness is baked into the shader
        g_gxState.dirty |= DirtyPipeline;
      }
      array.data = newData;
      array.size = arraySize;
      array.le = le;
      array.cachedRange = {};
      g_gxState.dirty |= DirtyImmediates;
    }
  } else if (subCmd == GX_AURORA_LOAD_TEXOBJ) {
    const auto texMapId = reader.read<u8>();
    CHECK(texMapId < MaxTextures, "invalid texture map id {}", texMapId);
    auto& slot = g_gxState.loadedTextures[texMapId];
    const auto newData = reinterpret_cast<const void*>(reader.read<u64>());
    const u32 newWidth = reader.read<u32>();
    const u32 newHeight = reader.read<u32>();
    const auto newFormat = static_cast<GXTexFmt>(reader.read<u32>());
    const auto newTlut = static_cast<GXTlut>(reader.read<u32>());
    u8 newFlags = slot.flags & ~0x80u; // Reset no-cache flag
    if (reader.read<u8>() != 0) {
      newFlags |= 1u;
    } else {
      newFlags &= ~1u;
    }
    const u32 newTexObjId = reader.read<u32>();
    const u32 newTexDataVersion = reader.read<u32>();
    if (slot.data != newData || slot.mWidth != newWidth || slot.mHeight != newHeight ||
        slot.mFormat != static_cast<u32>(newFormat) || slot.tlut != newTlut || slot.flags != newFlags ||
        slot.texObjId != newTexObjId || slot.texDataVersion != newTexDataVersion) {
      slot.data = newData;
      slot.mWidth = newWidth;
      slot.mHeight = newHeight;
      slot.mFormat = newFormat;
      slot.tlut = newTlut;
      slot.flags = newFlags;
      slot.texObjId = newTexObjId;
      slot.texDataVersion = newTexDataVersion;
      g_gxState.dirty |= DirtyTextures;
    }
    if (slot.has_data()) {
      replay::note_read(slot.data, texture::texture_source_size(slot.format(), slot.width(), slot.height(),
                                                                slot.mip_count()));
    }
  } else if (subCmd == GX_AURORA_LOAD_TLUT) {
    const auto idx = reader.read<u8>();
    CHECK(idx < MaxTluts, "invalid tlut slot {}", idx);
    auto& slot = g_gxState.loadedTluts[idx];
    const auto newData = reinterpret_cast<const void*>(reader.read<u64>());
    const auto newFormat = static_cast<GXTlutFmt>(reader.read<u32>());
    const u16 newNumEntries = reader.read<u16>();
    const u32 newTlutObjId = reader.read<u32>();
    const u32 newTlutDataVersion = reader.read<u32>();
    const u8 newFlags = slot.flags & ~0x80u; // Reset no-cache flag
    if (slot.data != newData || slot.format != newFormat || slot.numEntries != newNumEntries ||
        slot.tlutObjId != newTlutObjId || slot.tlutDataVersion != newTlutDataVersion || slot.flags != newFlags) {
      if (slot.tlutObjId != newTlutObjId || slot.tlutDataVersion != newTlutDataVersion) {
        texture::invalidate_bindings();
      }
      slot.data = newData;
      slot.format = newFormat;
      slot.numEntries = newNumEntries;
      slot.tlutObjId = newTlutObjId;
      slot.tlutDataVersion = newTlutDataVersion;
      slot.flags = newFlags;
      g_gxState.dirty |= DirtyTextures;
    }
    replay::note_read(slot.data, texture::tlut_source_size(slot.numEntries));
  } else if (subCmd == GX2_SET_POLYGON_OFFSET) {
    const f32 frontOffset = reader.read<f32>();
    const f32 frontScale = reader.read<f32>();
    const f32 backOffset = reader.read<f32>();
    const f32 backScale = reader.read<f32>();
    const f32 clamp = reader.read<f32>();
    if (g_gxState.frontOffset != frontOffset || g_gxState.frontScale != frontScale ||
        g_gxState.backOffset != backOffset || g_gxState.backScale != backScale || g_gxState.clamp != clamp) {
      g_gxState.frontOffset = frontOffset;
      g_gxState.frontScale = frontScale;
      g_gxState.backOffset = backOffset;
      g_gxState.backScale = backScale;
      g_gxState.clamp = clamp;
      g_gxState.dirty |= DirtyPipeline;
    }
  } else if (subCmd == GX_AURORA_LOAD_COPY_SRC) {
    const s32 left = reader.read<s32>();
    const s32 top = reader.read<s32>();
    const s32 width = reader.read<s32>();
    const s32 height = reader.read<s32>();
    g_gxState.texCopySrc = {left, top, width, height};
  } else if (subCmd == GX_AURORA_LOAD_COPY_DST) {
    g_gxState.texCopyDstWidth = reader.read<u32>();
    g_gxState.texCopyDstHeight = reader.read<u32>();
    g_gxState.texCopyFmt = static_cast<GXTexFmt>(reader.read<u32>());
    reader.skip(1); // mipmap is not implemented, but remains part of the command payload
    g_gxState.texCopyDstWide = true;
  } else if (subCmd == GX_AURORA_LOAD_COPY_DEST) {
    g_gxState.texCopyDest = reinterpret_cast<const void*>(reader.read<u64>());
  } else if (subCmd == GX_AURORA_REQUEST_DEPTH_SNAPSHOT) {
    if (!replay::replaying()) {
      gfx::depth_peek::request_snapshot();
    }
  } else if (subCmd == GX_AURORA_BEGIN_OFFSCREEN) {
    const u32 width = reader.read<u32>();
    const u32 height = reader.read<u32>();
    gfx::begin_offscreen(width, height);
  } else if (subCmd == GX_AURORA_END_OFFSCREEN) {
    if (replay::replaying() && !gfx::is_offscreen()) {
      // The pass was opened outside the stream (aurora::gfx::create_pass), so the replay
      // never saw it begin: stop the replay rather than close a pass that is not open.
      replay::fail("an offscreen pass ends that the stream did not open");
      return;
    }
    gfx::end_offscreen();
  } else if (subCmd == GX_AURORA_DESTROY_TEXOBJ) {
    const u32 id = reader.read<u32>();
    if (!replay::replaying()) { // the first decode evicted it
      evict_texture_object(id);
    }
  } else if (subCmd == GX_AURORA_DESTROY_TLUT) {
    const u32 id = reader.read<u32>();
    if (!replay::replaying()) {
      evict_tlut_object(id);
    }
  } else if (subCmd == GX_AURORA_DESTROY_COPY_TEX) {
    const auto* dest = reinterpret_cast<const void*>(reader.read<u64>());
    if (!replay::replaying()) {
      evict_copy_texture(dest);
    }
  } else if (subCmd == GX_AURORA_DRAW_SIZED) {
    const u8 cmd = reader.read<u8>();
    const u32 byteLen = reader.read<u32>();
    const GXVtxFmt fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
    const GXPrimitive prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
    if (byteLen != 0) {
      u32 vtxSize;
      if (g_gxState.lastVtxFmt == fmt) {
        vtxSize = g_gxState.lastVtxSize;
      } else {
        vtxSize = calc_vtx_size(fmt);
      }
      AURORA_ASSERT(vtxSize != 0 && byteLen % vtxSize == 0,
                    "GX_AURORA_DRAW_SIZED: {} bytes is not a whole number of size-{} vertices", byteLen, vtxSize);
      u32 vtxCount = byteLen / vtxSize;
      AURORA_ASSERT(vtxCount <= 0xFFFF, "GX_AURORA_DRAW_SIZED: too many vertices ({})", vtxCount);
      draw_prim(prim, fmt, static_cast<u16>(vtxCount), reader);
    }
  } else if (subCmd == GX_AURORA_DRAW_INDEXED) {
    ZoneScopedN("DRAW_INDEXED");
    const u8 cmd = reader.read<u8>();
    const u16 vtxCount = reader.read<u16>();
    const u32 indexCount = reader.read<u32>();
    const GXVtxFmt fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
    const GXPrimitive prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
    AURORA_ASSERT(prim == GX_TRIANGLES, "GX_AURORA_DRAW_INDEXED: primitive must be GX_TRIANGLES, got {}",
                  static_cast<u32>(prim));
    const size_t idxBytes = static_cast<size_t>(indexCount) * sizeof(u16);
    // Index data is always host-endian; push it to the GPU buffer as-is
    const auto indexData = reader.take(idxBytes);
    const gfx::Range idxRange = gfx::push_indices(indexData.data(), indexData.size(), 4);
    u32 vtxSize;
    if (g_gxState.lastVtxFmt == fmt) {
      vtxSize = g_gxState.lastVtxSize;
    } else {
      vtxSize = calc_vtx_size(fmt);
    }
    const u32 totalVtxBytes = vtxCount * vtxSize;
    const auto vertexData = reader.take(totalVtxBytes);
    if (trace_2d::enabled() && !replay::replaying())
      UNLIKELY { trace_2d::draw(prim, fmt, vtxCount, vtxSize, vertexData.data()); }
    const u8* vertices = replay::vertices(prim, fmt, vtxCount, vtxSize, vertexData.data());
    const gfx::Range vertRange = gfx::push_verts(vertices, vertexData.size(), 4);
    if (indexCount != 0) {
      push_gx_draw(prim, fmt, vtxCount, vertRange, idxRange, indexCount);
    }
    replay::note_draw_textures();
  } else if (subCmd == GX_AURORA_DEBUG_GROUP_PUSH) {
    auto label = reader.read_string();
    gfx::push_debug_group(std::move(label));
  } else if (subCmd == GX_AURORA_DEBUG_GROUP_POP) {
    pop_debug_group();
  } else if (subCmd == GX_AURORA_DEBUG_MARKER_INSERT) {
    auto label = reader.read_string();
    gfx::insert_debug_marker(std::move(label));
  } else if (subCmd == GX_AURORA_REPLAY_VIEW) {
    std::array<f32, 12> view{};
    const bool has = reader.read<u8>() != 0;
    if (has) {
      for (auto& v : view) {
        v = reader.read<f32>();
      }
    }
    replay::on_view(has ? view.data() : nullptr);
  } else if (subCmd == GX_AURORA_REPLAY_TASK) {
    replay::on_task(reader.read<u64>());
  }

  else {
    Log.error("Unknown Aurora subcommand: {:04X}", subCmd);
  }
}

void clear_draw_cache() noexcept {
  sDrawCache.bindGeneration = 0;
  sDrawCache.uniformRange = {};
  sDrawCache.fogRange = {};
  sDrawCache.hasFogRange = false;
}

namespace {
std::array<DrawCache, 2> sSavedDrawCache;
} // namespace

void set_merge_small_draws(bool enabled) noexcept { sMergeSmallDrawsSetting.store(enabled, std::memory_order_relaxed); }

void forget_array_uploads() noexcept {
  sMergeSmallDraws = sMergeSmallDrawsSetting.load(std::memory_order_relaxed);
  for (auto& upload : sArrayUploads) {
    upload.range = {};
  }
}

void save_draw_cache(int slot) noexcept {
  auto& saved = sSavedDrawCache[slot];
  saved = sDrawCache;
  // As clear_draw_cache leaves it between frames: nothing points into a frame's buffers.
  saved.bindGeneration = 0;
  saved.uniformRange = {};
  saved.fogRange = {};
  saved.hasFogRange = false;
}

void restore_draw_cache(int slot) noexcept { sDrawCache = sSavedDrawCache[slot]; }

} // namespace aurora::gx::fifo

void aurora_gx_merge_small_draws(bool enabled) { aurora::gx::fifo::set_merge_small_draws(enabled); }
