#pragma once

#include "frame_packet.hpp"

namespace aurora::gfx {

bool bind_pipeline(PipelineRef ref, const wgpu::RenderPassEncoder& pass);
void bind_pipeline(const wgpu::RenderPipeline& pipeline, const wgpu::RenderPassEncoder& pass);
// Binds the shared 16-bit index buffer whole, once a pass (a draw picks its range with firstIndex);
// a caller that binds another index buffer calls forget_index_buffer afterwards.
void bind_index_buffer(const wgpu::RenderPassEncoder& pass);
void forget_index_buffer();
// AURORA_DEBUG_DRAWS=empty: draws keep their state changes but draw no instances (profiling aid).
bool debug_empty_draws();
// AURORA_DEBUG_DRAWS=oneuniform|onetexture: true when this draw should skip that bind.
bool debug_bind_once(bool texture);
// Profiling counts since the last call (gpu_prof's AURORA_GPU_TIMES line): pipelines bound,
// texture bind groups set and gx draws encoded.
void note_texture_bind() noexcept;
void note_gx_draw(uint64_t vertices) noexcept;
void take_encode_counts(uint64_t& pipelines, uint64_t& textures, uint64_t& draws, uint64_t& vertices) noexcept;

namespace detail {
void encode_op(wgpu::CommandEncoder& encoder, FramePacket& frame, const FrameOp& op);
// Only the op's staging copies and texture uploads (a discarded frame's op).
void encode_op_uploads(wgpu::CommandEncoder& encoder, FramePacket& frame, const FrameOp& op);
}

} // namespace aurora::gfx
