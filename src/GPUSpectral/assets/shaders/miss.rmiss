#version 460
#pragma shader_stage(miss)

#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_scalar_block_layout : require

#include "pt_common.glsl"

layout(location = 0) rayPayloadInEXT HitPayload prd;

void main()
{
    prd.done = 1;
}