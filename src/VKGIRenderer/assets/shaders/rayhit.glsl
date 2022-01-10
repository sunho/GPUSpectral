#version 460
#pragma shader_stage(closest)

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) rayPayloadInEXT vec3 hitValue;
layout(location = 2) rayPayloadEXT bool shadowed;
hitAttributeEXT vec3 attribs;

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
// layout(binding = 2, set = 0) buffer Vertices { vec4 v[]; } vertices;

struct Vertex
{
  vec3 pos;
  vec3 normal;
  vec2 uv;
 };
void main()
{
	ivec3 index = ivec3(3 * gl_PrimitiveID, 3 * gl_PrimitiveID + 1, 3 * gl_PrimitiveID + 2);
	hitValue = vec3(index) / 256.0;
}