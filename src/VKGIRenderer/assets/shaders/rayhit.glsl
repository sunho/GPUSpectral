#version 460
#pragma shader_stage(closest)

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_scalar_block_layout : require

layout(location = 0) rayPayloadInEXT vec3 hitValue;
layout(location = 2) rayPayloadEXT bool shadowed;
hitAttributeEXT vec3 attribs;

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer PositionBuffer
{
   	vec3 positions[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer NormalBuffer
{
   	vec3 normals[];
};

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;

struct Instance {
	mat4 transformInvT;
	uvec2 positionBuffer;
	uvec2 normalBuffer;
};

layout(binding = 2, std430, set = 0) buffer InstanceBuffer { 
	Instance instances[];
 } instanceBuffer;

struct Vertex
{
  vec3 pos;
  vec3 normal;
  vec2 uv;
 };
 
void main()
{
	ivec3 index = ivec3(3 * gl_PrimitiveID, 3 * gl_PrimitiveID + 1, 3 * gl_PrimitiveID + 2);
	Instance instance = instanceBuffer.instances[gl_InstanceID];
	PositionBuffer posBuffer = PositionBuffer(instance.positionBuffer);
	NormalBuffer normalBuffer = NormalBuffer(instance.normalBuffer);

	vec3 pos0 = posBuffer.positions[index.x];
	vec3 pos1 = posBuffer.positions[index.y];
	vec3 pos2 = posBuffer.positions[index.z];
	pos0 = (gl_ObjectToWorldEXT * vec4(pos0, 1.0)).xyz;
	pos1 = (gl_ObjectToWorldEXT * vec4(pos1, 1.0)).xyz;
	pos2 = (gl_ObjectToWorldEXT * vec4(pos2, 1.0)).xyz;

	vec3 normal0 = normalBuffer.normals[index.x];
	vec3 normal1 = normalBuffer.normals[index.y];
	vec3 normal2 = normalBuffer.normals[index.z];
	normal0 = (instance.transformInvT * vec4(normal0, 0.0)).xyz;
	normal1 = (instance.transformInvT * vec4(normal1, 0.0)).xyz;
	normal2 = (instance.transformInvT * vec4(normal2, 0.0)).xyz;

	const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
	vec3 position = barycentrics.x * pos0 + barycentrics.y * pos1 + barycentrics.z * pos2;
	vec3 normal = barycentrics.x * normal0 + barycentrics.y * normal1 + barycentrics.z * normal2;
	hitValue = position;
}