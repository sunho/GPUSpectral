#version 460
#pragma shader_stage(raygen)

#extension GL_EXT_ray_tracing : require

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 1, set = 0, rgba8) uniform image2D image;

layout(location = 0) rayPayloadEXT vec3 hitValue;

vec3 rayDir(vec2 size, vec2 fragCoord, float fov) {
    vec2 xy = fragCoord - size / 2.0f;
    float z = (max(size.x,size.y)/2.0f) / tan(fov / 2.0f);
    vec3 dir = normalize(vec3(-xy.x, xy.y, z));
    return dir;
}

void main() 
{
	const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
	const vec2 inUV = pixelCenter/vec2(gl_LaunchSizeEXT.xy);
	vec2 d = inUV * 2.0 - 1.0;

	vec4 origin = vec4(0,1,4,1);
	vec4 direction = vec4(rayDir(vec2(gl_LaunchSizeEXT.xy), vec2(gl_LaunchIDEXT.xy), radians(45.0)), 0.0);
	direction.z *= -1;
	//vec4 direction = vec4(normalize(target.xyz / target.w), 0) ;

	uint rayFlags = gl_RayFlagsOpaqueEXT;
	uint cullMask = 0xff;
	float tmin = 0.001;
	float tmax = 10000.0;

	traceRayEXT(topLevelAS, rayFlags, cullMask, 0, 0, 0, origin.xyz, tmin, direction.xyz, tmax, 0);

	imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(hitValue, 0.0));
}