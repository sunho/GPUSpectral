#version 450
#pragma shader_stage(vertex)

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTex;

layout(location = 0) out vec2 uv;
layout(location = 1) out vec3 normal;
layout(location = 2) out vec3 pos;

layout(binding = 0) uniform TransformUniformBuffer {
    mat4 MVP;
    mat4 model;
    mat4 invModelT;
    vec3 cameraPos;
} transform;

void main() {
    gl_Position = transform.MVP*vec4(inPosition, 1.0);
    uv = inTex;
    normal = vec3(transform.invModelT*vec4(inNormal,0.0));
    pos = vec3(transform.model*vec4(inPosition,1.0));
}
