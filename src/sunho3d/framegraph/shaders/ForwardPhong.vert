#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTex;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 uv;
layout(location = 2) out vec3 normal;
layout(location = 3) out vec3 pos;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

vec2 positions[3] = vec2[](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);


void main() {
    mat4 M = ubo.proj*ubo.view*ubo.model;
    vec4 k = M*vec4(inPosition, 1.0);
    gl_Position = k;
    fragColor = inNormal;
    uv = inTex;
    pos = vec3(ubo.model*vec4(inPosition,1.0));
    normal = vec3(transpose(inverse(ubo.model))*vec4(inNormal, 0.0));
}
