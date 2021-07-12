#version 450

layout(location = 2) in vec3 inPosition;
layout(location = 3) in vec3 inNormal;
layout(location = 0) out vec3 fragColor;

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
    
    vec4 k = ubo.proj*ubo.view*vec4(inPosition, 1.0);
    gl_Position = vec4(k.x, k.y, 0, 1);
    fragColor = inNormal;
}
