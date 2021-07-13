#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec3 normal;
layout(location = 0) out vec4 outColor;

vec3 light = normalize(vec3(0.4,0.5,-1.0));

layout(binding = 0) uniform sampler2D texSampler;
void main() {
    vec4 diffuse = texture(texSampler, uv);
    outColor = diffuse * 0.2 + diffuse * dot(light,normal);
}
