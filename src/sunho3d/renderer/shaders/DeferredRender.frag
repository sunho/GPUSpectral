#version 450

#define MAX_LIGHTS 64

layout(location = 0) in vec2 inPos;
layout(location = 0) out vec4 outColor;

struct Light {
    vec3 pos;
    vec3 dir;
    vec2 RI;
};

layout(binding = 0) uniform LightUniformBuffer {
    Light light[MAX_LIGHTS];
    mat4 lightVP[MAX_LIGHTS];
    int numLights;
} light;

layout(set=1,binding = 0) uniform sampler2D positionBuffer;
layout(set=1,binding = 1) uniform sampler2D normalBuffer;
layout(set=1,binding = 2) uniform sampler2D diffuseBuffer;

layout( push_constant ) uniform PushConstants {
    vec3 cameraPos;
} constants;

void main() {
    vec2 uv = inPos / 2.0 + 0.5;

    vec3 normal = vec3(texture(positionBuffer, uv));
    vec3 pos = vec3(texture(normalBuffer, uv));
    vec3 v = normalize(constants.cameraPos - pos);
    vec4 diffuse = texture(diffuseBuffer, uv);
    vec4 color = diffuse * 0.35;
    
    for (int i = 0; i < light.numLights; i++) {
        vec3 lightV = light.light[i].pos - pos;
        vec3 light = normalize(lightV);
        vec3 h = normalize(light + v);
        float dis = length(lightV);
        float lightI = 1.0/(dis*dis);
        //color = smp;
        color += lightI* diffuse * max(dot(light,normal),0.0f);
    }
    outColor = color;
}
