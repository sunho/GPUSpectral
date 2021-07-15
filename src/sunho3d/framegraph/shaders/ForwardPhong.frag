#version 450

#define MAX_LIGHTS 64

layout(location = 0) in vec2 inUV;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inPos;

layout(location = 0) out vec4 outColor;

struct Light {
    vec3 pos;
    vec3 dir;
    vec2 RI;
};

layout(binding = 0) uniform TransformUniformBuffer {
    mat4 MVP;
    mat4 model;
    mat4 invModelT;
    vec3 cameraPos;
} transform;

layout(binding = 1) uniform LightUniformBuffer {
    Light light[MAX_LIGHTS];
    int numLights;
} light;

layout(binding = 2) uniform MaterialBuffer {
    vec3 specular;
    float phong;
} material;

layout(set=1,binding = 0) uniform sampler2D diffuseMap;

void main() {
    vec3 normal = normalize(inNormal);
    vec3 v = normalize(transform.cameraPos - inPos);
    vec4 diffuse = texture(diffuseMap, inUV);
    vec4 color = diffuse * 0.35;
    for (int i = 0; i < light.numLights; i++) {
        vec3 lightV = light.light[i].pos - inPos;
        vec3 light = normalize(lightV);
        vec3 h = normalize(light + v);
        float dis = length(lightV);
        float lightI = 1.0/(dis*dis);
        color += lightI* diffuse * max(dot(light,normal),0.0f);
        color += vec4(material.specular,1.0f) * lightI * pow(max(0.0f, dot(normal, h)),material.phong);
    }
    outColor = color;
}
