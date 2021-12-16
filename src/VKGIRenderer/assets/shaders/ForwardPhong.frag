#version 450
#pragma shader_stage(fragment)

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
    mat4 lightVP[MAX_LIGHTS];
    int numLights;
} light;

layout(binding = 2) uniform MaterialBuffer {
    vec3 specular;
    float phong;
} material;

layout(set=1,binding = 0) uniform sampler2D diffuseMap;
layout(set=1,binding = 1) uniform sampler2D shadowMap;

#define PCF_KERNEL_SIZE 6
#define BLOCK_SEARCH_SIZE 4
#define LIGHT_AREA 0.005f

float pcf(vec2 uv, float z, float radius) {
    float step = 2*radius / PCF_KERNEL_SIZE;
    float shadow = 0.0f;
    for (float i = -radius; i < radius; i += step) {
        for (float j = -radius; j < radius; j += step) {
            vec2 uv2 = vec2(uv.x + i, uv.y + j);
            float tex = texture(shadowMap, uv2).x;
            shadow += float(z < tex);
        }
    }
    return shadow / (PCF_KERNEL_SIZE*PCF_KERNEL_SIZE);
}

float blockerSearch(vec2 uv, float z, float radius, out float numBlockers) {
    float step = 2*radius / BLOCK_SEARCH_SIZE;
    float distance = 0.0;
    for (float i = -radius; i < radius; i += step) {
       for (float j = -radius; j < radius; j += step) {
           vec2 uv2 = vec2(uv.x + i, uv.y + j);
           float tex = texture(shadowMap, uv2).x;
           if (tex < z) {
               distance += tex;
               ++numBlockers;
           }
       }
    }
    if (distance == 0.0) {
        return 0.0;
    }
    return distance / numBlockers;
}

float pcss(vec2 uv, float z, float radius) {
    // it's not blocked
    /*if (z < texture(shadowMap, uv).x) {
        return 1.0;
    }*/
    
    float numBlockers;
    float dis = blockerSearch(uv, z, radius, numBlockers);
    if (numBlockers >= BLOCK_SEARCH_SIZE*BLOCK_SEARCH_SIZE) {
        return 0.0;
    }
    if (numBlockers == 0.0) {
        return 1.0;
    }
    
    // d_recv is always more than d_blocker
    // w_penumbra = (d_recv - d_blocker)*w_light/d_blocker
    float penum = (z - dis)*LIGHT_AREA/dis;
    
    return pcf(uv, z, penum);
}

void main() {
    vec3 normal = normalize(inNormal);
    vec3 v = normalize(transform.cameraPos - inPos);
    vec4 diffuse = texture(diffuseMap, inUV);
    vec4 color = diffuse * 0.35;
    float bias = 0.005;
    float textureSize = float(textureSize(shadowMap,0).x);
    float radius = 8 / textureSize;
    
    for (int i = 0; i < light.numLights; i++) {
        vec4 tmp = (light.lightVP[i] * vec4(inPos,1.0));
        tmp /= tmp.w;
        tmp.z-=bias;
        float shadow = pcf(tmp.xy / 2.0f + 0.5f, tmp.z, radius);
        vec3 lightV = light.light[i].pos - inPos;
        vec3 light = normalize(lightV);
        vec3 h = normalize(light + v);
        float dis = length(lightV);
        float lightI = 1.0/(dis*dis);
        //color = smp;
        color += shadow*lightI* diffuse * max(dot(light,normal),0.0f);
        color += shadow*vec4(material.specular,1.0f) * lightI * pow(max(0.0f, dot(normal, h)),material.phong);
    }
    outColor = color;
}
