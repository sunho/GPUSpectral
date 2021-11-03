#version 450

#define MAX_INSTANCES 32

struct Vertex {
    vec3 pos;
    vec3 normal;
    vec2 uv;
};

struct RayHit {
    vec2 bary;
    uint instId;
    uint primId;
};

struct Instance {
    mat4 transform;
    int vertexStart;
};

layout(std140, binding = 0) uniform SceneBuffer {
    uvec2 frameSize;
    uint instanceNum;
    Instance instances[MAX_INSTANCES];
} sceneBuffer;

layout(std140,set=3,binding = 0) readonly buffer VertexBuffer {
    Vertex vertices[];
} vertexBuffer;

layout(std140,set=3,binding = 1) readonly buffer RayHitBuffer {
    RayHit hits[];
} rayHitBuffer;

layout(location = 0) out vec4 outColor;

void main() {
    uvec2 xy = uvec2(gl_FragCoord);
    uint hitIndex = xy.x + xy.y * sceneBuffer.frameSize.x;
    RayHit hit = rayHitBuffer.hits[hitIndex];
    if (hit.instId == 0xFFFFFFFF) {
        outColor = vec4(1.0,0.0,0.0,1.0);
    } else {
        Instance instance = sceneBuffer.instances[hit.instId];
        Vertex v0 = vertexBuffer.vertices[instance.vertexStart + hit.primId * 3];
        Vertex v1 = vertexBuffer.vertices[instance.vertexStart + hit.primId * 3 + 1];
        Vertex v2 = vertexBuffer.vertices[instance.vertexStart + hit.primId * 3 + 2];
        vec3 n0 = vec3(instance.transform * vec4(v0.normal, 0.0));
        vec3 n1 = vec3(instance.transform * vec4(v1.normal, 0.0));
        vec3 n2 = vec3(instance.transform * vec4(v2.normal, 0.0));
        vec3 bary = vec3(hit.bary,1.0-(hit.bary.x+hit.bary.y));
        vec3 nor = normalize(bary.x * n1 + bary.y * n2 + bary.z * n0);
        outColor = vec4(nor, 1.0);
    }
}
