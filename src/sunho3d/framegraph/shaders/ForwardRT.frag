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
    mat4x3 transform;
    uint vertexStart;
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
        outColor = vec4(hit.bary,1.0-(hit.bary.x+hit.bary.y), 1.0);
    }
}
