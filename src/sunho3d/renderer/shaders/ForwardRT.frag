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

#define MATERIAL_DIFFUSE_TEXTURE 1
#define MATERIAL_DIFFUSE_COLOR 2
#define MATERIAL_EMISSION 3

struct Material {
    vec3 diffuseColor;
    int diffuseMapIndex;
    int typeID;
};

struct Instance {
    mat4 transform;
    int vertexStart;
    Material material;
};

layout(std140, binding = 0) uniform SceneBuffer {
    uvec2 frameSize;
    uint instanceNum;
    Instance instances[MAX_INSTANCES];
} sceneBuffer;

layout(std140,set=2,binding = 0) readonly buffer VertexBuffer {
    Vertex vertices[];
} vertexBuffer;

layout(std140,set=2,binding = 1) readonly buffer RayHitBuffer {
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
        mat4 invModelT = transpose(inverse(instance.transform));
        vec3 n0 = vec3(invModelT * vec4(v0.normal, 0.0));
        vec3 n1 = vec3(invModelT * vec4(v1.normal, 0.0));
        vec3 n2 = vec3(invModelT  * vec4(v2.normal, 0.0));
        vec3 p0 = vec3(instance.transform * vec4(v0.pos, 1.0));
        vec3 p1 = vec3(instance.transform * vec4(v1.pos, 1.0));
        vec3 p2 = vec3(instance.transform * vec4(v2.pos, 1.0));
        vec3 bary = vec3(hit.bary,1.0-(hit.bary.x+hit.bary.y));
        vec3 nor = normalize(bary.x * n1 + bary.y * n2 + bary.z * n0);
        // 0 1 2
        // 0 2 1
        // 1 0 2
        // 1 2 0
        // 2 0 1
        // 2 1 0
        vec3 pos = bary.x * p1 + bary.y * p2 + bary.z * p0;
        //nor.z *= -1.0;
        vec2 uv = bary.x * v1.uv + bary.y * v2.uv + bary.z * v0.uv;
        outColor = vec4(nor, 1.0);
    }
}
