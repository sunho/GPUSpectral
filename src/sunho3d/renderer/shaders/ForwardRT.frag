#version 450
#define MAX_MESH_COUNT 32
#define MAX_INSTANCES 32

#extension GL_EXT_nonuniform_qualifier : require

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
    int meshIndex;
    Material material;
};

layout(std140, binding = 0) uniform SceneBuffer {
    uvec2 frameSize;
    uint instanceNum;
    Instance instances[MAX_INSTANCES];
} sceneBuffer;

layout(std430,set=1,binding = 0) buffer VertexPositionBuffer {
    float position[];
} vertexPositionBuffer[MAX_MESH_COUNT];

layout(std430,set=1,binding = 1) buffer VertexNormalBuffer {
    float normal[];
} vertexNormalBuffer[MAX_MESH_COUNT];

layout(std430,set=1,binding = 2) buffer VertexUVBuffer {
    float uv[];
} vertexUVBuffer[MAX_MESH_COUNT];

layout(std140,set=1,binding = 3) readonly buffer RayHitBuffer {
    RayHit hits[];
} rayHitBuffer;


Vertex loadVertex(int meshIndex, uint faceIndex) {
    Vertex v;
    v.pos = vec3(vertexPositionBuffer[nonuniformEXT(meshIndex)].position[faceIndex*3],
                      vertexPositionBuffer[nonuniformEXT(meshIndex)].position[faceIndex*3 + 1],
                      vertexPositionBuffer[nonuniformEXT(meshIndex)].position[faceIndex*3 + 2]);
    v.normal = vec3(vertexNormalBuffer[nonuniformEXT(meshIndex)].normal[faceIndex*3],
                          vertexNormalBuffer[nonuniformEXT(meshIndex)].normal[faceIndex*3 + 1],
                          vertexNormalBuffer[nonuniformEXT(meshIndex)].normal[faceIndex*3 + 2]);
    v.uv = vec2(vertexUVBuffer[nonuniformEXT(meshIndex)].uv[faceIndex*2],
                      vertexUVBuffer[nonuniformEXT(meshIndex)].uv[faceIndex*2 + 1]);
    return v;
}

layout(location = 0) out vec4 outColor;

void main() {
    uvec2 xy = uvec2(gl_FragCoord);
    uint hitIndex = xy.x + xy.y * sceneBuffer.frameSize.x;
    RayHit hit = rayHitBuffer.hits[hitIndex];
    if (hit.instId == 0xFFFFFFFF) {
        outColor = vec4(1.0,0.0,0.0,1.0);
    } else {
        Instance instance = sceneBuffer.instances[hit.instId];
        Vertex v0 = loadVertex(instance.meshIndex, hit.primId * 3);
        Vertex v1 = loadVertex(instance.meshIndex, hit.primId * 3 + 1);
        Vertex v2 = loadVertex(instance.meshIndex, hit.primId * 3 + 2);
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
        outColor = vec4(pos, 1.0);
    }
}
