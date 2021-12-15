#pragma once

#include "../math/VectorMath.cuh"
#include <vector>
#include <unordered_map>

struct Camera {
    float3 eye;
    float3 u;
    float3 v;
    float3 w;
};

struct RenderObject {
    mat4 transform;
    int meshId;
    int materialId;
};

struct Material {
    float3 color = { 0.0f, 0.0f, 0.0f };
    float3 emission = { 0.0f,0.0f,0.0f };
};

struct Scene {
    Camera camera;
    std::vector<RenderObject> renderObjects;
    std::vector<Material> materials;
};