#pragma once

#include "../math/VectorMath.h"
#include <vector>
#include <unordered_map>

struct Camera {
    float3 eye;
    float3 look;
    float3 up;
    float fov;
};

struct RenderObject {
    mat4 transform;
    int meshId;
    int materialId;
};

struct Material {
    float3 color;
    float3 emission;
};

struct Scene {
    Camera camera;
    std::vector<RenderObject> renderObjects;
    std::vector<Material> materials;
};