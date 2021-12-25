#pragma once

#include "../kernels/VectorMath.cuh"
#include "../kernels/BSDFSampler.cuh"
#include "../kernels/LightSampler.cuh"
#include <vector>
#include <unordered_map>

struct Camera {
    float3 eye;
    float3 u;
    float3 v;
    float3 w;
    float fov;
};
class Renderer;

using TextureId = size_t;
using MaterialHandle = int;

struct RenderObject {
    mat4 transform;
    int meshId;
    MaterialHandle material;
};

struct Material {
    float3 emission = { 0.0f,0.0f,0.0f };
    bool twofaced = false;
    bool facenormals = false;
    BSDFHandle bsdf;
};

struct BoundingBox {
    float3 mins;
    float3 maxs;
};

struct SceneData {
    std::vector<float4> positions;
    std::vector<float4> normals;
    std::vector<float2> uvs;
    std::vector<int> matIndices;

    BoundingBox getBoundingBox() const {
        BoundingBox outBox = {
            .mins = make_float3(INFINITY, INFINITY, INFINITY),
            .maxs = make_float3(-INFINITY, -INFINITY, -INFINITY),
        };
        for (auto pos : positions) {
            outBox.mins = fminf(outBox.mins, make_float3(pos));
            outBox.maxs = fmaxf(outBox.maxs, make_float3(pos));
        }
        return outBox;
    }
};

struct Scene {
    Scene() {
    }
    
    MaterialHandle addMaterial(const Material& material) {
        MaterialHandle outHandle = materials.size();
        materials.push_back(material);
        return outHandle;
    }

    Material& getMaterial(const MaterialHandle& handle) {
        return materials[handle];
    }

    const Material& getMaterial(const MaterialHandle& handle) const {
        return materials[handle];
    }

    #define BSDFDefinition(BSDFNAME, BSDFFIELD, BSDFTYPE) \
    BSDFHandle add##BSDFNAME(const BSDFNAME& bsdf) { \
        BSDFHandle outHandle { BSDF_##BSDFTYPE, (uint32_t)BSDFFIELD##s.size() }; \
        BSDFFIELD##s.push_back(bsdf); \
        return outHandle; \
    }
    #include "../kernels/BSDF.inc"
    #undef BSDFDefinition

    void addRenderObject(const RenderObject& object) {
        renderObjects.push_back(object);
    }

    void addTriangleLight(const TriangleLight& light) {
        triangleLights.push_back(light);
    }

    void prepare(Renderer& renderer);
    
    Camera camera;
    std::vector<RenderObject> renderObjects;
    std::vector<Material> materials;
    std::vector<TriangleLight> triangleLights;
    TextureId envMap{ 0 };
    #define BSDFDefinition(BSDFNAME, BSDFFIELD, BSDFTYPE) std::vector<BSDFNAME> BSDFFIELD##s;
    #include "../kernels/BSDF.inc"
    #undef BSDFDefinition

    // baked data
    SceneData sceneData;
};