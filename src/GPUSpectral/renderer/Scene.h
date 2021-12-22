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
    
    Camera camera;
    std::vector<RenderObject> renderObjects;
    std::vector<Material> materials;
    std::vector<TriangleLight> triangleLights;
    #define BSDFDefinition(BSDFNAME, BSDFFIELD, BSDFTYPE) std::vector<BSDFNAME> BSDFFIELD##s;
    #include "../kernels/BSDF.inc"
    #undef BSDFDefinition
};