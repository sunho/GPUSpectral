#pragma once

#include "../kernels/VectorMath.cuh"
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

    BSDFHandle addDiffuseBSDF(const DiffuseBSDF& bsdf) {
        BSDFHandle outHandle { BSDF_DIFFUSE, (uint32_t)diffuseBSDFs.size() };
        diffuseBSDFs.push_back(bsdf);
        return outHandle;
    }

    BSDFHandle addSmoothDielectricBSDF(const SmoothDielectricBSDF& bsdf) {
        BSDFHandle outHandle { BSDF_SMOOTH_DIELECTRIC, (uint32_t)smoothDielectricBSDFs.size() };
        smoothDielectricBSDFs.push_back(bsdf);
        return outHandle;
    }

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
    std::vector<DiffuseBSDF> diffuseBSDFs;
    std::vector<SmoothDielectricBSDF> smoothDielectricBSDFs;
};