#pragma once

#include "Camera.h"
#include "Mesh.h"
#include "backend/Handles.h"
#include <vector>
#include <optional>
#include <unordered_map>

namespace VKGIRenderer {
class Engine;

using TextureId = size_t;
using MaterialHandle = int;

struct RenderObject {
    glm::mat4 transform;
    Mesh* mesh;
    MaterialHandle material;
};

enum BSDFType : uint16_t {
#define BSDFDefinition(BSDFNAME, BSDFFIELD, BSDFTYPE) BSDF_##BSDFTYPE,
#include "BSDF.inc"
#undef BSDFDefinition
};


struct DiffuseBSDF {
    glm::vec3 reflectance;
    Handle<HwTexture> reflectanceTex;
};

struct SmoothDielectricBSDF {
    float iorIn;
    float iorOut; // usually 1.0
};

struct SmoothConductorBSDF {
    float iorIn;
    float iorOut; // usually 1.0
};

struct SmoothFloorBSDF {
    glm::vec3 diffuse;
    float R0;
};

struct SmoothPlasticBSDF {
    glm::vec3 diffuse;
    float iorIn;
    float iorOut; // usually 1.0
    float R0;
};

enum MicrofacetType : uint32_t {
    BECKMANN,
    GGX
};

struct RoughConductorBSDF {
    glm::vec3 eta;
    glm::vec3 k;
    glm::vec3 reflectance;
    float alpha;
    MicrofacetType distribution;
};

struct RoughPlasticBSDF {
    glm::vec3 diffuse;
    Handle<HwTexture> diffuseTex;
    float iorIn;
    float iorOut; // usually 1.0
    float R0;
    float alpha;
    MicrofacetType distribution;
};

struct RoughFloorBSDF {
    glm::vec3 diffuse;
    float R0;
    float alpha;
    MicrofacetType distribution;
};

struct BSDFHandle {
    BSDFHandle() { }
    BSDFHandle(BSDFType type, uint32_t index) : handle((static_cast<uint32_t>(type) << 16) | (index & 0xFFFF)) {
    }
    BSDFType type() const {
        return static_cast<BSDFType>((handle >> 16) & 0xffff);
    }
    uint32_t index() const {
        return handle & 0xFFFF;
    }
    uint32_t handle;
};

struct Material {
    glm::vec3 emission{ 0.0f,0.0f,0.0f };
    bool twofaced = false;
    bool facenormals = false;
    BSDFHandle bsdf;
};

struct TriangleLight {
    glm::vec3 positions[3];
    glm::vec3 radiance;
};

struct BoundingBox {
    glm::vec3 mins;
    glm::vec3 maxs;
};

struct Envmap {
    Handle<HwTexture> texture;
    glm::mat4 transform;
};

struct SceneData {
    std::vector<glm::vec4> positions;
    std::vector<glm::vec4> normals;
    std::vector<glm::vec2> uvs;
    std::vector<int> matIndices;

    BoundingBox getBoundingBox() const {
        BoundingBox outBox = {
            .mins = glm::vec3(INFINITY, INFINITY, INFINITY),
            .maxs = glm::vec3(-INFINITY, -INFINITY, -INFINITY),
        };
        for (auto pos : positions) {
            outBox.mins = glm::min(outBox.mins, glm::vec3(pos));
            outBox.maxs = glm::max(outBox.maxs, glm::vec3(pos));
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

    void addRenderObject(const RenderObject& object) {
        renderObjects.push_back(object);
    }

    void addTriangleLight(const TriangleLight& light) {
        triangleLights.push_back(light);
    }

#define BSDFDefinition(BSDFNAME, BSDFFIELD, BSDFTYPE) \
    BSDFHandle add##BSDFNAME(const BSDFNAME& bsdf) { \
        BSDFHandle outHandle { BSDF_##BSDFTYPE, (uint32_t)BSDFFIELD##s.size() }; \
        BSDFFIELD##s.push_back(bsdf); \
        return outHandle; \
    }
#include "BSDF.inc"
#undef BSDFDefinition

    void prepare(Engine& engine);

    Camera camera;
    std::vector<RenderObject> renderObjects;
    std::vector<Material> materials;
    std::vector<TriangleLight> triangleLights;
#define BSDFDefinition(BSDFNAME, BSDFFIELD, BSDFTYPE) std::vector<BSDFNAME> BSDFFIELD##s;
#include "BSDF.inc"
#undef BSDFDefinition
    std::optional<Envmap> envMap;

    // baked data
    SceneData sceneData;
};
}