#pragma once

#include "utils/ResourceList.h"

namespace VKGIRenderer {

struct DiffuseTextureMaterialData {
    Handle<HwTexture> diffuseMap;
};

struct DiffuseColorMaterialData {
    glm::vec3 rgb;
};


struct EmissionMaterialData {
    glm::vec3 radiance;
};


using MaterialData = std::variant<DiffuseTextureMaterialData, DiffuseColorMaterialData, EmissionMaterialData, std::monostate>;

struct Material : public IdResource {
    bool twosided{false};
    MaterialData materialData{};
};


}  // namespace VKGIRenderer
