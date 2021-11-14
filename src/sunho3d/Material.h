#pragma once

#include "utils/ResourceList.h"

namespace sunho3d {

struct DiffuseTextureMaterialData {
    Handle<HwTexture> diffuseMap;
};

struct DiffuseColorMaterialData {
    glm::vec3 rgb;
};

using MaterialData = std::variant<DiffuseTextureMaterialData, DiffuseColorMaterialData, std::monostate>;

struct Material : public IdResource {
    bool twosided{false};
    MaterialData materialData{};
};


}  // namespace sunho3d
