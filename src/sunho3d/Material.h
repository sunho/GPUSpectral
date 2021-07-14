#pragma once

#include "utils/ResourceList.h"

namespace sunho3d {
struct Material : public IdResource {
    Handle<HwTexture> diffuseMap;
};
}  // namespace sunho3d
