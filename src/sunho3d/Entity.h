#pragma once

#include <sunho3d/backend/DriverTypes.h>

#include <string>
#include <vector>

#include "Material.h"
#include "utils/ResourceList.h"

namespace sunho3d {
struct Primitive {
    std::vector<uint16_t> indexBuffer;
    std::vector<std::vector<char>> vertexBuffers;
    AttributeArray attibutes;
    uint32_t elementCount;
    uint32_t attributeCount;
    Material material;
    PrimitiveMode mode;
};

class Entity : public IdResource {
  public:
    explicit Entity();
    void addNode(Entity *entity);
    void addPrimitive(Primitive &&primitive);
    const std::vector<Entity *> &getNodes();

    // private:
    std::vector<Entity *> nodes;
    std::vector<Primitive> primitives;
};

}  // namespace sunho3d
