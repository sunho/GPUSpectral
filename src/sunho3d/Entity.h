#pragma once

#include <string>
#include <vector>
#include "ResourceList.h"
#include "Material.h"
#include <sunho3d/backend/DriverTypes.h>

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
    explicit Entity(uint32_t id);
    void addNode(Entity* entity);
    void addPrimitive(Primitive&& primitive);
    const std::vector<Entity*>& getNodes();
    
//private:
    std::vector<Entity*> nodes;
    std::vector<Primitive> primitives;
};

}
