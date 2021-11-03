#pragma once

#include <sunho3d/backend/DriverBase.h>
#include <sunho3d/backend/DriverTypes.h>
#include <sunho3d/backend/Handles.h>
#include <list>

#include <string>
#include <vector>

#include "Material.h"
#include "Transform.h"
#include "utils/ResourceList.h"

namespace sunho3d {

struct Vertex {
    glm::vec3 pos;
    float _pad0;
    glm::vec3 normal;
    float _pad1;
    glm::vec2 uv;
    glm::vec2 _pad2;
};

struct Primitive {
    Handle<HwIndexBuffer> indexBuffer;
    Handle<HwVertexBuffer> vertexBuffer;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    AttributeArray attibutes;
    uint32_t elementCount;
    uint32_t attributeCount;
    Material *material;
    PrimitiveMode mode;
    Handle<HwPrimitive> hwInstance;
};

class Entity : public IdResource {
  public:
    explicit Entity();

    void addNode(Entity *entity);
    void addPrimitive(const Primitive &primitive);

    const std::vector<Entity *> &getNodes() {
        return nodes;
    }
    const std::list<Primitive> &getPrimitives() {
        return primitives;
    }

    const Transform &getTransform() const {
        return transform;
    }
    void setTransform(const Transform &transform) {
        this->transform = transform;
    }

  private:
    Transform transform;
    std::vector<Entity *> nodes;
    std::list<Primitive> primitives;
};

}  // namespace sunho3d
