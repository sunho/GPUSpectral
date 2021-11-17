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
    Handle<HwBufferObject> positionBuffer;
    Handle<HwBufferObject> normalBuffer;
    Handle<HwBufferObject> uvBuffer;
    AttributeArray attibutes;
    uint32_t elementCount;
    uint32_t attributeCount;
    PrimitiveMode mode;
    Handle<HwPrimitive> hwInstance;
};

class Mesh : public IdResource {
  public:
    Mesh();
    void addPrimitive(const Primitive &primitive);
    const std::list<Primitive> &getPrimitives() {
        return primitives;
    }

  private:
    std::list<Primitive> primitives;
};

class Entity : public IdResource {
  public:
    explicit Entity();

    void addNode(Entity *entity);

    const std::vector<Entity *> &getNodes() {
        return nodes;
    }

    const glm::mat4 &getTransform() const {
        return transform;
    }

    const glm::mat4 &getTransformInvT() const {
        return transformInvT;
    }

    Material *getMaterial() const {
        return material;
    }

    Mesh *getMesh() const {
        return mesh;
    }
   
    void setMesh(Mesh *mesh) {
        this->mesh = mesh;
    }

    void setMaterial(Material *material) {
        this->material = material;
    }

    void setTransform(const Transform &transform) {
        setTransformMatrix(transform.toMatrix());
    }

    void setTransformMatrix(const glm::mat4 &mat) {
        transform = mat;
        transformInvT = glm::inverse(glm::transpose(mat));
    }

  private:
    glm::mat4 transform;
    glm::mat4 transformInvT;
    std::vector<Entity *> nodes;
    Material *material{};
    Mesh *mesh{};
};

}  // namespace sunho3d
