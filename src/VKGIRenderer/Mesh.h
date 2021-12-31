#pragma once

#include <glm/glm.hpp>
#include "backend/DriverTypes.h"
#include <vector>
#include "backend/Handles.h"
#include "utils/ResourceList.h"

namespace VKGIRenderer {
struct Vertex {
    glm::vec3 pos;
    float _pad0;
    glm::vec3 normal;
    float _pad1;
    glm::vec2 uv;
    glm::vec2 _pad2;
};

struct Mesh : public IdResource {
    Handle<HwVertexBuffer> vertexBuffer;
    Handle<HwPrimitive> hwInstance;
    Handle<HwIndexBuffer> indexBuffer;
    Handle<HwBufferObject> positionBuffer;
    Handle<HwBufferObject> normalBuffer;
    Handle<HwBufferObject> uvBuffer;
    AttributeArray attributes;
    std::vector<Vertex> vertices;
    uint32_t elementCount;
    uint32_t attributeCount;
};
}
