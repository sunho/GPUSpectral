#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <span>
#include "../backend/DriverTypes.h"
#include "../backend/vulkan/VulkanDriver.h"
#include "../backend/Handles.h"

namespace GPUSpectral {

class Mesh {
public:
    struct Vertex {
        alignas(16) glm::vec3 pos;
        alignas(16) glm::vec3 normal;
        glm::vec2 uv;
    };

    Mesh(HwDriver& driver, uint32_t id, const std::span<Vertex>& vertices, const std::span<uint32_t>& indices);
    ~Mesh();

    Mesh(Mesh&&) noexcept = default;
    Mesh& operator=(Mesh&&) noexcept = default;

    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;
    
    [[nodiscard]] Handle<HwVertexBuffer> getVertexBuffer() const noexcept;

    [[nodiscard]] Handle<HwPrimitive> getPrimitive() const noexcept;

    [[nodiscard]] Handle<HwIndexBuffer> getIndexBuffer() const noexcept;

    [[nodiscard]] Handle<HwBufferObject> getPositionBuffer() const noexcept;

    [[nodiscard]] Handle<HwBufferObject> getNormalBuffer() const noexcept;

    [[nodiscard]] Handle<HwBufferObject> getUVBuffer() const noexcept;

    [[nodiscard]] const AttributeArray& getAttributes() const noexcept;

    [[nodiscard]] uint32_t getAttributeCount() const noexcept;

    [[nodiscard]] uint32_t getID() const noexcept;

    [[nodiscard]] std::span<const Vertex> getVertices() const noexcept;
    
private:
    uint32_t id;
    HwDriver& driver;
    Handle<HwVertexBuffer> vertexBuffer{};
    Handle<HwPrimitive> primitive{};
    Handle<HwIndexBuffer> indexBuffer{};
    Handle<HwBufferObject> positionBuffer{};
    Handle<HwBufferObject> normalBuffer{};
    Handle<HwBufferObject> uvBuffer{};

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    // only single attributes format is supported
    AttributeArray attributes;
};
}
