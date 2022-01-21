#include "Mesh.h"

using namespace GPUSpectral;

GPUSpectral::Mesh::Mesh(HwDriver& driver, uint32_t id, const std::span<Vertex>& vertices, const std::span<uint32_t>& indices)
    : driver(driver), vertices(vertices.begin(), vertices.end()), indices(indices.begin(), indices.end()), id(id) {
    std::vector<float> v;
    std::vector<float> vn;
    std::vector<float> vt;
    v.reserve(vertices.size() * 3);
    vn.reserve(vertices.size() * 3);
    vt.reserve(vertices.size() * 2);
    for (auto& vert : vertices) {
        v.push_back(vert.pos.x);
        v.push_back(vert.pos.y);
        v.push_back(vert.pos.z);
        vn.push_back(vert.normal.x);
        vn.push_back(vert.normal.y);
        vn.push_back(vert.normal.z);
        vt.push_back(vert.uv.x);
        vt.push_back(vert.uv.y);
    }

    attributes[0] = {
        .name = "position",
        .index = 0,
        .offset = 0,
        .stride = 12,
        .type = ElementType::FLOAT3
    };

    attributes[1] = {
        .name = "normal", .index = 1, .offset = 0, .stride = 12, .type = ElementType::FLOAT3
    };

    attributes[2] = {
        .name = "texcoord",
        .index = 2,
        .offset = 0,
        .stride = 8,
        .type = ElementType::FLOAT2,
    };

    positionBuffer = driver.createBufferObject(4 * v.size(), BufferUsage::VERTEX | BufferUsage::STORAGE | BufferUsage::BDA | BufferUsage::ACCELERATION_STRUCTURE_INPUT, BufferType::DEVICE);
    driver.updateBufferObjectSync(positionBuffer, { .data = (uint32_t*)v.data() }, 0);

    normalBuffer = driver.createBufferObject(4 * v.size(), BufferUsage::VERTEX | BufferUsage::STORAGE | BufferUsage::BDA | BufferUsage::ACCELERATION_STRUCTURE_INPUT, BufferType::DEVICE);
    driver.updateBufferObjectSync(normalBuffer, { .data = (uint32_t*)vn.data() }, 0);

    uvBuffer = driver.createBufferObject(4 * vt.size(), BufferUsage::VERTEX | BufferUsage::STORAGE | BufferUsage::BDA | BufferUsage::ACCELERATION_STRUCTURE_INPUT, BufferType::DEVICE);
    driver.updateBufferObjectSync(uvBuffer, { .data = (uint32_t*)vt.data() }, 0);

    vertexBuffer = driver.createVertexBuffer(3, v.size() / 3, 3, attributes);
    driver.setVertexBuffer(vertexBuffer, 0, positionBuffer);
    driver.setVertexBuffer(vertexBuffer, 1, normalBuffer);
    driver.setVertexBuffer(vertexBuffer, 2, uvBuffer);

    indexBuffer = driver.createIndexBuffer(indices.size());
    driver.updateIndexBuffer(indexBuffer, { .data = (uint32_t*)indices.data() }, 0);

    primitive = driver.createPrimitive(PrimitiveMode::TRIANGLES);
    driver.setPrimitiveBuffer(primitive, vertexBuffer, indexBuffer);
}

GPUSpectral::Mesh::~Mesh() {
    driver.destroyPrimitive(primitive);
    driver.destroyVertexBuffer(vertexBuffer);
    driver.destroyBufferObject(positionBuffer);
    driver.destroyBufferObject(normalBuffer);
    driver.destroyBufferObject(uvBuffer);
}

Handle<HwVertexBuffer> GPUSpectral::Mesh::getVertexBuffer() const noexcept {
    return vertexBuffer;
}

Handle<HwPrimitive> GPUSpectral::Mesh::getPrimitive() const noexcept {
    return primitive;
}

Handle<HwIndexBuffer> GPUSpectral::Mesh::getIndexBuffer() const noexcept {
    return indexBuffer;
}

Handle<HwBufferObject> GPUSpectral::Mesh::getPositionBuffer() const noexcept {
    return positionBuffer;
}

Handle<HwBufferObject> GPUSpectral::Mesh::getNormalBuffer() const noexcept {
    return normalBuffer;
}

Handle<HwBufferObject> GPUSpectral::Mesh::getUVBuffer() const noexcept {
    return uvBuffer;
}

const AttributeArray& GPUSpectral::Mesh::getAttributes() const noexcept {
    return attributes;
}

uint32_t GPUSpectral::Mesh::getAttributeCount() const noexcept {
    return 3;
}

uint32_t GPUSpectral::Mesh::getID() const noexcept {
    return id;
}

std::span<const Mesh::Vertex> GPUSpectral::Mesh::getVertices() const noexcept {
    return { vertices.data(), vertices.size() };
}
