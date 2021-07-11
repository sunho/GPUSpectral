#pragma once

#include "DriverTypes.h"
#include "Program.h"

struct HwBase {
};

struct HwVertexBuffer : public HwBase {
    AttributeArray attributes{};
    uint32_t vertexCount{};
    uint8_t attributeCount{};
    HwVertexBuffer() = default;
    explicit HwVertexBuffer(uint32_t vertexCount, uint8_t attributeCount, const AttributeArray& attributes) :
        vertexCount(vertexCount), attributeCount(attributeCount), attributes(attributes) { }
};

struct HwIndexBuffer : public HwBase {
    uint32_t count{};
    HwIndexBuffer() = default;
    explicit HwIndexBuffer(uint32_t count) : count(count) { }
};

struct HwProgram : public HwBase {
    Program program;
    HwProgram() = default;
    explicit HwProgram(const Program& program) : program(program) { }
};

struct HwRenderTarget : public HwBase {
    uint32_t width{};
    uint32_t height{};
    HwRenderTarget() = default;
    explicit HwRenderTarget(uint32_t w, uint32_t h) : width(w), height(h) { }
};

struct HwBufferObject {
    uint32_t size{};
    HwBufferObject() = default;
    explicit HwBufferObject(uint32_t size) : size(size) { }
};

struct HwPrimitive {
//    uint32_t offset{};
//    uint32_t minIndex{};
//    uint32_t maxIndex{};
//    uint32_t count{};
//    uint32_t maxVertexCount{};
};

