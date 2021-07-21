#pragma once

#include "DriverTypes.h"
#include "Program.h"

struct HwBase {};

struct HwVertexBuffer : public HwBase {
    AttributeArray attributes{};
    uint32_t vertexCount{};
    uint8_t attributeCount{};
    HwVertexBuffer() = default;
    explicit HwVertexBuffer(uint32_t vertexCount, uint8_t attributeCount,
                            const AttributeArray &attributes)
        : vertexCount(vertexCount), attributeCount(attributeCount), attributes(attributes) {
    }
};

struct HwIndexBuffer : public HwBase {
    uint32_t count{};
    HwIndexBuffer() = default;
    explicit HwIndexBuffer(uint32_t count)
        : count(count) {
    }
};

struct HwProgram : public HwBase {
    Program program;
    HwProgram() = default;
    explicit HwProgram(const Program &program)
        : program(program) {
    }
};

struct HwTexture : public HwBase {
    SamplerType type{};
    uint8_t levels{};
    TextureFormat format{};
    uint32_t width{};
    uint32_t height{};
    HwTexture() = default;
    HwTexture(SamplerType type, uint8_t levels, TextureFormat format, uint32_t width,
              uint32_t height)
        : type(type), levels(levels), format(format), width(width), height(height) {
    }
};

struct HwRenderTarget : public HwBase {
    uint32_t width{};
    uint32_t height{};
    HwRenderTarget() = default;
    explicit HwRenderTarget(uint32_t w, uint32_t h)
        : width(w), height(h) {
    }
};

struct HwBufferObject : public HwBase {
    uint32_t size{};
    HwBufferObject() = default;
    explicit HwBufferObject(uint32_t size)
        : size(size) {
    }
};

struct HwUniformBuffer : public HwBase {
    uint32_t size{};
    HwUniformBuffer() = default;
    explicit HwUniformBuffer(uint32_t size)
        : size(size) {
    }
};

struct HwPrimitive : public HwBase {
    PrimitiveMode mode{};
    HwPrimitive() = default;
    explicit HwPrimitive(PrimitiveMode mode)
        : mode(mode) {
    }
};