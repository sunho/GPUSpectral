#pragma once

#include <array>

enum class ElementType : uint8_t {
    BYTE,
    BYTE2,
    BYTE3,
    BYTE4,
    UBYTE,
    UBYTE2,
    UBYTE3,
    UBYTE4,
    SHORT,
    SHORT2,
    SHORT3,
    SHORT4,
    USHORT,
    USHORT2,
    USHORT3,
    USHORT4,
    INT,
    UINT,
    FLOAT,
    FLOAT2,
    FLOAT3,
    FLOAT4,
    HALF,
    HALF2,
    HALF3,
    HALF4,
};

struct Attribute {
    static constexpr uint8_t FLAG_NORMALIZED     = 0x1;
    static constexpr uint8_t FLAG_INTEGER_TARGET = 0x2;
    uint32_t offset{};
    uint8_t stride{};
    ElementType type{ElementType::BYTE};
    uint8_t flags{0x0};
};

static constexpr const size_t MAX_VERTEX_ATTRIBUTE_COUNT = 16;
using AttributeArray = std::array<Attribute, MAX_VERTEX_ATTRIBUTE_COUNT>;

struct BufferDescriptor {
    uint32_t* data;
};

struct Viewport {
    int32_t left;
    int32_t bottom;
    uint32_t width;
    uint32_t height;
    int32_t right() const { return left + width; }
    int32_t top() const { return bottom + height; }
};

struct RenderPassParams {
    //RenderPassFlags flags{};
    Viewport viewport{};
    //DepthRange depthRange{};
};
