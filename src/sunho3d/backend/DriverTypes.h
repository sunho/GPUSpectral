#pragma once

#include <array>
#include <string>

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

enum class SamplerType {
    SAMPLER2D,
    SAMPLER3D,
    SAMPLERCUBE
};

enum class TextureFormat : uint16_t {
    R8,
    R8_SNORM,
    R8UI,
    R8I,
    R16F,
    R16UI,
    R16I,
    DEPTH16,
    // RGB8, SRGB8, RGB8_SNORM, RGB8UI, RGB8I,
    R32F,
    R32UI,
    R32I,
    RG16F,
    RG16UI,
    RG16I,
    R11F_G11F_B10F,
    RGBA8,
    SRGB8_A8,
    RGBA8_SNORM,
    RGB10_A2,
    RGBA8UI,
    RGBA8I,
    DEPTH32F,
    DEPTH24_STENCIL8,
    DEPTH32F_STENCIL8
};

enum class TextureUsage : uint8_t {
    NONE = 0x0,
    COLOR_ATTACHMENT = 0x1,
    DEPTH_ATTACHMENT = 0x2,
    SAMPLEABLE = 0x4,
    UPLOADABLE = 0x8,
    INPUT_ATTACHMENT = 0x16
};

static TextureUsage operator|(TextureUsage lhs, TextureUsage rhs) {
    return static_cast<TextureUsage>(static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

static uint8_t operator&(TextureUsage lhs, TextureUsage rhs) {
    return static_cast<uint8_t>(lhs) & static_cast<uint8_t>(rhs);
}

enum class PrimitiveMode {
    TRIANGLES,
    TRIANGLE_FANS,
    TRIANGLE_STRIPS
};

struct Attribute {
    static constexpr uint8_t FLAG_NORMALIZED = 0x1;
    static constexpr uint8_t FLAG_INTEGER_TARGET = 0x2;
    std::string name;
    uint32_t index{};
    uint32_t offset{};
    uint8_t stride{};
    ElementType type{ ElementType::BYTE };
    uint8_t flags{ 0x0 };
};

static constexpr const size_t MAX_VERTEX_ATTRIBUTE_COUNT = 16;
using AttributeArray = std::array<Attribute, MAX_VERTEX_ATTRIBUTE_COUNT>;

struct BufferDescriptor {
    uint32_t *data;
};

struct Viewport {
    int32_t left;
    int32_t bottom;
    uint32_t width;
    uint32_t height;
    int32_t right() const {
        return left + width;
    }
    int32_t top() const {
        return bottom + height;
    }
    bool operator==(const Viewport &other) const {
        return left == other.left && bottom == other.bottom && width == other.width &&
               height == other.height;
    }
};

struct RenderPassParams {
    // RenderPassFlags flags{};
    Viewport viewport{};
    // DepthRange depthRange{};
};
