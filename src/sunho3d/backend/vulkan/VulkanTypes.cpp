#include "VulkanTypes.h"

vk::Format translateTextureFormat(TextureFormat format) {
    switch (format) {
        case TextureFormat::R8:
            return vk::Format::eR8Unorm;
        case TextureFormat::R8_SNORM:
            return vk::Format::eR8Snorm;
        case TextureFormat::R8UI:
            return vk::Format::eR8Uint;
        case TextureFormat::R8I:
            return vk::Format::eR8Sint;
        case TextureFormat::R16F:
            return vk::Format::eR16Sfloat;
        case TextureFormat::R16UI:
            return vk::Format::eR16Uint;
        case TextureFormat::R16I:
            return vk::Format::eR16Sint;
        case TextureFormat::DEPTH16:
            return vk::Format::eD16Unorm;
        case TextureFormat::R32F:
            return vk::Format::eR32Sfloat;
        case TextureFormat::R32UI:
            return vk::Format::eR32Uint;
        case TextureFormat::R32I:
            return vk::Format::eR32Sint;
        case TextureFormat::RG16F:
            return vk::Format::eR16G16Sfloat;
        case TextureFormat::RG16UI:
            return vk::Format::eR16G16Uint;
        case TextureFormat::RG16I:
            return vk::Format::eR16G16Sint;
        case TextureFormat::R11F_G11F_B10F:
            return vk::Format::eB10G11R11UfloatPack32;
        case TextureFormat::RGBA8:
            return vk::Format::eR8G8B8A8Unorm;
        case TextureFormat::SRGB8_A8:
            return vk::Format::eR8G8B8A8Srgb;
        case TextureFormat::RGBA8_SNORM:
            return vk::Format::eR8G8B8A8Snorm;
        case TextureFormat::RGB10_A2:
            return vk::Format::eA2B10G10R10UnormPack32;
        case TextureFormat::RGBA8UI:
            return vk::Format::eR8G8B8A8Uint;
        case TextureFormat::RGBA8I:
            return vk::Format::eR8G8B8A8Sint;
        case TextureFormat::DEPTH32F:
            return vk::Format::eD32Sfloat;
        case TextureFormat::DEPTH24_STENCIL8:
            return vk::Format::eD24UnormS8Uint;
        case TextureFormat::DEPTH32F_STENCIL8:
            return vk::Format::eD32SfloatS8Uint;
        default: {
            assert(false);
        }
    }
}

vk::Format translateElementFormat(ElementType type, bool normalized, bool integer) {
    using ElementType = ElementType;
    if (normalized) {
        switch (type) {
            case ElementType::BYTE:
                return vk::Format::eR8Snorm;
            case ElementType::UBYTE:
                return vk::Format::eR8Unorm;;
            case ElementType::SHORT:
                return vk::Format::eR16Snorm;
            case ElementType::USHORT:
                return vk::Format::eR16Unorm;
            case ElementType::BYTE2:
                return vk::Format::eR8G8Snorm;
            case ElementType::UBYTE2:
                return vk::Format::eR8G8Unorm;
            case ElementType::SHORT2:
                return vk::Format::eR16G16Snorm;
            case ElementType::USHORT2:
                return vk::Format::eR16G16Unorm;
            case ElementType::BYTE3:
                return vk::Format::eR8G8B8Snorm;
            case ElementType::UBYTE3:
                return vk::Format::eR8G8B8Unorm;
            case ElementType::SHORT3:
                return vk::Format::eR16G16B16Snorm;
            case ElementType::USHORT3:
                return vk::Format::eR16G16B16Unorm;
            case ElementType::BYTE4:
                return vk::Format::eR8G8B8A8Snorm;
            case ElementType::UBYTE4:
                return vk::Format::eR8G8B8A8Unorm;
            case ElementType::SHORT4:
                return vk::Format::eR16G16B16A16Snorm;
            case ElementType::USHORT4:
                return vk::Format::eR16G16B16A16Unorm;
            default:
                return vk::Format::eUndefined;
        }
    }
    switch (type) {
        case ElementType::BYTE:
            return integer ? vk::Format::eR8Sint : vk::Format::eR8Sscaled;
        case ElementType::UBYTE:
            return integer ? vk::Format::eR8Uint : vk::Format::eR8Uscaled;
        case ElementType::SHORT:
            return integer ? vk::Format::eR16Sint : vk::Format::eR16Sscaled;
        case ElementType::USHORT:
            return integer ? vk::Format::eR16Uint : vk::Format::eR16Uscaled;
        case ElementType::HALF:
            return vk::Format::eR16Sfloat;
        case ElementType::INT:
            return vk::Format::eR32Sint;
        case ElementType::UINT:
            return vk::Format::eR32Uint;
        case ElementType::FLOAT:
            return vk::Format::eR32Sfloat;
        case ElementType::BYTE2:
            return integer ? vk::Format::eR8G8Sint : vk::Format::eR8G8Sscaled;
        case ElementType::UBYTE2:
            return integer ? vk::Format::eR8G8Uint : vk::Format::eR8G8Uscaled;
        case ElementType::SHORT2:
            return integer ? vk::Format::eR16G16Sint : vk::Format::eR16G16Sscaled;
        case ElementType::USHORT2:
            return integer ? vk::Format::eR16G16Uint : vk::Format::eR16G16Uscaled;
        case ElementType::HALF2:
            return vk::Format::eR16G16Sfloat;
        case ElementType::FLOAT2:
            return vk::Format::eR32G32Sfloat;
        case ElementType::BYTE3:
            return vk::Format::eR8G8B8Sint;
        case ElementType::UBYTE3:
            return vk::Format::eR8G8B8Uint;
        case ElementType::SHORT3:
            return vk::Format::eR16G16B16Sint;
        case ElementType::USHORT3:
            return vk::Format::eR16G16B16Uint;
        case ElementType::HALF3:
            return vk::Format::eR16G16B16Sfloat;
        case ElementType::FLOAT3:
            return vk::Format::eR32G32B32Sfloat;
        case ElementType::BYTE4:
            return integer ? vk::Format::eR8G8B8A8Sint : vk::Format::eR8G8B8A8Sscaled;
        case ElementType::UBYTE4:
            return integer ? vk::Format::eR8G8B8A8Uint : vk::Format::eR8G8B8A8Uscaled;
        case ElementType::SHORT4:
            return integer ? vk::Format::eR16G16B16A16Sint : vk::Format::eR16G16B16A16Sscaled;
        case ElementType::USHORT4:
            return integer ? vk::Format::eR16G16B16A16Uint : vk::Format::eR16G16B16A16Uscaled;
        case ElementType::HALF4:
            return vk::Format::eR16G16B16A16Sfloat;
        case ElementType::FLOAT4:
            return vk::Format::eR32G32B32A32Sfloat;
    }
    return vk::Format::eUndefined;
}

vk::CompareOp translateCompareOp(CompareOp op) {
    switch (op) {
        case CompareOp::ALWAYS:
            return vk::CompareOp::eAlways;
        case CompareOp::EQUAL:
            return vk::CompareOp::eEqual;
        case CompareOp::GREATER:
            return vk::CompareOp::eGreater;
        case CompareOp::GREATER_OR_EQUAL:
            return vk::CompareOp::eGreaterOrEqual;
        case CompareOp::LESS:
            return vk::CompareOp::eLess;
        case CompareOp::LESS_OR_EQUAL:
            return vk::CompareOp::eLessOrEqual;
        case CompareOp::NEVER:
            return vk::CompareOp::eNever;
        case CompareOp::NOT_EQUAL:
            return vk::CompareOp::eNotEqual;
    };
}

vk::BufferUsageFlags translateBufferUsage(BufferUsage usage) {
    switch (usage) {
        case BufferUsage::TRANSFER_SRC:
            return vk::BufferUsageFlagBits::eTransferSrc;
        case BufferUsage::VERTEX:
            return vk::BufferUsageFlagBits::eVertexBuffer;
        case BufferUsage::INDEX:
            return vk::BufferUsageFlagBits::eIndexBuffer;
        case BufferUsage::UNIFORM:
            return vk::BufferUsageFlagBits::eUniformBuffer;
    }
}