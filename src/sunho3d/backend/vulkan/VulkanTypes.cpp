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
        case TextureFormat::RGBA16F:
            return vk::Format::eR16G16B16A16Sfloat;
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
    vk::BufferUsageFlags out = {};
    if (usage & BufferUsage::TRANSFER_SRC) {
        out |= vk::BufferUsageFlagBits::eTransferSrc;
    }
    if (usage & BufferUsage::VERTEX) {
        out |= vk::BufferUsageFlagBits::eVertexBuffer;
    }
    if (usage & BufferUsage::UNIFORM) {
        out |= vk::BufferUsageFlagBits::eUniformBuffer;
    }
    if (usage & BufferUsage::INDEX) {
        out |= vk::BufferUsageFlagBits::eIndexBuffer;
    }
    if (usage & BufferUsage::STORAGE) {
        out |= vk::BufferUsageFlagBits::eStorageBuffer;
    }
    if (usage & BufferUsage::TRANSFER_DST) {
        out |= vk::BufferUsageFlagBits::eTransferDst;
    }
    return out;
}

vk::DescriptorType translateDescriptorType(ProgramParameterType type) {
    switch (type) {
        case ProgramParameterType::IMAGE:
            return vk::DescriptorType::eStorageImage;
        case ProgramParameterType::UNIFORM:
            return vk::DescriptorType::eUniformBuffer;
        case ProgramParameterType::STORAGE:
            return vk::DescriptorType::eStorageBuffer;
        case ProgramParameterType::TEXTURE:
            return vk::DescriptorType::eCombinedImageSampler;
        case ProgramParameterType::ATTACHMENT:
            return vk::DescriptorType::eInputAttachment;
    }
    return vk::DescriptorType();
}

vk::PipelineStageFlags2KHR translateStageMask(BarrierStageMask mask) {
    vk::PipelineStageFlags2KHR out = {};
    if (mask & BarrierStageMask::BOTTOM_OF_PIPE) {
        out |= vk::PipelineStageFlagBits2KHR::eBottomOfPipe;
    }
    if (mask & BarrierStageMask::COLOR_ATTACHMENT_OUTPUT) {
        out |= vk::PipelineStageFlagBits2KHR::eColorAttachmentOutput;
    }
    if (mask & BarrierStageMask::COMPUTE) {
        out |= vk::PipelineStageFlagBits2KHR::eComputeShader;
    }
    if (mask & BarrierStageMask::EARLY_FRAGMENT_TESTS) {
        out |= vk::PipelineStageFlagBits2KHR::eEarlyFragmentTests;
    }
    if (mask & BarrierStageMask::FRAGMENT_SHADER) {
        out |= vk::PipelineStageFlagBits2KHR::eFragmentShader;
    }
    if (mask & BarrierStageMask::LATE_FRAGMENT_TESTS) {
        out |= vk::PipelineStageFlagBits2KHR::eLateFragmentTests;
    }
    if (mask & BarrierStageMask::TOP_OF_PIPE) {
        out |= vk::PipelineStageFlagBits2KHR::eTopOfPipe;
    }
    if (mask & BarrierStageMask::TRANSFER) {
        out |= vk::PipelineStageFlagBits2KHR::eTransfer;
    }
    if (mask & BarrierStageMask::VERTEX_SHADER) {
        out |= vk::PipelineStageFlagBits2KHR::eVertexShader;
    }
    return out;
}

vk::AccessFlags2KHR translateAccessMask(BarrierAccessFlag flag) {
    switch (flag) {
        case BarrierAccessFlag::SHADER_READ:
            return vk::AccessFlagBits2KHR::eShaderRead;
        case BarrierAccessFlag::SHADER_WRITE:
            return vk::AccessFlagBits2KHR::eShaderWrite;
        case BarrierAccessFlag::COLOR_WRITE:
            return vk::AccessFlagBits2KHR::eColorAttachmentWrite;
        case BarrierAccessFlag::DEPTH_STENCIL_WRITE:
            return vk::AccessFlagBits2KHR::eDepthStencilAttachmentWrite;
    }
}

vk::ImageLayout translateImageLayout(ImageLayout layout) {
    switch (layout) {
        case ImageLayout::NONE:
            return vk::ImageLayout::eUndefined;
        case ImageLayout::COLOR_ATTACHMENT_OPTIMAL:
            return vk::ImageLayout::eColorAttachmentOptimal;
        case ImageLayout::DEPTH_ATTACHMENT_OPTIMAL:
            return vk::ImageLayout::eDepthAttachmentOptimal;
        case ImageLayout::READ_ONLY_OPTIMAL:
            return vk::ImageLayout::eReadOnlyOptimalKHR;
        case ImageLayout::SHADER_READ_ONLY_OPTIMAL:
            return vk::ImageLayout::eShaderReadOnlyOptimal;
    }
}
