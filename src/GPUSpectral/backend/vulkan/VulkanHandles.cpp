#include "VulkanHandles.h"
#include <spirv_cross.hpp>

inline static VkShaderModule createShaderModule(VulkanDevice& device, const uint32_t* code,
    uint32_t codeSize) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = codeSize;
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code);

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device.device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }

    return shaderModule;
}

VulkanProgram::VulkanProgram(VulkanDevice& device, const Program& program)
    : HwProgram(program) {
        shaderModule = createShaderModule(device, program.code().data(), program.code().size() * 4);
        parseParameterLayout(program.code());

}

void VulkanProgram::parseParameterLayout(const CompiledCode& code) {
    spirv_cross::Compiler refl(code.data(), code.size());
    auto resources = refl.get_shader_resources();
    for (auto& ub : resources.uniform_buffers) {
        auto type = refl.get_type(ub.type_id);
        auto binding = refl.get_decoration(ub.id, spv::DecorationBinding);
        auto set = refl.get_decoration(ub.id, spv::DecorationDescriptorSet);
        size_t arraySize = type.array.size() > 0 ? type.array[0] : 1;
        program.parameterLayout.addUniformBufferArray(set, binding, arraySize);
    }

    for (auto& sb : resources.storage_buffers) {
        auto type = refl.get_type(sb.type_id);
        auto binding = refl.get_decoration(sb.id, spv::DecorationBinding);
        auto set = refl.get_decoration(sb.id, spv::DecorationDescriptorSet);
        size_t arraySize = type.array.size() > 0 ? type.array[0] : 1;
        program.parameterLayout.addStorageBufferArray(set, binding, arraySize);
    }

    for (auto& si : resources.storage_images) {
        auto type = refl.get_type(si.type_id);
        auto binding = refl.get_decoration(si.id, spv::DecorationBinding);
        auto set = refl.get_decoration(si.id, spv::DecorationDescriptorSet);
        size_t arraySize = type.array.size() > 0 ? type.array[0] : 1;
        program.parameterLayout.addStorageImageArray(set, binding, arraySize);
    }

    for (auto& si : resources.sampled_images) {
        auto type = refl.get_type(si.type_id);
        auto binding = refl.get_decoration(si.id, spv::DecorationBinding);
        auto set = refl.get_decoration(si.id, spv::DecorationDescriptorSet);
        size_t arraySize = type.array.size() > 0 ? type.array[0] : 1;
        program.parameterLayout.addTextureArray(set, binding, arraySize);
    }

    for (auto& ai : resources.acceleration_structures) {
        auto type = refl.get_type(ai.type_id);
        auto binding = refl.get_decoration(ai.id, spv::DecorationBinding);
        auto set = refl.get_decoration(ai.id, spv::DecorationDescriptorSet);
        size_t arraySize = type.array.size() > 0 ? type.array[0] : 1;
        program.parameterLayout.addTLASArray(set, binding, arraySize);
    }
}

VulkanRenderTarget::VulkanRenderTarget()
    : HwRenderTarget(0, 0), surface(true), attachmentCount(1) {
}

VulkanRenderTarget::VulkanRenderTarget(uint32_t w, uint32_t h, VulkanAttachments attachments)
    : HwRenderTarget(w, h), attachments(attachments), surface(false) {
    for (size_t i = 0; i < RenderAttachments::MAX_MRT_NUM; ++i) {
        if (attachments.colors[i].valid)
            ++attachmentCount;
    }
}

vk::Extent2D VulkanRenderTarget::getExtent(VulkanDevice& device) const {
    if (surface) {
        return device.wsi->getExtent();
    }
    auto outExtent = vk::Extent2D();
    if (width == HwTexture::FRAME_WIDTH) {
        outExtent.width = device.wsi->getExtent().width;
    }
    else {
        outExtent.width = width;
    }
    if (height == HwTexture::FRAME_HEIGHT) {
        outExtent.height = device.wsi->getExtent().height;
    }
    else {
        outExtent.height = height;
    }
    return outExtent;
}
