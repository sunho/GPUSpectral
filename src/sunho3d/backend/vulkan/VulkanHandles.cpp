#include "VulkanHandles.h"
#include <spirv_cross/spirv_cross.hpp>

inline static VkShaderModule createShaderModule(VulkanDevice& device, const char* code,
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
    if (program.type == ProgramType::PIPELINE) {
        vertex = createShaderModule(device, program.vertex().data(), program.vertex().size());
        fragment = createShaderModule(device, program.frag().data(), program.frag().size());
    }
    else {
        compute = createShaderModule(device, program.compute().data(), program.compute().size());
    }
    parseParameterLayout();
}

void VulkanProgram::parseParameterLayout() {

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
