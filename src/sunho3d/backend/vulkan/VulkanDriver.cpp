#include "VulkanDriver.h"

#include <iostream>
#include <stdexcept>
#include <GLFW/glfw3.h>
#include <vector>
#include <set>
#include <map>

#include <sunho3d/shaders/triangle_vert.h>
#include <sunho3d/shaders/triangle_frag.h>

using namespace sunho3d;

static void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

static VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

VulkanDriver::VulkanDriver(Window* window) {
    initContext(context);
    initSurfaceContext(context, surface, window);
    pickPhysicalDevice(context, surface);
    createLogicalDevice(context, surface);
    createSwapChain(context, surface, window);
    populateSwapContexts(context, surface);
    setupDebugMessenger();
}

VulkanDriver::~VulkanDriver() {
    vkDestroyPipeline(context.device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
    for (auto framebuffer : framebuffers) {
        vkDestroyFramebuffer(context.device, framebuffer, nullptr);
    }

    destroyContext(context, surface);
    DestroyDebugUtilsMessengerEXT(context.instance, debugMessenger, nullptr);
    vkDestroyInstance(context.instance, nullptr);

}

RenderTargetHandle VulkanDriver::createDefaultRenderTarget(int dummy) {
    Handle<HwRenderTarget> handle = alloc_handle<VulkanRenderTarget, HwRenderTarget>();
    construct_handle<VulkanRenderTarget>(handle, surface.extent.width, surface.extent.height);
    return handle;
}

VertexBufferHandle VulkanDriver::createVertexBuffer(uint32_t vertexCount, uint8_t attributeCount, AttributeArray attributes) {
    Handle<HwVertexBuffer> handle = alloc_handle<VulkanVertexBuffer, HwVertexBuffer>();
    construct_handle<VulkanVertexBuffer>(handle, vertexCount, attributeCount, attributes);
    return handle;
}

IndexBufferHandle VulkanDriver::createIndexBuffer(uint32_t indexCount) {
    Handle<HwIndexBuffer> handle = alloc_handle<VulkanIndexBuffer, HwIndexBuffer>();
    construct_handle<VulkanIndexBuffer>(handle, indexCount);
    handle_cast<VulkanIndexBuffer>(handle)->allocate(context);
    return handle;
}


ProgramHandle VulkanDriver::createProgram(Program program) {
    Handle<HwProgram> handle = alloc_handle<VulkanProgram, HwProgram>();
    construct_handle<VulkanProgram>(handle, program);
    handle_cast<VulkanProgram>(handle)->compile(context);
    return handle;
}

BufferObjectHandle VulkanDriver::createBufferObject(uint32_t size) {
    Handle<HwBufferObject> handle = alloc_handle<VulkanBufferObject, HwBufferObject>();
    construct_handle<VulkanBufferObject>(handle, size);
    handle_cast<VulkanBufferObject>(handle)->allocate(context, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    
    return handle;
}

void VulkanDriver::setVertexBuffer(VertexBufferHandle handle, BufferObjectHandle bufferObject) {
    handle_cast<VulkanVertexBuffer>(handle)->buffer = handle_cast<VulkanBufferObject>(bufferObject);
}

void VulkanDriver::updateIndexBuffer(IndexBufferHandle handle, BufferDescriptor data, uint32_t offset) {
    handle_cast<VulkanIndexBuffer>(handle)->buffer->upload(context, data);
}

void VulkanDriver::updateBufferObject(BufferObjectHandle handle, BufferDescriptor data, uint32_t offset) {
    handle_cast<VulkanBufferObject>(handle)->upload(context, data);
}

VkRenderPass VulkanDriver::createRenderPass() {
    VkRenderPass out;
    VkAttachmentDescription colorAttachment{
        .format = surface.format.format,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
    };
    VkAttachmentReference colorAttachmentRef{
        .attachment = 0,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    };

    VkSubpassDescription subpass{
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachmentRef
    };

    VkRenderPassCreateInfo renderPassInfo{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &colorAttachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
    };
    
    
    
    if (vkCreateRenderPass(context.device, &renderPassInfo, nullptr, &out) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }
    
    renderPasses.push_back(out);
    return out;
}

void VulkanDriver::beginRenderPass(RenderTargetHandle renderTarget, RenderPassParams params) {
    const VkCommandBuffer cmdbuffer = surface.currentContext->commands;

    VkRenderPass renderPass = createRenderPass();
    VkImageView attachments[] = {
         surface.currentContext->attachment.view
    };
    VkFramebuffer framebuffer;
    VkFramebufferCreateInfo framebufferInfo{
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .renderPass = renderPass,
        .attachmentCount = 1,
        .pAttachments = attachments,
        .width = surface.extent.width,
        .height = surface.extent.height,
        .layers = 1
    };

    if (vkCreateFramebuffer(context.device, &framebufferInfo, nullptr, &framebuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create framebuffer!");
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0; // Optional
    beginInfo.pInheritanceInfo = nullptr; // Optional

    if (vkBeginCommandBuffer(cmdbuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
    }

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = framebuffer;
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = surface.extent;
    VkClearValue clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;
    vkCmdBeginRenderPass(cmdbuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    
    VkViewport viewport = {
       .x = (float) params.viewport.left,
       .y = (float) params.viewport.bottom,
       .width = (float) params.viewport.width,
       .height = (float) params.viewport.height,
//       .minDepth = params.depthRange.near,
//       .maxDepth = params.depthRange.far
   };
   //vkCmdSetViewport(cmdbuffer, 0, 1, &viewport);
    context.currentRenderPass = renderPassInfo.renderPass;
}

PrimitiveHandle VulkanDriver::createPrimitive(int dummy) {
    Handle<HwPrimitive> handle = alloc_handle<VulkanPrimitive, HwPrimitive>();
    construct_handle<VulkanPrimitive>(handle);
    return handle;
}

void VulkanDriver::setPrimitiveBuffer(PrimitiveHandle handle, VertexBufferHandle vertexBuffer, IndexBufferHandle indexBuffer) {
    VulkanVertexBuffer* vertex = handle_cast<VulkanVertexBuffer>(vertexBuffer);
    VulkanIndexBuffer* index = handle_cast<VulkanIndexBuffer>(indexBuffer);
    handle_cast<VulkanPrimitive>(handle)->index = index;
    handle_cast<VulkanPrimitive>(handle)->vertex = vertex;
}

VkPipeline VulkanDriver::createPipeline(const std::vector<VkVertexInputAttributeDescription>& attributes, const std::vector<VkVertexInputBindingDescription>& bindings, VkShaderModule vertex, VkShaderModule frag) {
      VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
      vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
      vertShaderStageInfo.module = vertex;
      vertShaderStageInfo.pName = "main";
      
      VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
      fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
      fragShaderStageInfo.module = frag;
      fragShaderStageInfo.pName = "main";
      VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};
      
      VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
      vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = bindings.size();
        vertexInputInfo.pVertexBindingDescriptions = bindings.data();
        vertexInputInfo.vertexAttributeDescriptionCount = attributes.size();
        vertexInputInfo.pVertexAttributeDescriptions = attributes.data();
      
      VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
      inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
      inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      inputAssembly.primitiveRestartEnable = VK_FALSE;

      VkViewport viewport{};
      viewport.x = 0.0f;
      viewport.y = 0.0f;
      viewport.width = (float) surface.extent.width;
      viewport.height = (float) surface.extent.height;
      viewport.minDepth = 0.0f;
      viewport.maxDepth = 1.0f;
      
      VkRect2D scissor{};
      scissor.offset = {0, 0};
      scissor.extent = surface.extent;
      
      VkPipelineViewportStateCreateInfo viewportState{};
      viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
      viewportState.viewportCount = 1;
      viewportState.pViewports = &viewport;
      viewportState.scissorCount = 1;
      viewportState.pScissors = &scissor;
      
      VkPipelineRasterizationStateCreateInfo rasterizer{};
      rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
      rasterizer.depthClampEnable = VK_FALSE;
      rasterizer.rasterizerDiscardEnable = VK_FALSE;
      rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
      rasterizer.lineWidth = 1.0f;
      rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
      rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
      rasterizer.depthBiasEnable = VK_FALSE;
      rasterizer.depthBiasConstantFactor = 0.0f; // Optional
      rasterizer.depthBiasClamp = 0.0f; // Optional
      rasterizer.depthBiasSlopeFactor = 0.0f; // Optional
      
      VkPipelineMultisampleStateCreateInfo multisampling{};
      multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
      multisampling.sampleShadingEnable = VK_FALSE;
      multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
      multisampling.minSampleShading = 1.0f; // Optional
      multisampling.pSampleMask = nullptr; // Optional
      multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
      multisampling.alphaToOneEnable = VK_FALSE; // Optional
      
      VkPipelineColorBlendAttachmentState colorBlendAttachment{};
      colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
      colorBlendAttachment.blendEnable = VK_FALSE;
      colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
      colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
      colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
      colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
      colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
      colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional
      
      colorBlendAttachment.blendEnable = VK_TRUE;
      colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
      colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
      colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
      colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
      
      VkPipelineColorBlendStateCreateInfo colorBlending{};
      colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
      colorBlending.logicOpEnable = VK_FALSE;
      colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
      colorBlending.attachmentCount = 1;
      colorBlending.pAttachments = &colorBlendAttachment;
      colorBlending.blendConstants[0] = 0.0f; // Optional
      colorBlending.blendConstants[1] = 0.0f; // Optional
      colorBlending.blendConstants[2] = 0.0f; // Optional
      colorBlending.blendConstants[3] = 0.0f; // Optional
      
      VkDynamicState dynamicStates[] = {
          VK_DYNAMIC_STATE_VIEWPORT,
          VK_DYNAMIC_STATE_LINE_WIDTH
      };

      VkPipelineDynamicStateCreateInfo dynamicState{};
      dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
      dynamicState.dynamicStateCount = 2;
      dynamicState.pDynamicStates = dynamicStates;
      
      VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
      pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
      pipelineLayoutInfo.setLayoutCount = 0; // Optional
      pipelineLayoutInfo.pSetLayouts = nullptr; // Optional
      pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
      pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

      if (vkCreatePipelineLayout(context.device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
          throw std::runtime_error("failed to create pipeline layout!");
      }
      
      VkGraphicsPipelineCreateInfo pipelineInfo{};
      pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
      pipelineInfo.stageCount = 2;
      pipelineInfo.pStages = shaderStages;
      pipelineInfo.pVertexInputState = &vertexInputInfo;
      pipelineInfo.pInputAssemblyState = &inputAssembly;
      pipelineInfo.pViewportState = &viewportState;
      pipelineInfo.pRasterizationState = &rasterizer;
      pipelineInfo.pMultisampleState = &multisampling;
      pipelineInfo.pDepthStencilState = nullptr; // Optional
      pipelineInfo.pColorBlendState = &colorBlending;
      pipelineInfo.pDynamicState = nullptr; // Optional
      pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = context.currentRenderPass;
      pipelineInfo.subpass = 0;
      pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
      pipelineInfo.basePipelineIndex = -1; // Optional
      if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
          throw std::runtime_error("failed to create graphics pipeline!");
      }
    
    return graphicsPipeline;
}

void VulkanDriver::draw(PipelineState pipeline, PrimitiveHandle handle) {
    VulkanPrimitive* prim = handle_cast<VulkanPrimitive>(handle);
    
    const VkCommandBuffer cmdbuffer = surface.currentContext->commands;

    const uint32_t bufferCount = prim->vertex->attributeCount;
    VkBuffer buffers[MAX_VERTEX_ATTRIBUTE_COUNT] = {};
    VkDeviceSize offsets[MAX_VERTEX_ATTRIBUTE_COUNT] = {};
    std::vector<VkVertexInputAttributeDescription> attributes(bufferCount);
    std::vector<VkVertexInputBindingDescription> bindings(bufferCount);

    for (uint32_t i = 0; i < bufferCount; ++i) {
        Attribute attrib = prim->vertex->attributes[i];
        VulkanBufferObject* buffer = prim->vertex->buffer;

        buffers[i] = buffer->buffer;
        offsets[i] = attrib.offset;
        attributes[i] = {
            .location = i,
            .binding = i,
            .format = VK_FORMAT_R32G32_SFLOAT
        };
        bindings[i] = {
            .binding = i,
            .stride = attrib.stride,
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
        };
    }

    VulkanProgram* program = handle_cast<VulkanProgram>(pipeline.program);
    
    VkPipeline pl = createPipeline(attributes, bindings, program->vertex, program->fragment);
    vkCmdBindPipeline(cmdbuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pl);
    vkCmdBindVertexBuffers(cmdbuffer, 0, bufferCount, buffers, offsets);
    vkCmdBindIndexBuffer(cmdbuffer, prim->index->buffer->buffer, 0, VK_INDEX_TYPE_UINT16);
    vkCmdDraw(cmdbuffer, prim->vertex->vertexCount, 1, 0, 0);
}

void VulkanDriver::endRenderPass(int dummy) {
    const VkCommandBuffer cmdbuffer = surface.currentContext->commands;
    vkCmdEndRenderPass(cmdbuffer);
}

void VulkanDriver::commit(int dummy) {
    uint32_t imageIndex2;
    vkAcquireNextImageKHR(context.device, surface.swapChain, UINT64_MAX, surface.imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex2);
    const VkCommandBuffer cmdbuffer = surface.currentContext->commands;
    vkEndCommandBuffer(cmdbuffer);
    
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = {surface.imageAvailableSemaphore};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &surface.currentContext->commands;
    VkSemaphore signalSemaphores[] = {surface.renderFinishedSemaphore};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;
    if (vkQueueSubmit(context.graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
    }
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    uint32_t imageIndex = surface.swapContextIndex;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    VkSwapchainKHR swapChains[] = {surface.swapChain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr;
    vkQueuePresentKHR(surface.presentQueue, &presentInfo);
    
    surface.swapContextIndex =(surface.swapContextIndex + 1 )% surface.swapContexts.size();
}

void VulkanDriver::drawFrame() {

}

VkBool32 VulkanDriver::debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    }
    return VK_FALSE;
}

void VulkanDriver::setupDebugMessenger() { 
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = VulkanDriver::debugCallback;
    createInfo.pUserData = nullptr;
    if (CreateDebugUtilsMessengerEXT(context.instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
        throw std::runtime_error("failed to set up debug messenger!");
    }
}
