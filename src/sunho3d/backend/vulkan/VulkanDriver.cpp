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
    pipelineCache.init(context);
}

VulkanDriver::~VulkanDriver() {
    destroyContext(context, surface);
    DestroyDebugUtilsMessengerEXT(context.instance, debugMessenger, nullptr);
    vkDestroyInstance(context.instance, nullptr);

}

RenderTargetHandle VulkanDriver::createDefaultRenderTarget(int dummy) {
    VulkanAttachment depth;
    auto tex = VulkanTexture(context, SamplerType::SAMPLER2D, TextureUsage::DEPTH_ATTACHMENT, 1, TextureFormat::DEPTH32F, surface.extent.width, surface.extent.height);
    depth.image = tex.image;
    depth.format = tex.vkFormat;
    depth.view = tex.view;
    Handle<HwRenderTarget> handle = alloc_handle<VulkanRenderTarget, HwRenderTarget>();
    construct_handle<VulkanRenderTarget>(handle, surface.extent.width, surface.extent.height, depth);
    return handle;
}

VertexBufferHandle VulkanDriver::createVertexBuffer(uint32_t bufferCount, uint32_t vertexCount, uint8_t attributeCount, AttributeArray attributes) {
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

void VulkanDriver::setVertexBuffer(VertexBufferHandle handle, uint32_t index, BufferObjectHandle bufferObject) {
    handle_cast<VulkanVertexBuffer>(handle)->buffers[index] = handle_cast<VulkanBufferObject>(bufferObject);
}

void VulkanDriver::updateIndexBuffer(IndexBufferHandle handle, BufferDescriptor data, uint32_t offset) {
    handle_cast<VulkanIndexBuffer>(handle)->buffer->upload(context, data);
}

void VulkanDriver::updateBufferObject(BufferObjectHandle handle, BufferDescriptor data, uint32_t offset) {
    handle_cast<VulkanBufferObject>(handle)->upload(context, data);
}

void VulkanDriver::beginRenderPass(RenderTargetHandle renderTarget, RenderPassParams params) {
    vkAcquireNextImageKHR(context.device, surface.swapChain, UINT64_MAX, context.commands.imageAvailableSemaphore(), VK_NULL_HANDLE, &surface.swapContextIndex);
    context.currentSwapContext = &surface.swapContexts[surface.swapContextIndex];
    
    const VkCommandBuffer cmdbuffer = context.commands.get();
    VulkanRenderTarget* rt = handle_cast<VulkanRenderTarget>(renderTarget);
    
    VkRenderPass renderPass = pipelineCache.getOrCreateRenderPass(context, rt);
    VkFramebuffer frameBuffer = pipelineCache.getOrCreateFrameBuffer(context, renderPass, rt);
    
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
    renderPassInfo.framebuffer = frameBuffer;
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = surface.extent;
    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {0.0f, 0.0f, 0.0f, 1.0f};
    clearValues[1].depthStencil = {1.0f, 0};
    renderPassInfo.clearValueCount = 2;
    renderPassInfo.pClearValues = clearValues.data();
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
    context.currentRenderPass = renderPass;
}

PrimitiveHandle VulkanDriver::createPrimitive(PrimitiveMode mode) {
    Handle<HwPrimitive> handle = alloc_handle<VulkanPrimitive, HwPrimitive>();
    construct_handle<VulkanPrimitive>(handle, mode);
    return handle;
}

TextureHandle VulkanDriver::createTexture(SamplerType type, TextureUsage usage, TextureFormat format, uint32_t width, uint32_t height) {
    Handle<HwTexture> handle = alloc_handle<VulkanTexture, HwTexture>();
    construct_handle<VulkanTexture>(handle, context, type, usage, 1, format, width, height);
    return handle;
}

void VulkanDriver::updateTexture(TextureHandle handle, BufferDescriptor data) {
    handle_cast<VulkanTexture>(handle)->update2DImage(context, data);
}

void VulkanDriver::setPrimitiveBuffer(PrimitiveHandle handle, VertexBufferHandle vertexBuffer, IndexBufferHandle indexBuffer) {
    VulkanVertexBuffer* vertex = handle_cast<VulkanVertexBuffer>(vertexBuffer);
    VulkanIndexBuffer* index = handle_cast<VulkanIndexBuffer>(indexBuffer);
    handle_cast<VulkanPrimitive>(handle)->index = index;
    handle_cast<VulkanPrimitive>(handle)->vertex = vertex;
}

void VulkanDriver::updateUniformBuffer(UniformBufferHandle handle, BufferDescriptor data, uint32_t offset) {
    handle_cast<VulkanUniformBuffer>(handle)->buffer->upload(context, data);
}

void VulkanDriver::bindUniformBuffer(uint32_t binding, UniformBufferHandle handle) {
    const VkCommandBuffer cmdbuffer = context.commands.get();
    
    VulkanUniformBuffer* ubo = handle_cast<VulkanUniformBuffer>(handle);
    currentBinding.uniformBuffers[binding] = ubo->buffer->buffer;
    currentBinding.uniformBufferOffsets[binding] = 0;
    currentBinding.uniformBufferSizes[binding] = ubo->size;
    
    pipelineCache.bindDescriptors(context, currentBinding);
}

void VulkanDriver::bindTexture(uint32_t binding, TextureHandle handle) {
    const VkCommandBuffer cmdbuffer = context.commands.get();
       
    VulkanTexture* tex = handle_cast<VulkanTexture>(handle);
    VkDescriptorImageInfo info;
    info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    info.imageView = tex->view;
    info.sampler = tex->sampler;
    currentBinding.samplers[binding] = info;

    pipelineCache.bindDescriptors(context, currentBinding);
}

inline VkFormat getVkFormat(ElementType type, bool normalized, bool integer) {
    using ElementType = ElementType;
    if (normalized) {
        switch (type) {
            case ElementType::BYTE: return VK_FORMAT_R8_SNORM;
            case ElementType::UBYTE: return VK_FORMAT_R8_UNORM;
            case ElementType::SHORT: return VK_FORMAT_R16_SNORM;
            case ElementType::USHORT: return VK_FORMAT_R16_UNORM;
            case ElementType::BYTE2: return VK_FORMAT_R8G8_SNORM;
            case ElementType::UBYTE2: return VK_FORMAT_R8G8_UNORM;
            case ElementType::SHORT2: return VK_FORMAT_R16G16_SNORM;
            case ElementType::USHORT2: return VK_FORMAT_R16G16_UNORM;
            case ElementType::BYTE3: return VK_FORMAT_R8G8B8_SNORM;
            case ElementType::UBYTE3: return VK_FORMAT_R8G8B8_UNORM;
            case ElementType::SHORT3: return VK_FORMAT_R16G16B16_SNORM;
            case ElementType::USHORT3: return VK_FORMAT_R16G16B16_UNORM;
            case ElementType::BYTE4: return VK_FORMAT_R8G8B8A8_SNORM;
            case ElementType::UBYTE4: return VK_FORMAT_R8G8B8A8_UNORM;
            case ElementType::SHORT4: return VK_FORMAT_R16G16B16A16_SNORM;
            case ElementType::USHORT4: return VK_FORMAT_R16G16B16A16_UNORM;
            default:
                return VK_FORMAT_UNDEFINED;
        }
    }
    switch (type) {
        case ElementType::BYTE: return integer ? VK_FORMAT_R8_SINT : VK_FORMAT_R8_SSCALED;
        case ElementType::UBYTE: return integer ? VK_FORMAT_R8_UINT : VK_FORMAT_R8_USCALED;
        case ElementType::SHORT: return integer ? VK_FORMAT_R16_SINT : VK_FORMAT_R16_SSCALED;
        case ElementType::USHORT: return integer ? VK_FORMAT_R16_UINT : VK_FORMAT_R16_USCALED;
        case ElementType::HALF: return VK_FORMAT_R16_SFLOAT;
        case ElementType::INT: return VK_FORMAT_R32_SINT;
        case ElementType::UINT: return VK_FORMAT_R32_UINT;
        case ElementType::FLOAT: return VK_FORMAT_R32_SFLOAT;
        case ElementType::BYTE2: return integer ? VK_FORMAT_R8G8_SINT : VK_FORMAT_R8G8_SSCALED;
        case ElementType::UBYTE2: return integer ? VK_FORMAT_R8G8_UINT : VK_FORMAT_R8G8_USCALED;
        case ElementType::SHORT2: return integer ? VK_FORMAT_R16G16_SINT : VK_FORMAT_R16G16_SSCALED;
        case ElementType::USHORT2: return integer ? VK_FORMAT_R16G16_UINT : VK_FORMAT_R16G16_USCALED;
        case ElementType::HALF2: return VK_FORMAT_R16G16_SFLOAT;
        case ElementType::FLOAT2: return VK_FORMAT_R32G32_SFLOAT;
        case ElementType::BYTE3: return VK_FORMAT_R8G8B8_SINT;
        case ElementType::UBYTE3: return VK_FORMAT_R8G8B8_UINT;
        case ElementType::SHORT3: return VK_FORMAT_R16G16B16_SINT;
        case ElementType::USHORT3: return VK_FORMAT_R16G16B16_UINT;
        case ElementType::HALF3: return VK_FORMAT_R16G16B16_SFLOAT;
        case ElementType::FLOAT3: return VK_FORMAT_R32G32B32_SFLOAT;
        case ElementType::BYTE4: return integer ? VK_FORMAT_R8G8B8A8_SINT : VK_FORMAT_R8G8B8A8_SSCALED;
        case ElementType::UBYTE4: return integer ? VK_FORMAT_R8G8B8A8_UINT : VK_FORMAT_R8G8B8A8_USCALED;
        case ElementType::SHORT4: return integer ? VK_FORMAT_R16G16B16A16_SINT : VK_FORMAT_R16G16B16A16_SSCALED;
        case ElementType::USHORT4: return integer ? VK_FORMAT_R16G16B16A16_UINT : VK_FORMAT_R16G16B16A16_USCALED;
        case ElementType::HALF4: return VK_FORMAT_R16G16B16A16_SFLOAT;
        case ElementType::FLOAT4: return VK_FORMAT_R32G32B32A32_SFLOAT;
    }
    return VK_FORMAT_UNDEFINED;
}

Handle<HwUniformBuffer> VulkanDriver::createUniformBuffer(uint32_t size) {
    Handle<HwUniformBuffer> handle = alloc_handle<VulkanUniformBuffer, HwUniformBuffer>();
    construct_handle<VulkanUniformBuffer>(handle, size);
    handle_cast<VulkanUniformBuffer>(handle)->allocate(context);
    return handle;
}

void VulkanDriver::draw(PipelineState pipeline, PrimitiveHandle handle) {
    VulkanPrimitive* prim = handle_cast<VulkanPrimitive>(handle);
    
    const VkCommandBuffer cmdbuffer = context.commands.get();

    const uint32_t bufferCount = prim->vertex->attributeCount;
    VkBuffer buffers[MAX_VERTEX_ATTRIBUTE_COUNT] = {};
    VkDeviceSize offsets[MAX_VERTEX_ATTRIBUTE_COUNT] = {};
    std::vector<VkVertexInputAttributeDescription> attributes(bufferCount);
    std::vector<VkVertexInputBindingDescription> bindings(bufferCount);

    for (uint32_t i = 0; i < bufferCount; ++i) {
        Attribute attrib = prim->vertex->attributes[i];
        buffers[i] =  prim->vertex->buffers[attrib.index]->buffer;
        offsets[i] = attrib.offset;
        attributes[i] = {
            .location = i,
            .binding = i,
            .format = getVkFormat(attrib.type, attrib.flags & Attribute::FLAG_NORMALIZED, false)
        };
        bindings[i] = {
            .binding = i,
            .stride = attrib.stride,
           
        };
    }

    VulkanProgram* program = handle_cast<VulkanProgram>(pipeline.program);

    Viewport vp = {
        .left=0,
        .bottom=0,
        .width = surface.extent.width,
        .height = surface.extent.height
    };
    VulkanPipelineKey key = {
        .attributes = attributes,
        .bindings = bindings,
        .program = program,
        .viewport = vp
    };
    
    VkPipeline pl = pipelineCache.getOrCreatePipeline(context, key);
    vkCmdBindPipeline(cmdbuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pl);
    vkCmdBindVertexBuffers(cmdbuffer, 0, bufferCount, buffers, offsets);
    vkCmdBindIndexBuffer(cmdbuffer, prim->index->buffer->buffer, 0, VK_INDEX_TYPE_UINT16);
    vkCmdDraw(cmdbuffer, prim->index->count, 1, 0, 0);
}

void VulkanDriver::endRenderPass(int dummy) {
    const VkCommandBuffer cmdbuffer = context.commands.get();

    vkCmdEndRenderPass(cmdbuffer);
}

void VulkanDriver::commit(int dummy) {
    VkFence fence = context.commands.fence();
    vkWaitForFences(context.device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkResetFences(context.device, 1, &fence);
    uint32_t imageIndex = surface.swapContextIndex;
    const VkCommandBuffer cmdbuffer = context.commands.get();
    
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    VkImageMemoryBarrier  barrier = {
       .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
       .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
       .dstAccessMask = 0,
       .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
       .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
       .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
       .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
       .image = context.currentSwapContext->attachment.image,
       .subresourceRange = {
           .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
           .levelCount = 1,
           .layerCount = 1,
       },
    };
    vkCmdPipelineBarrier( context.commands.get(),
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    vkEndCommandBuffer(cmdbuffer);

    VkSemaphore waitSemaphores[] = { context.commands.imageAvailableSemaphore()};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdbuffer;
    VkSemaphore signalSemaphores[] = {context.commands.renderFinishedSemaphore()};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;
    if (vkQueueSubmit(context.graphicsQueue, 1, &submitInfo, context.commands.fence()) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
    }
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    VkSwapchainKHR swapChains[] = {surface.swapChain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr;
    vkQueuePresentKHR(surface.presentQueue, &presentInfo);
    context.commands.next();
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
