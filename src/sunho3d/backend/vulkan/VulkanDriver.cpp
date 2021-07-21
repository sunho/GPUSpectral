#include "VulkanDriver.h"

#include <GLFW/glfw3.h>

#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

using namespace sunho3d;

static void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                          VkDebugUtilsMessengerEXT debugMessenger,
                                          const VkAllocationCallbacks *pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

static VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
                                             const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
                                             const VkAllocationCallbacks *pAllocator,
                                             VkDebugUtilsMessengerEXT *pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

VulkanDriver::VulkanDriver(Window *window) {
    initContext(context);
    initSurfaceContext(context, surface, window);
    pickPhysicalDevice(context, surface);
    createLogicalDevice(context, surface);
    createSwapChain(context, surface, window);
    populateSwapContexts(context, surface);
    setupDebugMessenger();
    pipelineCache.init(context);
    pipelineCache.setDummyTexture(context.emptyTexture->view);
}

VulkanDriver::~VulkanDriver() {
    destroyContext(context, surface);
    DestroyDebugUtilsMessengerEXT(context.instance, debugMessenger, nullptr);
    vkDestroyInstance(context.instance, nullptr);
}

uint32_t VulkanDriver::acquireCommandBuffer(int dummy) {
    const VkCommandBuffer cmdbuffer = context.commands->get();
    VkFence fence = context.commands->fence();
    vkWaitForFences(context.device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkResetFences(context.device, 1, &fence);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0;                   // Optional
    beginInfo.pInheritanceInfo = nullptr;  // Optional

    if (vkBeginCommandBuffer(cmdbuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
    }
    return context.commands->getIndex();
}

RenderTargetHandle VulkanDriver::createDefaultRenderTarget(int dummy) {
    VulkanAttachment depth;
    depth.image = surface.depthTexture->image;
    depth.format = surface.depthTexture->vkFormat;
    depth.view = surface.depthTexture->view;
    Handle<HwRenderTarget> handle = alloc_handle<VulkanRenderTarget, HwRenderTarget>();
    construct_handle<VulkanRenderTarget>(handle, surface.extent.width, surface.extent.height,
                                         depth);
    return handle;
}

RenderTargetHandle VulkanDriver::createRenderTarget(uint32_t width, uint32_t height, ColorAttachment color, TextureAttachment depth) {
    std::array<VulkanAttachment, ColorAttachment::MAX_MRT_NUM> colorTargets = {};
    for (size_t i = 0; i < color.targetNum; ++i) {
        colorTargets[i].texture = handle_cast<VulkanTexture>(color.colors[i].handle);
    }
    VulkanAttachment depthTarget = {};
    if (depth.handle) {
        depthTarget.texture = handle_cast<VulkanTexture>(depth.handle);
    }
    Handle<HwRenderTarget> handle = alloc_handle<VulkanRenderTarget, HwRenderTarget>();
    construct_handle<VulkanRenderTarget>(handle, width, height,
                                         colorTargets, depthTarget);
    return handle;
}

VertexBufferHandle VulkanDriver::createVertexBuffer(uint32_t bufferCount, uint32_t vertexCount,
                                                    uint8_t attributeCount,
                                                    AttributeArray attributes) {
    Handle<HwVertexBuffer> handle = alloc_handle<VulkanVertexBuffer, HwVertexBuffer>();
    construct_handle<VulkanVertexBuffer>(handle, vertexCount, attributeCount, attributes);
    return handle;
}

IndexBufferHandle VulkanDriver::createIndexBuffer(uint32_t indexCount) {
    Handle<HwIndexBuffer> handle = alloc_handle<VulkanIndexBuffer, HwIndexBuffer>();
    construct_handle<VulkanIndexBuffer>(handle, context, indexCount);
    return handle;
}

ProgramHandle VulkanDriver::createProgram(Program program) {
    Handle<HwProgram> handle = alloc_handle<VulkanProgram, HwProgram>();
    construct_handle<VulkanProgram>(handle, context, program);
    return handle;
}

BufferObjectHandle VulkanDriver::createBufferObject(uint32_t size, BufferUsage usage) {
    Handle<HwBufferObject> handle = alloc_handle<VulkanBufferObject, HwBufferObject>();
    construct_handle<VulkanBufferObject>(handle, context, size, usage);

    return handle;
}

void VulkanDriver::setVertexBuffer(VertexBufferHandle handle, uint32_t index,
                                   BufferObjectHandle bufferObject) {
    handle_cast<VulkanVertexBuffer>(handle)->buffers[index] =
        handle_cast<VulkanBufferObject>(bufferObject);
}

void VulkanDriver::updateIndexBuffer(IndexBufferHandle handle, BufferDescriptor data,
                                     uint32_t offset) {
    handle_cast<VulkanIndexBuffer>(handle)->buffer->upload(data);
}

void VulkanDriver::updateBufferObject(BufferObjectHandle handle, BufferDescriptor data,
                                      uint32_t offset) {
    handle_cast<VulkanBufferObject>(handle)->upload(data);
}

void VulkanDriver::beginRenderPass(RenderTargetHandle renderTarget, RenderPassParams params) {
    const VkCommandBuffer cmdbuffer = context.commands->get();
    if (context.firstPass) {
        vkAcquireNextImageKHR(context.device, surface.swapChain, UINT64_MAX,
                              context.commands->imageAvailableSemaphore(), VK_NULL_HANDLE,
                              &surface.swapContextIndex);
        context.currentSwapContext = &surface.swapContexts[surface.swapContextIndex];
        context.firstPass = false;
    }

    VulkanRenderTarget *rt = handle_cast<VulkanRenderTarget>(renderTarget);

    VkRenderPass renderPass = pipelineCache.getOrCreateRenderPass(context, rt);
    VkFramebuffer frameBuffer = pipelineCache.getOrCreateFrameBuffer(context, renderPass, rt);

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = frameBuffer;
    renderPassInfo.renderArea.offset = { 0, 0 };
    renderPassInfo.renderArea.extent = { .width = rt->width, .height = rt->height };
    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
    clearValues[1].depthStencil = { 1.0f, 0 };
    renderPassInfo.clearValueCount = 2;
    renderPassInfo.pClearValues = clearValues.data();
    vkCmdBeginRenderPass(cmdbuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    if (params.viewport.width == 0 && params.viewport.height == 0) {
        params.viewport.width = rt->width;
        params.viewport.height = rt->height;
    }
    context.viewport = params.viewport;
    context.currentRenderPass = renderPass;
}

PrimitiveHandle VulkanDriver::createPrimitive(PrimitiveMode mode) {
    Handle<HwPrimitive> handle = alloc_handle<VulkanPrimitive, HwPrimitive>();
    construct_handle<VulkanPrimitive>(handle, mode);
    return handle;
}

TextureHandle VulkanDriver::createTexture(SamplerType type, TextureUsage usage,
                                          TextureFormat format, uint32_t width, uint32_t height) {
    Handle<HwTexture> handle = alloc_handle<VulkanTexture, HwTexture>();
    construct_handle<VulkanTexture>(handle, context, type, usage, 1, format, width, height);
    return handle;
}

void VulkanDriver::updateTexture(TextureHandle handle, BufferDescriptor data) {
    handle_cast<VulkanTexture>(handle)->update2DImage(context, data);
}

void VulkanDriver::setPrimitiveBuffer(PrimitiveHandle handle, VertexBufferHandle vertexBuffer,
                                      IndexBufferHandle indexBuffer) {
    VulkanVertexBuffer *vertex = handle_cast<VulkanVertexBuffer>(vertexBuffer);
    VulkanIndexBuffer *index = handle_cast<VulkanIndexBuffer>(indexBuffer);
    handle_cast<VulkanPrimitive>(handle)->index = index;
    handle_cast<VulkanPrimitive>(handle)->vertex = vertex;
}

void VulkanDriver::updateUniformBuffer(UniformBufferHandle handle, BufferDescriptor data,
                                       uint32_t offset) {
    handle_cast<VulkanUniformBuffer>(handle)->buffer->upload(data);
}

void VulkanDriver::bindUniformBuffer(uint32_t binding, UniformBufferHandle handle) {
    const VkCommandBuffer cmdbuffer = context.commands->get();

    VulkanUniformBuffer *ubo = handle_cast<VulkanUniformBuffer>(handle);
    currentBinding.uniformBuffers[binding] = ubo->buffer->buffer;
    currentBinding.uniformBufferOffsets[binding] = 0;
    currentBinding.uniformBufferSizes[binding] = ubo->size;

    pipelineCache.bindDescriptor(context, currentBinding);
}

void VulkanDriver::bindTexture(uint32_t binding, TextureHandle handle) {
    const VkCommandBuffer cmdbuffer = context.commands->get();

    VulkanTexture *tex = handle_cast<VulkanTexture>(handle);
    VkDescriptorImageInfo info;
    info.imageLayout = tex->imageLayout;
    info.imageView = tex->view;
    info.sampler = tex->sampler;
    currentBinding.samplers[binding] = info;

    pipelineCache.bindDescriptor(context, currentBinding);
}

Handle<HwUniformBuffer> VulkanDriver::createUniformBuffer(size_t size) {
    Handle<HwUniformBuffer> handle = alloc_handle<VulkanUniformBuffer, HwUniformBuffer>();
    construct_handle<VulkanUniformBuffer>(handle, context, size);
    return handle;
}

void VulkanDriver::draw(PipelineState pipeline, PrimitiveHandle handle) {
    VulkanPrimitive *prim = handle_cast<VulkanPrimitive>(handle);
    const VkCommandBuffer cmdbuffer = context.commands->get();

    const uint32_t bufferCount = prim->vertex->attributeCount;
    VkBuffer buffers[MAX_VERTEX_ATTRIBUTE_COUNT] = {};
    VkDeviceSize offsets[MAX_VERTEX_ATTRIBUTE_COUNT] = {};

    for (uint32_t i = 0; i < bufferCount; ++i) {
        Attribute attrib = prim->vertex->attributes[i];
        buffers[i] = prim->vertex->buffers[attrib.index]->buffer;
        offsets[i] = attrib.offset;
    }

    VulkanProgram *program = handle_cast<VulkanProgram>(pipeline.program);

    VulkanPipelineKey key = {
        .attributes = prim->vertex->attributes, .attributeCount = prim->vertex->attributeCount, .program = program, .viewport = context.viewport, .renderPass = context.currentRenderPass
    };

    VkPipeline pl = pipelineCache.getOrCreatePipeline(context, key);
    vkCmdBindPipeline(cmdbuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pl);
    vkCmdBindVertexBuffers(cmdbuffer, 0, bufferCount, buffers, offsets);
    vkCmdBindIndexBuffer(cmdbuffer, prim->index->buffer->buffer, 0, VK_INDEX_TYPE_UINT16);
    vkCmdDrawIndexed(cmdbuffer, prim->index->count, 1, 0, 0, 0);
}

void VulkanDriver::destroyVertexBuffer(VertexBufferHandle handle) {
    destruct_handle<VulkanVertexBuffer>(handle);
}

void VulkanDriver::destroyIndexBuffer(IndexBufferHandle handle) {
    destruct_handle<VulkanIndexBuffer>(handle);
}

void VulkanDriver::destroyBufferObject(BufferObjectHandle handle) {
    destruct_handle<VulkanBufferObject>(handle);
}

void VulkanDriver::destroyPrimitive(PrimitiveHandle handle) {
    destruct_handle<VulkanPrimitive>(handle);
}

void VulkanDriver::destroyUniformBuffer(UniformBufferHandle handle) {
    destruct_handle<VulkanUniformBuffer>(handle);
}

void VulkanDriver::destroyTexture(TextureHandle handle) {
    destruct_handle<VulkanTexture>(handle);
}

void VulkanDriver::destroyRenderTarget(RenderTargetHandle handle) {
    destruct_handle<VulkanRenderTarget>(handle);
}

void VulkanDriver::endRenderPass(int dummy) {
    const VkCommandBuffer cmdbuffer = context.commands->get();
    vkCmdEndRenderPass(cmdbuffer);
    currentBinding = {};
}

void VulkanDriver::dispatch(int dummy) {

}

uint32_t VulkanDriver::commit(int dummy) {
    uint32_t imageIndex = surface.swapContextIndex;
    const VkCommandBuffer cmdbuffer = context.commands->get();

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    vkEndCommandBuffer(cmdbuffer);

    VkSemaphore waitSemaphores[] = { context.commands->imageAvailableSemaphore() };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdbuffer;
    VkSemaphore signalSemaphores[] = { context.commands->renderFinishedSemaphore() };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;
    if (vkQueueSubmit(context.graphicsQueue, 1, &submitInfo, context.commands->fence()) !=
        VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
    }
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    VkSwapchainKHR swapChains[] = { surface.swapChain };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr;
    vkQueuePresentKHR(surface.presentQueue, &presentInfo);
    pipelineCache.tick();
    context.firstPass = true;
    return context.commands->next();
}

VkBool32 VulkanDriver::debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                     VkDebugUtilsMessageTypeFlagsEXT messageType,
                                     const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
                                     void *pUserData) {
    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    }
    return VK_FALSE;
}

void VulkanDriver::setupDebugMessenger() {
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = VulkanDriver::debugCallback;
    createInfo.pUserData = nullptr;
    if (CreateDebugUtilsMessengerEXT(context.instance, &createInfo, nullptr, &debugMessenger) !=
        VK_SUCCESS) {
        throw std::runtime_error("failed to set up debug messenger!");
    }
}
