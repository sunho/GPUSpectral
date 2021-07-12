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

void VulkanDriver::beginRenderPass(RenderTargetHandle renderTarget, RenderPassParams params) {
    vkAcquireNextImageKHR(context.device, surface.swapChain, UINT64_MAX, context.commands.imageAvailableSemaphore(), VK_NULL_HANDLE, &surface.swapContextIndex);
    context.currentSwapContext = &surface.swapContexts[surface.swapContextIndex];
    
    const VkCommandBuffer cmdbuffer = context.commands.get();
    VulkanRenderTarget* rt = handle_cast<VulkanRenderTarget>(renderTarget);
    
    VkRenderPass renderPass = pipelineCache.createRenderPass(context, rt);
    VkFramebuffer frameBuffer = pipelineCache.createFrameBuffer(context, renderPass, rt);
    
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
    context.currentRenderPass = renderPass;
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
    
    VkPipeline pl = pipelineCache.createPipeline(context, key);
    vkCmdBindPipeline(cmdbuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pl);
    vkCmdBindVertexBuffers(cmdbuffer, 0, bufferCount, buffers, offsets);
    vkCmdBindIndexBuffer(cmdbuffer, prim->index->buffer->buffer, 0, VK_INDEX_TYPE_UINT16);
    vkCmdDraw(cmdbuffer, prim->vertex->vertexCount, 1, 0, 0);
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

       VkImageMemoryBarrier barrier {
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
        vkCmdPipelineBarrier(cmdbuffer,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
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
