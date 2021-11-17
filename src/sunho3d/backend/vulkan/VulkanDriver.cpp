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
    device = std::make_unique<VulkanDevice>(window);
    rayTracer = std::make_unique<VulkanRayTracer>(*device);
    //setupDebugMessenger();
}

VulkanDriver::~VulkanDriver() {
    DestroyDebugUtilsMessengerEXT(device->instance, debugMessenger, nullptr);
}

InflightHandle VulkanDriver::beginFrame(FenceHandle handle) {
    VulkanFence* fence = handle_cast<VulkanFence>(handle);
    Handle<HwInflight> inflightHandle = alloc_handle<VulkanInflight, HwInflight>();
    construct_handle<VulkanInflight>(inflightHandle, *device, *rayTracer);
    auto inflight = handle_cast<VulkanInflight>(inflightHandle);
    context.inflight = inflight;
    inflight->inflightFence = fence->fence;
    if (!tracyContext) {
        tracyContext = TracyVkContext(device->physicalDevice, device->device, device->graphicsQueue, inflight->cmd);
    }
    
    inflight->cmd.begin(vk::CommandBufferBeginInfo());
    
    device->wsi->beginFrame(inflight->imageSemaphore);
    return inflightHandle;
}

void VulkanDriver::endFrame(int) {
    context.inflight->cmd.end();
    TracyVkCollect(tracyContext, context.inflight->cmd)
    //TracyVkDestroy(tracyContext)
    std::array<vk::PipelineStageFlags, 1> waitStages = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
    std::array<vk::Semaphore, 1> waitSemaphores = { context.inflight->imageSemaphore };
    std::array<vk::Semaphore, 1> signalSemaphores = { context.inflight->renderSemaphore };
    auto info = vk::SubmitInfo()
        .setWaitSemaphores(waitSemaphores)
        .setSignalSemaphores(signalSemaphores)
        .setWaitDstStageMask(waitStages)
        .setPCommandBuffers(&context.inflight->cmd)
        .setCommandBufferCount(1);
    device->graphicsQueue.submit(1, &info, context.inflight->inflightFence);
    device->wsi->endFrame(context.inflight->renderSemaphore);
    device->cache->tick();
}

void VulkanDriver::releaseInflight(InflightHandle handle) {
    destruct_handle<VulkanInflight>(handle);
}

RenderTargetHandle VulkanDriver::createDefaultRenderTarget(int dummy) {
    Handle<HwRenderTarget> handle = alloc_handle<VulkanRenderTarget, HwRenderTarget>();
    construct_handle<VulkanRenderTarget>(handle);
    return handle;
}

RenderTargetHandle VulkanDriver::createRenderTarget(uint32_t width, uint32_t height, RenderAttachments renderAttachments) {
    VulkanAttachments attachments = {};
    std::array<VulkanAttachment, RenderAttachments::MAX_MRT_NUM> colorTargets = {};
    for (size_t i = 0; i < RenderAttachments::MAX_MRT_NUM; ++i) {
        if (renderAttachments.colors[i]) {
            attachments.colors[i] = VulkanAttachment(handle_cast<VulkanTexture>(renderAttachments.colors[i]));
        }
    }
    if (renderAttachments.depth) {
        attachments.depth = VulkanAttachment(handle_cast<VulkanTexture>(renderAttachments.depth));
    }
    Handle<HwRenderTarget> handle = alloc_handle<VulkanRenderTarget, HwRenderTarget>();
    construct_handle<VulkanRenderTarget>(handle, width, height, attachments);
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
    construct_handle<VulkanIndexBuffer>(handle, *device, indexCount);
    return handle;
}

ProgramHandle VulkanDriver::createProgram(Program program) {
    Handle<HwProgram> handle = alloc_handle<VulkanProgram, HwProgram>();
    construct_handle<VulkanProgram>(handle, *device, program);
    return handle;
}

BufferObjectHandle VulkanDriver::createBufferObject(uint32_t size, BufferUsage usage) {
    Handle<HwBufferObject> handle = alloc_handle<VulkanBufferObject, HwBufferObject>();
    construct_handle<VulkanBufferObject>(handle, *device, size, usage);

    return handle;
}

FenceHandle VulkanDriver::createFence(int) {
    Handle<HwFence> handle = alloc_handle<VulkanFence, HwFence>();
    construct_handle<VulkanFence>(handle, *device);

    return handle;
}

void VulkanDriver::waitFence(FenceHandle handle) {
    std::array<vk::Fence, 1> fences = {handle_cast<VulkanFence>(handle)->fence};
    device->device.waitForFences(fences, true, UINT64_MAX); 
    device->device.resetFences(fences);
}

void VulkanDriver::setVertexBuffer(VertexBufferHandle handle, uint32_t index,
                                   BufferObjectHandle bufferObject) {
    handle_cast<VulkanVertexBuffer>(handle)->buffers[index] =
        handle_cast<VulkanBufferObject>(bufferObject);
}

void VulkanDriver::updateIndexBuffer(IndexBufferHandle handle, BufferDescriptor data,
                                     uint32_t offset) {
    handle_cast<VulkanIndexBuffer>(handle)->buffer->uploadSync(data);

}

void VulkanDriver::updateStagingBufferObject(BufferObjectHandle handle, BufferDescriptor data,
                                      uint32_t offset) {
    handle_cast<VulkanBufferObject>(handle)->upload(data);
}

void VulkanDriver::updateBufferObjectSync(BufferObjectHandle handle, BufferDescriptor data,
                                             uint32_t offset) {
    handle_cast<VulkanBufferObject>(handle)->uploadSync(data);
}

Extent2D VulkanDriver::getFrameSize(int dummy) {
    auto e = device->wsi->getExtent();
    return {e.width, e.height};
}

void VulkanDriver::beginRenderPass(RenderTargetHandle renderTarget, RenderPassParams params) {
    auto& cmd = context.inflight->cmd;
    VulkanRenderTarget *rt = handle_cast<VulkanRenderTarget>(renderTarget);

    vk::RenderPass renderPass = device->cache->getOrCreateRenderPass(device->wsi->currentSwapChain(), rt);
    vk::Framebuffer frameBuffer = device->cache->getOrCreateFrameBuffer(renderPass, device->wsi->currentSwapChain(), rt);

    vk::RenderPassBeginInfo renderPassInfo{};
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = frameBuffer;
    renderPassInfo.renderArea.offset = { .x = 0, .y= 0 };
    renderPassInfo.renderArea.extent = rt->getExtent(*device);
    std::vector<vk::ClearValue> clearValues{};
    for (size_t i = 0; i < rt->attachmentCount; ++i) {
        clearValues.push_back(vk::ClearColorValue(std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f })); 
    }
    if (rt->attachments.depth.valid) {
        clearValues.push_back(vk::ClearDepthStencilValue().setDepth(1.0f));
    }
    
    renderPassInfo.clearValueCount = clearValues.size();
    renderPassInfo.pClearValues = clearValues.data();
    cmd.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
    if (params.viewport.width == 0 && params.viewport.height == 0) {
        auto extent = rt->getExtent(*device);
        params.viewport.width = extent.width;
        params.viewport.height = extent.height;
    }
    context.viewport = params.viewport;
    context.currentRenderTarget = rt;
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
    construct_handle<VulkanTexture>(handle, *device, type, usage, 1, format, width, height);
    return handle;
}

void VulkanDriver::updateTexture(TextureHandle handle, BufferDescriptor data) {
    handle_cast<VulkanTexture>(handle)->update2DImage(*device, data);
}

void VulkanDriver::setPrimitiveBuffer(PrimitiveHandle handle, VertexBufferHandle vertexBuffer,
                                      IndexBufferHandle indexBuffer) {
    VulkanVertexBuffer *vertex = handle_cast<VulkanVertexBuffer>(vertexBuffer);
    VulkanIndexBuffer *index = handle_cast<VulkanIndexBuffer>(indexBuffer);
    handle_cast<VulkanPrimitive>(handle)->index = index;
    handle_cast<VulkanPrimitive>(handle)->vertex = vertex;
}

void VulkanDriver::draw(PipelineState pipeline, PrimitiveHandle handle) {
    ZoneScopedN("Draw")
    VulkanPrimitive *prim = handle_cast<VulkanPrimitive>(handle);
    auto& cmd = context.inflight->cmd;
    std::string zonename = profileZoneName + " draw";
    TracyVkZoneTransient(tracyContext, vkzone, cmd, zonename.c_str(), true)
   
    const uint32_t bufferCount = prim->vertex->attributeCount;
    vk::Buffer buffers[MAX_VERTEX_ATTRIBUTE_COUNT] = {};
    vk::DeviceSize offsets[MAX_VERTEX_ATTRIBUTE_COUNT] = {};

    for (uint32_t i = 0; i < bufferCount; ++i) {
        Attribute attrib = prim->vertex->attributes[i];
        buffers[i] = prim->vertex->buffers[attrib.index]->buffer;
        offsets[i] = attrib.offset;
    }

    VulkanProgram *program = handle_cast<VulkanProgram>(pipeline.program);

    VulkanPipelineState state = {
        .attributes = prim->vertex->attributes,
        .attributeCount = prim->vertex->attributeCount,
        .program = program,
        .viewport = context.viewport,
        .renderPass = context.currentRenderPass,
        .attachmentCount = context.currentRenderTarget->attachmentCount,
        .depthTest = pipeline.depthTest
    };

    VulkanPipeline vkpipe = device->cache->getOrCreateGraphicsPipeline(state);
    device->cache->bindDescriptor(cmd, *program, translateBindingMap(pipeline.bindings));
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, vkpipe.pipeline);
    if (!pipeline.pushConstants.empty()) {
        cmd.pushConstants(vkpipe.layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute, 0, pipeline.pushConstants.size(), pipeline.pushConstants.data()); 
    } 
    cmd.bindVertexBuffers(0, bufferCount, buffers, offsets);
    cmd.bindIndexBuffer(prim->index->buffer->buffer, 0, vk::IndexType::eUint32);
    cmd.drawIndexed(prim->index->count, 1, 0, 0, 0);
}

void VulkanDriver::setProfileZoneName(const char* name) {
    profileZoneName = name;
}

void VulkanDriver::dispatch(PipelineState pipeline, size_t groupCountX, size_t groupCountY, size_t groupCountZ) {
    ZoneScopedN("Dispatch")
    auto& cmd = context.inflight->cmd;
    std::string zonename = (profileZoneName + " dispatch");
    TracyVkZoneTransient(tracyContext, vkzone, cmd, zonename.c_str(), true)
    VulkanProgram* program = handle_cast<VulkanProgram>(pipeline.program);
    VulkanPipeline vkpipe = device->cache->getOrCreateComputePipeline(*program);
    device->cache->bindDescriptor(cmd, *program, translateBindingMap(pipeline.bindings));
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, vkpipe.pipeline);
    if (!pipeline.pushConstants.empty()) {
        ZoneScopedN("Push constants");
        cmd.pushConstants(vkpipe.layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute, 0, pipeline.pushConstants.size(), pipeline.pushConstants.data());
    }
    cmd.dispatch(groupCountX, groupCountY, groupCountZ);
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

void VulkanDriver::destroyTexture(TextureHandle handle) {
    destruct_handle<VulkanTexture>(handle);
}

void VulkanDriver::destroyRenderTarget(RenderTargetHandle handle) {
    destruct_handle<VulkanRenderTarget>(handle);
}

void VulkanDriver::endRenderPass(int dummy) {
    auto& cmd = context.inflight->cmd;
    vkCmdEndRenderPass(cmd);
}

Handle<HwBLAS> VulkanDriver::createBLAS(int dummy) {
    Handle<HwBLAS> handle = alloc_handle<VulkanBLAS, HwBLAS>();
    construct_handle<VulkanBLAS>(handle, *rayTracer);
    return handle;
}

Handle<HwTLAS> VulkanDriver::createTLAS(int dummy) {
    Handle<HwTLAS> handle = alloc_handle<VulkanTLAS, HwTLAS>();
    construct_handle<VulkanTLAS>(handle, *rayTracer);
    return handle;
}

void VulkanDriver::buildBLAS(Handle<HwBLAS> handle, Handle<HwPrimitive> primitiveHandle) {
    VulkanBLAS* blas = handle_cast<VulkanBLAS>(handle);
    VulkanPrimitive* primitive = handle_cast<VulkanPrimitive>(primitiveHandle);
    auto& cmd = context.inflight->cmd;
    std::string zonename = profileZoneName + " build blas";
    TracyVkZoneTransient(tracyContext, vkzone, cmd, zonename.c_str(), true)
    rayTracer->buildBLAS(context.inflight->rayFrameContext, blas, primitive);
}

void VulkanDriver::copyBufferObject(Handle<HwBufferObject> destHandle, Handle<HwBufferObject> srcHandle) {
    auto& cmd = context.inflight->cmd;
    VulkanBufferObject* dest = handle_cast<VulkanBufferObject>(destHandle);
    VulkanBufferObject* src = handle_cast<VulkanBufferObject>(srcHandle);
    dest->copy(cmd, *src);
}

void VulkanDriver::buildTLAS(Handle<HwTLAS> handle, RTSceneDescriptor descriptor) {
    VulkanTLAS* tlas = handle_cast<VulkanTLAS>(handle);
    VulkanRTSceneDescriptor desc = {};
    auto& cmd = context.inflight->cmd;
    std::string zonename = profileZoneName + " build tlas";
    TracyVkZoneTransient(tracyContext, vkzone, cmd, zonename.c_str(), true)
    for (size_t i = 0; i < descriptor.count; ++i) {
        VulkanRTInstance instance = {};
        instance.blas = handle_cast<VulkanBLAS>(descriptor.instances[i].blas);
        instance.transfom = descriptor.instances[i].transfom;
        desc.instances.push_back(instance);
    }
    rayTracer->buildTLAS(context.inflight->rayFrameContext, tlas, desc);
}

ImageLayout VulkanDriver::getTextureImageLayout(Handle<HwTexture> handle) {
    auto texture = handle_cast<VulkanTexture, HwTexture>(handle);
    return texture->imageLayout;
}

void VulkanDriver::intersectRays(Handle<HwTLAS> tlasHandle, uint32_t rayCount, Handle<HwBufferObject> raysHandle, Handle<HwBufferObject> hitsHandle) {
    VulkanTLAS* tlas = handle_cast<VulkanTLAS>(tlasHandle);
    VulkanBufferObject* rays = handle_cast<VulkanBufferObject>(raysHandle);
    VulkanBufferObject* hits = handle_cast<VulkanBufferObject>(hitsHandle);
    auto& cmd = context.inflight->cmd;
    ZoneScopedN("Intersect")
    std::string zonename = profileZoneName + " intersect";
    TracyVkZoneTransient(tracyContext, vkzone, cmd, zonename.c_str(), true)
    rayTracer->intersectRays(context.inflight->rayFrameContext, tlas, rayCount, rays, hits);
}

void VulkanDriver::destroyBLAS(Handle<HwBLAS> handle) {
    destruct_handle<VulkanBLAS>(handle);
}

void VulkanDriver::destroyTLAS(Handle<HwTLAS> handle) {
    destruct_handle<VulkanTLAS>(handle);
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
    if (CreateDebugUtilsMessengerEXT(device->instance, &createInfo, nullptr, &debugMessenger) !=
        VK_SUCCESS) {
        throw std::runtime_error("failed to set up debug messenger!");
    }
}

VulkanBindings VulkanDriver::translateBindingMap(const BindingMap & binds) {
    ZoneScopedN("Translate binding map")
    VulkanBindings bindings;
    for (auto [key, binding] : binds) {
        VulkanBinding vb = {};
        vb.binding = key.binding;
        vb.arraySize = 0;
        vb.set = key.set;
        vb.type = translateDescriptorType(binding.type);
        switch (binding.type) {
            case ProgramParameterType::UNIFORM: {
                for (auto handle : binding.handles) {
                    if (!handle.texture && !handle.buffer) {
                        break;
                    }
                    auto ub = handle_cast<VulkanBufferObject>(handle.buffer);
                    vk::DescriptorBufferInfo bufferInfo{};
                    bufferInfo.offset = 0;
                    bufferInfo.buffer = ub->buffer;
                    bufferInfo.range = ub->size;
                    vb.bufferInfo.push_back(bufferInfo);
                    ++vb.arraySize;
                }
                break;
            }
            case ProgramParameterType::STORAGE: {
                for (auto handle : binding.handles) {
                    if (!handle.texture && !handle.buffer) {
                        break;
                    }
                    auto sb = handle_cast<VulkanBufferObject>(handle.buffer);
                    vk::DescriptorBufferInfo bufferInfo{};
                    bufferInfo.offset = 0;
                    bufferInfo.buffer = sb->buffer;
                    bufferInfo.range = sb->size;
                    vb.bufferInfo.push_back(bufferInfo);
                    ++vb.arraySize;
                }
                break;
            }
            case ProgramParameterType::IMAGE:
            case ProgramParameterType::TEXTURE: {
                for (auto handle : binding.handles) {
                    if (!handle.texture && !handle.buffer) {
                        break;
                    }
                    auto tex = handle_cast<VulkanTexture>(handle.texture);
                    vk::DescriptorImageInfo imageInfo{};
                    imageInfo.imageLayout = tex->vkImageLayout;
                    imageInfo.imageView = tex->view;
                    imageInfo.sampler = tex->sampler;
                    vb.imageInfo.push_back(imageInfo);
                    ++vb.arraySize;
                }
                break;
            }
        }
        bindings.push_back(vb);
    }

    return bindings;
}

void VulkanDriver::setBarrier(Barrier barrier) {
    auto& cmd = context.inflight->cmd;
    if (barrier.image) {
        auto texture = handle_cast<VulkanTexture, HwTexture>(barrier.image);
        vk::ImageMemoryBarrier imageBarrier{};
        imageBarrier.srcAccessMask = translateAccessMask(barrier.srcAccess);
        imageBarrier.dstAccessMask = translateAccessMask(barrier.dstAccess);
        imageBarrier.oldLayout = translateImageLayout(barrier.initialLayout);
        imageBarrier.newLayout = translateImageLayout(barrier.finalLayout);
        imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imageBarrier.subresourceRange.aspectMask = decideAsepctFlags(barrier.srcAccess) | decideAsepctFlags(barrier.dstAccess);
        imageBarrier.subresourceRange.baseArrayLayer = 0;
        imageBarrier.subresourceRange.layerCount = 1;
        imageBarrier.subresourceRange.baseMipLevel = 0;
        imageBarrier.subresourceRange.levelCount= 1;
        imageBarrier.image = texture->image;
        texture->vkImageLayout = translateImageLayout(barrier.finalLayout);
        texture->imageLayout = barrier.finalLayout;

        cmd.pipelineBarrier(translateStageMask(barrier.srcStage),
                            translateStageMask(barrier.dstStage), vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &imageBarrier);
    } else if (barrier.srcAccess != BarrierAccessFlag::NONE || barrier.dstAccess != BarrierAccessFlag::NONE){
        vk::MemoryBarrier memBarrier{};
        memBarrier.srcAccessMask = translateAccessMask(barrier.srcAccess);
        memBarrier.dstAccessMask = translateAccessMask(barrier.dstAccess);

        cmd.pipelineBarrier(translateStageMask(barrier.srcStage),
                            translateStageMask(barrier.dstStage), vk::DependencyFlags(), 1, &memBarrier, 0, nullptr, 0, nullptr);
    } else {
        cmd.pipelineBarrier(translateStageMask(barrier.srcStage),
                            translateStageMask(barrier.dstStage), vk::DependencyFlags(), 0, nullptr, 0, nullptr, 0, nullptr);
    
    }
}