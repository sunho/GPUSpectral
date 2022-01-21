#include "VulkanDriver.h"

#include <fmt/format.h>
#include <GLFW/glfw3.h>
#include <GPUSpectral/utils/Log.h>
#include <GPUSpectral/utils/Util.h>

#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>
#include <filesystem>
#include <fstream>

using namespace GPUSpectral;

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

VulkanDriver::VulkanDriver(Window *window, const std::filesystem::path& basePath) : basePath(basePath) {
    device = std::make_unique<VulkanDevice>(window);
    dummyTex = std::make_unique<VulkanTexture>(*device, SamplerType::SAMPLER2D, TextureUsage::UPLOADABLE | TextureUsage::STORAGE | TextureUsage::COLOR_ATTACHMENT, 1, TextureFormat::RGBA8, 1, 1, 1);
    uint32_t zero = 0;
    dummyTex->copyInitialData({ .data = &zero }, ImageLayout::GENERAL);
    dummyBuf = std::make_unique<VulkanBufferObject>(*device, 1, BufferUsage::UNIFORM | BufferUsage::STORAGE);
    setupDebugMessenger();
}

VulkanDriver::~VulkanDriver() {
    DestroyDebugUtilsMessengerEXT(device->instance, debugMessenger, nullptr);
    TracyVkDestroy(context.tracyContext)
}

InflightHandle VulkanDriver::beginFrame(FenceHandle handle) {
    VulkanFence* fence = handleCast<VulkanFence>(handle);
    Handle<HwInflight> inflightHandle = allocHandle<VulkanInflight, HwInflight>();
    constructHandle<VulkanInflight>(inflightHandle, *device);
    auto inflight = handleCast<VulkanInflight>(inflightHandle);
    context.inflight = inflight;
    inflight->inflightFence = fence->fence;
    if (!context.tracyContext) {
        context.tracyContext = TracyVkContext(device->physicalDevice, device->device, device->graphicsQueue, inflight->cmd);
    }
    
    inflight->cmd.begin(vk::CommandBufferBeginInfo());
    
    device->wsi->beginFrame(inflight->imageSemaphore);
    return inflightHandle;
}

void VulkanDriver::endFrame(int) {
    context.inflight->cmd.end();
    TracyVkCollect(context.tracyContext, context.inflight->cmd)
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
    destructHandle<VulkanInflight>(handle);
}

RenderTargetHandle VulkanDriver::createDefaultRenderTarget(int dummy) {
    Handle<HwRenderTarget> handle = allocHandle<VulkanRenderTarget, HwRenderTarget>();
    constructHandle<VulkanRenderTarget>(handle);
    return handle;
}

RenderTargetHandle VulkanDriver::createRenderTarget(uint32_t width, uint32_t height, RenderAttachments renderAttachments) {
    VulkanAttachments attachments = {};
    std::array<VulkanAttachment, RenderAttachments::MAX_MRT_NUM> colorTargets = {};
    for (size_t i = 0; i < RenderAttachments::MAX_MRT_NUM; ++i) {
        if (renderAttachments.colors[i]) {
            attachments.colors[i] = VulkanAttachment(handleCast<VulkanTexture>(renderAttachments.colors[i]));
        }
    }
    if (renderAttachments.depth) {
        attachments.depth = VulkanAttachment(handleCast<VulkanTexture>(renderAttachments.depth));
    }
    Handle<HwRenderTarget> handle = allocHandle<VulkanRenderTarget, HwRenderTarget>();
    constructHandle<VulkanRenderTarget>(handle, width, height, attachments);
    return handle;
}

VertexBufferHandle VulkanDriver::createVertexBuffer(uint32_t bufferCount, uint32_t vertexCount,
                                                    uint8_t attributeCount,
                                                    AttributeArray attributes) {
    Handle<HwVertexBuffer> handle = allocHandle<VulkanVertexBuffer, HwVertexBuffer>();
    constructHandle<VulkanVertexBuffer>(handle, vertexCount, attributeCount, attributes);
    return handle;
}

IndexBufferHandle VulkanDriver::createIndexBuffer(uint32_t indexCount) {
    Handle<HwIndexBuffer> handle = allocHandle<VulkanIndexBuffer, HwIndexBuffer>();
    constructHandle<VulkanIndexBuffer>(handle, *device, indexCount);
    return handle;
}

ProgramHandle VulkanDriver::createProgram(Program program) {
    Handle<HwProgram> handle = allocHandle<VulkanProgram, HwProgram>();
    constructHandle<VulkanProgram>(handle, *device, program);
    return handle;
}

BufferObjectHandle VulkanDriver::createBufferObject(uint32_t size, BufferUsage usage, BufferType type) {
    Handle<HwBufferObject> handle = allocHandle<VulkanBufferObject, HwBufferObject>();
    constructHandle<VulkanBufferObject>(handle, *device, size, usage, type);

    return handle;
}

FenceHandle VulkanDriver::createFence(int) {
    Handle<HwFence> handle = allocHandle<VulkanFence, HwFence>();
    constructHandle<VulkanFence>(handle, *device);

    return handle;
}

void VulkanDriver::waitFence(FenceHandle handle) {
    std::array<vk::Fence, 1> fences = {handleCast<VulkanFence>(handle)->fence};
    device->device.waitForFences(fences, true, UINT64_MAX); 
    device->device.resetFences(fences);
}

void VulkanDriver::destroyFence(FenceHandle handle) {
    destructHandle<VulkanFence>(handle);
}

void VulkanDriver::setVertexBuffer(VertexBufferHandle handle, uint32_t index,
                                   BufferObjectHandle bufferObject) {
    handleCast<VulkanVertexBuffer>(handle)->buffers[index] =
        handleCast<VulkanBufferObject>(bufferObject);
}

void VulkanDriver::updateIndexBuffer(IndexBufferHandle handle, BufferDescriptor data,
                                     uint32_t offset) {
    handleCast<VulkanIndexBuffer>(handle)->buffer->uploadSync(data);

}

void VulkanDriver::updateCPUBufferObject(BufferObjectHandle handle, BufferDescriptor data,
                                      uint32_t offset) {
    auto buffer = handleCast<VulkanBufferObject>(handle);
    if (data.size == 0) {
        data.size = buffer->size;
    }
    memcpy(buffer->mapped, data.data, data.size);
}

void VulkanDriver::updateBufferObjectSync(BufferObjectHandle handle, BufferDescriptor data,
                                             uint32_t offset) {
    handleCast<VulkanBufferObject>(handle)->uploadSync(data);
}

Extent2D VulkanDriver::getFrameSize(int dummy) {
    auto e = device->wsi->getExtent();
    return {e.width, e.height};
}

void VulkanDriver::beginRenderPass(RenderTargetHandle renderTarget, RenderPassParams params) {
    auto& cmd = context.inflight->cmd;
    VulkanRenderTarget *rt = handleCast<VulkanRenderTarget>(renderTarget);
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
    Handle<HwPrimitive> handle = allocHandle<VulkanPrimitive, HwPrimitive>();
    constructHandle<VulkanPrimitive>(handle, mode);
    return handle;
}

TextureHandle VulkanDriver::createTexture(SamplerType type, TextureUsage usage,
                                          TextureFormat format, uint8_t levels, uint32_t width, uint32_t height, uint32_t layers) {
    Handle<HwTexture> handle = allocHandle<VulkanTexture, HwTexture>();
    constructHandle<VulkanTexture>(handle, *device, type, usage, levels, format, width, height, layers);
    return handle;
}

void VulkanDriver::copyTextureInitialData(TextureHandle handle, BufferDescriptor data) {
    handleCast<VulkanTexture>(handle)->copyInitialData(data);
}

void VulkanDriver::copyBufferToTexture(TextureHandle handle, ImageSubresource subresource, BufferObjectHandle bufferHandle) {
    auto& cmd = context.inflight->cmd;
    VulkanTexture* texture = handleCast<VulkanTexture>(handle);
    VulkanBufferObject* buffer = handleCast<VulkanBufferObject>(bufferHandle);
    texture->copyBuffer(cmd, buffer->buffer, texture->width, texture->height, subresource);
}

void VulkanDriver::blitTexture(TextureHandle destHandle, ImageSubresource destSubresource, TextureHandle srcHandle, ImageSubresource srcSubresource) {
    auto& cmd = context.inflight->cmd;
    VulkanTexture* dest = handleCast<VulkanTexture>(destHandle);
    VulkanTexture* src = handleCast<VulkanTexture>(srcHandle);
    dest->blitImage(cmd, *src, dest->width, dest->height, srcSubresource, destSubresource);
}

void VulkanDriver::setPrimitiveBuffer(PrimitiveHandle handle, VertexBufferHandle vertexBuffer,
                                      IndexBufferHandle indexBuffer) {
    VulkanVertexBuffer *vertex = handleCast<VulkanVertexBuffer>(vertexBuffer);
    VulkanIndexBuffer *index = handleCast<VulkanIndexBuffer>(indexBuffer);
    handleCast<VulkanPrimitive>(handle)->index = index;
    handleCast<VulkanPrimitive>(handle)->vertex = vertex;
}

void VulkanDriver::draw(GraphicsPipeline pipeline, PrimitiveHandle handle) {
    auto& cmd = context.inflight->cmd;
    ZoneScopedN("Draw")
    TracyVkZoneTransient(context.tracyContext, vkzone, cmd, profileZoneName("draw").c_str(), true)

    VulkanPrimitive *prim = handleCast<VulkanPrimitive>(handle);
    const uint32_t bufferCount = prim->vertex->attributeCount;
    vk::Buffer buffers[MAX_VERTEX_ATTRIBUTE_COUNT] = {};
    vk::DeviceSize offsets[MAX_VERTEX_ATTRIBUTE_COUNT] = {};

    for (uint32_t i = 0; i < bufferCount; ++i) {
        Attribute attrib = prim->vertex->attributes[i];
        buffers[i] = prim->vertex->buffers[attrib.index]->buffer;
        offsets[i] = attrib.offset;
    }

    VulkanProgram *vertex = handleCast<VulkanProgram>(pipeline.vertex);
    VulkanProgram *fragment = handleCast<VulkanProgram>(pipeline.fragment);
    ProgramParameterLayout parameterLayout = vertex->program.parameterLayout + fragment->program.parameterLayout;

    VulkanPipelineState state = {
        .attributes = prim->vertex->attributes,
        .attributeCount = prim->vertex->attributeCount,
        .vertex = vertex,
        .fragment = fragment,
        .viewport = context.viewport,
        .renderPass = context.currentRenderPass,
        .attachmentCount = context.currentRenderTarget->attachmentCount,
        .depthTest = pipeline.depthTest,
        .parameterLayout = parameterLayout
    };

    VulkanPipeline vkpipe = device->cache->getOrCreateGraphicsPipeline(state);
    device->cache->bindDescriptor(cmd, vk::PipelineBindPoint::eGraphics, parameterLayout, translateBindingMap(parameterLayout, pipeline.bindings));
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, vkpipe.pipeline);
    if (!pipeline.pushConstants.empty()) {
        cmd.pushConstants(vkpipe.layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute, 0, pipeline.pushConstants.size(), pipeline.pushConstants.data()); 
    } 
    cmd.bindVertexBuffers(0, bufferCount, buffers, offsets);
    cmd.bindIndexBuffer(prim->index->buffer->buffer, 0, vk::IndexType::eUint32);
    cmd.drawIndexed(prim->index->count, 1, 0, 0, 0);
}

void VulkanDriver::dispatch(ComputePipeline pipeline, size_t groupCountX, size_t groupCountY, size_t groupCountZ) {
    auto& cmd = context.inflight->cmd;
    ZoneScopedN("Dispatch")
    TracyVkZoneTransient(context.tracyContext, vkzone, cmd, profileZoneName("dispatch").c_str(), true)

    VulkanProgram* program = handleCast<VulkanProgram>(pipeline.program);
    VulkanPipeline vkpipe = device->cache->getOrCreateComputePipeline(*program);
    device->cache->bindDescriptor(cmd, vk::PipelineBindPoint::eCompute, program->program.parameterLayout, translateBindingMap(program->program.parameterLayout, pipeline.bindings));
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, vkpipe.pipeline);
    if (!pipeline.pushConstants.empty()) {
        ZoneScopedN("Push constants");
        cmd.pushConstants(vkpipe.layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute, 0, pipeline.pushConstants.size(), pipeline.pushConstants.data());
    }
    cmd.dispatch(groupCountX, groupCountY, groupCountZ);
}

void VulkanDriver::traceRays(RTPipeline pipeline, size_t width, size_t height) {
    auto& cmd = context.inflight->cmd;
    ProgramParameterLayout parameterLayout = handleCast<VulkanProgram>(pipeline.raygenGroup)->program.parameterLayout;
    const auto unwrapVector = [&](const std::vector<Handle<HwProgram>>& programs) {
        std::vector<VulkanProgram*> outPrograms;
        for (auto program : programs) {
            outPrograms.push_back(handleCast<VulkanProgram>(program));
            parameterLayout = parameterLayout + outPrograms.back()->program.parameterLayout;
        }
        return outPrograms;
    }; 
    VulkanRTPipelineState state = {
        .raygenGroup = handleCast<VulkanProgram>(pipeline.raygenGroup),
        .missGroups = unwrapVector(pipeline.missGroups),
        .hitGroups = unwrapVector(pipeline.hitGroups),
        .callableGroups = unwrapVector(pipeline.callableGroups),
        .parameterLayout = parameterLayout
    };
    auto vkpipe = device->cache->getOrCreateRTPipeline(state);
    auto sbt = device->cache->getOrCreateSBT(state);
    cmd.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, vkpipe.pipeline);
    device->cache->bindDescriptor(cmd, vk::PipelineBindPoint::eRayTracingKHR, parameterLayout, translateBindingMap(parameterLayout, pipeline.bindings));
    device->dld.vkCmdTraceRaysKHR(cmd, 
        &sbt.raygen.stridedDeviceAddressRegion, 
        &sbt.miss.stridedDeviceAddressRegion, 
        &sbt.hit.stridedDeviceAddressRegion, 
        &sbt.callable.stridedDeviceAddressRegion, 
        width, height, 1);
}

uint64_t VulkanDriver::getDeviceAddress(Handle<HwBufferObject> handle) {
    VulkanBufferObject* buffer = handleCast<VulkanBufferObject, HwBufferObject>(handle);
    return device->getBufferDeviceAddress(buffer->buffer);
}

void VulkanDriver::setProfileSectionName(const char* name) {
    context.profileSectionName = name;
}

std::string GPUSpectral::VulkanDriver::profileZoneName(std::string zoneName) {
    return fmt::format("[{}] {}", context.profileSectionName, zoneName);
}

static inline void preprocessShader(std::string& preprocessedSource, const std::string& source, const std::filesystem::path& basePath, uint64_t& hash) {
    auto lines = Split(source, "\n");
    hash ^= hashBuffer<char>(source.data(), source.size());
    unsigned line_index = 1;
    for (auto& line : lines) {
        if (line.find("#include \"") == 0) {
            auto includePath = line.substr(10);
            if (!includePath.empty() && includePath.back() == '"')
                includePath.pop_back();

            auto path = basePath / includePath;
            std::ifstream file(path);
            std::string includedSource((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

            preprocessShader(preprocessedSource, includedSource, basePath, hash);
        } else {
            preprocessedSource += line;
            preprocessedSource += '\n';
        }

        line_index++;
    }
}

/*CompiledCode VulkanDriver::compileCode(const char* path) {
    shaderc::SpvCompilationResult result;
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);

    std::ifstream file(path);
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    auto filePath = std::filesystem::path(path);
    std::filesystem::path shadersFolderPath = filePath.parent_path();

    std::string preprocessedSource;
    uint64_t hash = 0x53FF53530053FFFF;
    preprocessShader(preprocessedSource, content, shadersFolderPath, hash);
    
    std::string cacheName = filePath.filename().string() + std::to_string(hash) + ".spv";
    if (!std::filesystem::is_directory(basePath / "shader_cache")) {
        std::filesystem::create_directory(basePath / "shader_cache");
    }

    auto cachePath = basePath / "shader_cache" / cacheName;
    std::ifstream cacheFile(cachePath, std::ifstream::binary);
    if (cacheFile.is_open()) {
        std::vector<uint8_t> cacheFileContent((std::istreambuf_iterator<char>(cacheFile)), std::istreambuf_iterator<char>());
        CompiledCode compiledCode(cacheFileContent.size() / 4);
        memcpy(compiledCode.data(), cacheFileContent.data(), cacheFileContent.size());
        return compiledCode;
    }

    result = compiler.CompileGlslToSpv(preprocessedSource, shaderc_glsl_infer_from_source, path, options);
    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        std::string message = result.GetErrorMessage();
        Log("Shader compilation error: {}", message);
        throw std::runtime_error("shader compilation error");
    }

    auto compiledCode = CompiledCode(result.begin(), result.end());

    std::ofstream cacheFileWrite;
    cacheFileWrite.open(cachePath, std::ios::out | std::ios::binary);
    cacheFileWrite.write((const char*)compiledCode.data(), compiledCode.size() * 4);
    cacheFileWrite.close();
    /*spvtools::SpirvTools core(SPV_ENV_VULKAN_1_2);

    core.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
        Log("Shader validation error: {}", message);
    });*/
    /*
    spvtools::ValidatorOptions opts;
    opts.SetScalarBlockLayout(true);
    if (!core.Validate(compiledCode.data(), compiledCode.size(), opts))
    {
        Log("validation error");
        throw std::runtime_error("shader validation error");
    }
    return compiledCode;
}*/

void VulkanDriver::destroyVertexBuffer(VertexBufferHandle handle) {
    destructHandle<VulkanVertexBuffer>(handle);
}

void VulkanDriver::destroyIndexBuffer(IndexBufferHandle handle) {
    destructHandle<VulkanIndexBuffer>(handle);
}

void VulkanDriver::destroyBufferObject(BufferObjectHandle handle) {
    destructHandle<VulkanBufferObject>(handle);
}

void VulkanDriver::destroyPrimitive(PrimitiveHandle handle) {
    destructHandle<VulkanPrimitive>(handle);
}

void VulkanDriver::destroyTexture(TextureHandle handle) {
    destructHandle<VulkanTexture>(handle);
}

void VulkanDriver::destroyRenderTarget(RenderTargetHandle handle) {
    destructHandle<VulkanRenderTarget>(handle);
}

void VulkanDriver::endRenderPass(int dummy) {
    auto& cmd = context.inflight->cmd;
    vkCmdEndRenderPass(cmd);
}

Handle<HwBLAS> VulkanDriver::createBLAS(Handle<HwPrimitive> primitiveHandle) {
    auto& cmd = context.inflight->cmd;
    VulkanPrimitive* primitive = handleCast<VulkanPrimitive>(primitiveHandle);

    VulkanBufferObject* scratch;
    Handle<HwBLAS> handle = allocHandle<VulkanBLAS, HwBLAS>();
    constructHandle<VulkanBLAS>(handle, *device, cmd, primitive, &scratch);
    return handle;
}

Handle<HwTLAS> VulkanDriver::createTLAS(RTSceneDescriptor descriptor) {
    auto& cmd = context.inflight->cmd;
    VulkanRTSceneDescriptor desc = {};

    for (size_t i = 0; i < descriptor.count; ++i) {
        VulkanRTInstance instance = {};
        instance.blas = handleCast<VulkanBLAS>(descriptor.instances[i].blas);
        instance.transfom = descriptor.instances[i].transfom;
        desc.instances.push_back(instance);
    }
    VulkanBufferObject* scratch;
    Handle<HwTLAS> handle = allocHandle<VulkanTLAS, HwTLAS>();
    constructHandle<VulkanTLAS>(handle, *device, cmd, desc, &scratch);
    return handle;
}

void VulkanDriver::copyBufferObject(Handle<HwBufferObject> destHandle, Handle<HwBufferObject> srcHandle) {
    auto& cmd = context.inflight->cmd;
    VulkanBufferObject* dest = handleCast<VulkanBufferObject>(destHandle);
    VulkanBufferObject* src = handleCast<VulkanBufferObject>(srcHandle);
    dest->copy(cmd, *src);
}

ImageLayout VulkanDriver::getTextureImageLayout(Handle<HwTexture> handle) {
    auto texture = handleCast<VulkanTexture, HwTexture>(handle);
    return texture->imageLayout;
}


void VulkanDriver::destroyBLAS(Handle<HwBLAS> handle) {
    destructHandle<VulkanBLAS>(handle);
}

void VulkanDriver::destroyTLAS(Handle<HwTLAS> handle) {
    destructHandle<VulkanTLAS>(handle);
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

VulkanBindings VulkanDriver::translateBindingMap(const ProgramParameterLayout& layout, const BindingMap & binds) {
    ZoneScopedN("Translate binding map")
    VulkanBindings bindings;
    for (auto [key, binding] : binds) {
        VulkanBinding vb = {};
        vb.binding = key.binding;
        vb.arraySize = 0;
        vb.set = key.set;
        vb.type = translateDescriptorType(binding.type);
        auto& field = layout.fields[key.set * ProgramParameterLayout::MAX_BINDINGS + key.binding];
        switch (binding.type) {
            case ProgramParameterType::UNIFORM: {
                for (auto handle : binding.handles) {
                    if (!handle.texture && !handle.buffer) {
                        break;
                    }
                    auto ub = handleCast<VulkanBufferObject>(handle.buffer);
                    vk::DescriptorBufferInfo bufferInfo{};
                    bufferInfo.offset = 0;
                    bufferInfo.buffer = ub->buffer;
                    bufferInfo.range = ub->size;
                    vb.bufferInfo.push_back(bufferInfo);
                    ++vb.arraySize;
                }
                /*size_t rem = field.arraySize() - vb.arraySize;
                for (size_t t = 0; t < rem; ++t) {
                    vk::DescriptorBufferInfo bufferInfo{};
                    bufferInfo.offset = 0;
                    bufferInfo.buffer = dummyBuf->buffer;
                    bufferInfo.range = dummyBuf->size;
                    vb.bufferInfo.push_back(bufferInfo);
                    ++vb.arraySize;
                }*/
                break;
            }
            case ProgramParameterType::STORAGE: {
                for (auto handle : binding.handles) {
                    if (!handle.texture && !handle.buffer) {
                        break;
                    }
                    auto sb = handleCast<VulkanBufferObject>(handle.buffer);
                    vk::DescriptorBufferInfo bufferInfo{};
                    bufferInfo.offset = 0;
                    bufferInfo.buffer = sb->buffer;
                    bufferInfo.range = sb->size;
                    vb.bufferInfo.push_back(bufferInfo);
                    ++vb.arraySize;
                }
                /*size_t rem = field.arraySize() - vb.arraySize;
                for (size_t t = 0; t < rem; ++t) {
                    vk::DescriptorBufferInfo bufferInfo{};
                    bufferInfo.offset = 0;
                    bufferInfo.buffer = dummyBuf->buffer;
                    bufferInfo.range = dummyBuf->size;
                    vb.bufferInfo.push_back(bufferInfo);
                    ++vb.arraySize;
                }*/
                break;
            }
            case ProgramParameterType::IMAGE:
            case ProgramParameterType::TEXTURE: {
                for (auto handle : binding.handles) {
                    if (!handle.texture && !handle.buffer) {
                        break;
                    }
                    auto tex = handleCast<VulkanTexture>(handle.texture);
                    vk::DescriptorImageInfo imageInfo{};
                    imageInfo.imageLayout = tex->vkImageLayout;
                    imageInfo.imageView = tex->view;
                    imageInfo.sampler = tex->sampler;
                    vb.imageInfo.push_back(imageInfo);
                    ++vb.arraySize;
                }
                if (vb.imageInfo.empty()) {
                    size_t rem = field.arraySize() - vb.arraySize;
                    for (size_t t = 0; t < rem; ++t) {
                        vk::DescriptorImageInfo imageInfo{};
                        imageInfo.imageLayout = dummyTex->vkImageLayout;
                        imageInfo.imageView = dummyTex->view;
                        imageInfo.sampler = dummyTex->sampler;
                        vb.imageInfo.push_back(imageInfo);
                        ++vb.arraySize;
                    }
                }
                break;
            }
            case ProgramParameterType::TLAS: {
                for (auto handle : binding.handles) {
                    if (!handle.tlas) {
                        break;
                    }
                    auto tlas = handleCast<VulkanTLAS>(handle.tlas);
                    vb.tlasInfo.push_back(tlas->handle);
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
        auto texture = handleCast<VulkanTexture, HwTexture>(barrier.image);
        vk::ImageMemoryBarrier imageBarrier{};
        imageBarrier.srcAccessMask = translateAccessMask(barrier.srcAccess);
        imageBarrier.dstAccessMask = translateAccessMask(barrier.dstAccess);
        imageBarrier.oldLayout = translateImageLayout(barrier.initialLayout);
        imageBarrier.newLayout = translateImageLayout(barrier.finalLayout);
        imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imageBarrier.subresourceRange.aspectMask = texture->aspect;
        imageBarrier.subresourceRange.baseArrayLayer = 0;
        imageBarrier.subresourceRange.layerCount = texture->layers;
        imageBarrier.subresourceRange.baseMipLevel = 0;
        imageBarrier.subresourceRange.levelCount= texture->levels;
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