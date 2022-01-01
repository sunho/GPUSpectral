#include "VulkanPipelineCache.h"
#include <iostream>
#include <Tracy.hpp>

#include <VKGIRenderer/utils/Log.h>

void VulkanDescriptorAllocator::init(vk::Device newDevice) {
    device = newDevice;
}

void VulkanDescriptorAllocator::cleanup() {
    for (auto p : freePools) {
        vkDestroyDescriptorPool(device, p, nullptr);
    }
    for (auto p : usedPools) {
        vkDestroyDescriptorPool(device, p, nullptr);
    }
}

vk::DescriptorPool createPool(vk::Device device, const VulkanDescriptorAllocator::PoolSizes &poolSizes, int count, vk::DescriptorPoolCreateFlags flags) {
    std::vector<vk::DescriptorPoolSize> sizes;
    sizes.reserve(poolSizes.sizes.size());
    for (auto sz : poolSizes.sizes) {
        sizes.push_back({ sz.first, uint32_t(sz.second * count) });
    }
    vk::DescriptorPoolCreateInfo pi = {};
    pi.flags = flags;
    pi.maxSets = count;
    pi.poolSizeCount = (uint32_t)sizes.size();
    pi.pPoolSizes = sizes.data();

    return device.createDescriptorPool(pi, nullptr);
}

vk::DescriptorPool VulkanDescriptorAllocator::grabPool() {
    if (freePools.size() > 0) {
        VkDescriptorPool pool = freePools.back();
        freePools.pop_back();
        return pool;
    } else {
        return createPool(device, descriptorSizes, 1024, {});
    }
}

vk::DescriptorSet VulkanDescriptorAllocator::allocate(vk::DescriptorSetLayout layout) {
    if (!currentPool) {
        currentPool = grabPool();
        usedPools.push_back(currentPool);
    }

    vk::DescriptorSetAllocateInfo allocInfo = {};
    allocInfo.pSetLayouts = &layout;
    allocInfo.descriptorPool = currentPool;
    allocInfo.descriptorSetCount = 1;

    //try to allocate the descriptor set
    vk::DescriptorSet descriptorSet;
    vk::Result res = device.allocateDescriptorSets(&allocInfo, &descriptorSet);
    switch (res) {
        case vk::Result::eSuccess:
            //all good, return
            return descriptorSet;
        case vk::Result::eErrorFragmentedPool:
        case vk::Result::eErrorOutOfPoolMemory:
            break;
        default:
            throw std::runtime_error("pool allocation error");
    }
    //allocate a new pool and retry
    currentPool = grabPool();
    usedPools.push_back(currentPool);
    allocInfo.descriptorPool = currentPool;
    res = device.allocateDescriptorSets(&allocInfo, &descriptorSet);
    if (res != vk::Result::eSuccess) {
        throw std::runtime_error("pool allocation error");
    }
    return descriptorSet;
}

void VulkanDescriptorAllocator::resetPools() {
    //reset all used pools and add them to the free pools
    for (auto p : usedPools) {
        device.resetDescriptorPool(p);
        freePools.push_back(p);
    }

    //clear the used pools, since we've put them all in the free pools
    usedPools.clear();

    //reset the current pool handle back to null
    currentPool = nullptr;
}


VulkanPipelineCache::VulkanPipelineCache(VulkanDevice &device) : device(device) {
    computePipelines.setDestroyer([=](VkPipeline pipeline) {
        vkDestroyPipeline(this->device.device, pipeline, nullptr);
    });
    graphicsPipelines.setDestroyer([=](VkPipeline pipeline) {
        vkDestroyPipeline(this->device.device, pipeline, nullptr);
    });
    framebuffers.setDestroyer([=](VkFramebuffer framebuffer) {
        std::cout << "destroy " << framebuffer << std::endl;
        vkDestroyFramebuffer(this->device.device, framebuffer, nullptr);
    });
    renderpasses.setDestroyer([=](VkRenderPass renderPass) {
        vkDestroyRenderPass(this->device.device, renderPass, nullptr);
    });
    for (auto& alloc : descriptorAllocators) {
        alloc.init(device.device);
    }
}

void VulkanPipelineCache::tick() {
    computePipelines.tick();
    graphicsPipelines.tick();
    framebuffers.tick();
    renderpasses.tick();
    ++currentFrame;
    currentDescriptorAllocator().resetPools();
}

VulkanPipelineCache::PipelineLayout VulkanPipelineCache::getOrCreatePipelineLayout(const ProgramParameterLayout &layout) {
    ZoneScopedN("PipelineCache create layout")
    auto it = pipelineLayouts.get(layout);
    
    if (it) {
        return *it;
    }
    
    VulkanPipelineCache::PipelineLayout pipelineLayout{};
    vk::PipelineLayoutCreateInfo createInfo{};
    vk::PushConstantRange pushConstants{};
    pushConstants.offset = 0;
    pushConstants.size = MAX_PUSH_CONSTANT_SIZE;
    pushConstants.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eFragment;
    createInfo.pPushConstantRanges = &pushConstants;
    createInfo.pushConstantRangeCount = 1;


    for (size_t i = 0; i < ProgramParameterLayout::MAX_SET; ++i) {
        std::vector<vk::DescriptorSetLayoutBinding> bindings;
        for (size_t j = 0; j < ProgramParameterLayout::MAX_BINDINGS; ++j) {
            auto &field = layout.fields[i * ProgramParameterLayout::MAX_BINDINGS + j];
            if (field) {
                vk::DescriptorSetLayoutBinding binding{};
                binding.descriptorType = translateDescriptorType(field.type());
                binding.binding = j;
                binding.descriptorCount = field.arraySize();
                binding.stageFlags = vk::ShaderStageFlagBits::eAll;
                bindings.push_back(binding);
            }
        }
        vk::DescriptorSetLayoutCreateInfo di{};
        di.setBindings(bindings);
        pipelineLayout.descriptorSetLayout[i] = device.device.createDescriptorSetLayout(di);
    }
    createInfo.setLayoutCount = 3;
    createInfo.pSetLayouts = pipelineLayout.descriptorSetLayout.data();
    pipelineLayout.pipelineLayout = device.device.createPipelineLayout(createInfo);

    pipelineLayouts.add(layout, pipelineLayout);
    return pipelineLayout;
}

void VulkanPipelineCache::bindDescriptor(vk::CommandBuffer cmd, const vk::PipelineBindPoint& bindPoint, const ProgramParameterLayout& layout, const VulkanBindings& bindings) {
    ZoneScopedN("PipelineCache bind descriptor")
    auto pipelineLayout = getOrCreatePipelineLayout(layout);
    DescriptorSets descriptorSets{};
    for (size_t i = 0; i < ProgramParameterLayout::MAX_SET; ++i) {
        descriptorSets[i] = currentDescriptorAllocator().allocate(pipelineLayout.descriptorSetLayout[i]);
    }

    std::vector<vk::WriteDescriptorSet> writes;
    writes.reserve(bindings.size());
    for (auto &binding : bindings) {
        vk::WriteDescriptorSet write{};
        write.dstBinding = binding.binding;
        write.dstSet = descriptorSets[binding.set];
        write.dstArrayElement = 0;
        write.descriptorCount = binding.arraySize;
        write.descriptorType = binding.type;
        write.pBufferInfo = binding.bufferInfo.data();
        write.pImageInfo = binding.imageInfo.data();
        writes.push_back(write);
    }
    device.device.updateDescriptorSets(writes.size(), writes.data(), 0, nullptr);
    cmd.bindDescriptorSets(bindPoint, pipelineLayout.pipelineLayout, 0, ProgramParameterLayout::MAX_SET, descriptorSets.data(), 0, nullptr);
}

VulkanPipeline VulkanPipelineCache::getOrCreateGraphicsPipeline(const VulkanPipelineState &state) {
    ZoneScopedN("PipelineCache create graphics pipeline")
    auto pipelineLayout = getOrCreatePipelineLayout(state.parameterLayout);
    auto it = graphicsPipelines.get(state);
    if (it) {
        return { *it, pipelineLayout.pipelineLayout };
    }
    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
    vertShaderStageInfo.module = state.vertex->shaderModule;
    vertShaderStageInfo.pName = "main";

    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
    fragShaderStageInfo.module = state.fragment->shaderModule;
    fragShaderStageInfo.pName = "main";
    vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.vertexBindingDescriptionCount = state.attributeCount;
    std::vector<vk::VertexInputAttributeDescription> attributes(state.attributeCount);
    std::vector<vk::VertexInputBindingDescription> bindings(state.attributeCount);

    for (uint32_t i = 0; i < state.attributeCount; ++i) {
        Attribute attrib = state.attributes[i];
        attributes[i] = { .location = i,
                          .binding = i,
                          .format = (VkFormat)translateElementFormat(attrib.type,
                                                                   attrib.flags & Attribute::FLAG_NORMALIZED, false) };
        bindings[i] = {
            .binding = i,
            .stride = attrib.stride,
        };
    }

    vertexInputInfo.pVertexBindingDescriptions = bindings.data();
    vertexInputInfo.vertexAttributeDescriptionCount = state.attributeCount;
    vertexInputInfo.pVertexAttributeDescriptions = attributes.data();

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
    inputAssembly.primitiveRestartEnable = false;

    vk::Viewport viewport{};
    viewport.x = (float)state.viewport.left;
    viewport.y = (float)state.viewport.top;
    viewport.width = (float)state.viewport.width;
    viewport.height = (float)state.viewport.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    vk::Rect2D scissor{};
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent = { .width = (uint32_t)std::numeric_limits<int32_t>::max(), .height = (uint32_t)std::numeric_limits<int32_t>::max() };

    vk::PipelineViewportStateCreateInfo viewportState{};
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    vk::PipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.depthTestEnable = state.depthTest.enabled ? true : false;
    depthStencil.depthWriteEnable = state.depthTest.write ? true : false;
    depthStencil.depthCompareOp = translateCompareOp(state.depthTest.compareOp);
    depthStencil.depthBoundsTestEnable = false;
    depthStencil.minDepthBounds = 0.0f;  // Optional
    depthStencil.maxDepthBounds = 1.0f;  // Optional
    depthStencil.stencilTestEnable = false;

    vk::PipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.depthClampEnable = false;
    rasterizer.rasterizerDiscardEnable = false;
    rasterizer.polygonMode = vk::PolygonMode::eFill;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = vk::CullModeFlagBits::eBack;
    rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
    rasterizer.depthBiasEnable = false;
    rasterizer.depthBiasConstantFactor = 0.0f;  // Optional
    rasterizer.depthBiasClamp = 0.0f;           // Optional
    rasterizer.depthBiasSlopeFactor = 0.0f;     // Optional

    vk::PipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sampleShadingEnable = false;
    multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
    multisampling.minSampleShading = 1.0f;           // Optional
    multisampling.pSampleMask = nullptr;             // Optional
    multisampling.alphaToCoverageEnable = false;  // Optional
    multisampling.alphaToOneEnable = false;          // Optional

    std::vector<vk::PipelineColorBlendAttachmentState> attachments{};
    vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
    
    colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    colorBlendAttachment.blendEnable = false;
    colorBlendAttachment.srcColorBlendFactor = vk::BlendFactor::eOne;
    colorBlendAttachment.dstColorBlendFactor = vk::BlendFactor::eZero;
    colorBlendAttachment.colorBlendOp = vk::BlendOp::eAdd;
    colorBlendAttachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
    colorBlendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eZero;
    colorBlendAttachment.alphaBlendOp = vk::BlendOp::eAdd;

    colorBlendAttachment.blendEnable = false;
    colorBlendAttachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
    colorBlendAttachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
    colorBlendAttachment.colorBlendOp = vk::BlendOp::eAdd;
    colorBlendAttachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
    colorBlendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eZero;
    colorBlendAttachment.alphaBlendOp = vk::BlendOp::eAdd;

    for (size_t i = 0; i < state.attachmentCount; ++i) {
        attachments.push_back(colorBlendAttachment);
    }
    vk::PipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.logicOpEnable = false;
    colorBlending.logicOp = vk::LogicOp::eCopy;
    colorBlending.attachmentCount = state.attachmentCount;
    colorBlending.pAttachments = attachments.data();
    colorBlending.blendConstants[0] = 0.0f;  // Optional
    colorBlending.blendConstants[1] = 0.0f;  // Optional
    colorBlending.blendConstants[2] = 0.0f;  // Optional
    colorBlending.blendConstants[3] = 0.0f;  // Optional

    vk::DynamicState dynamicStates[] = { vk::DynamicState::eViewport, vk::DynamicState::eLineWidth };

    vk::PipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;


    vk::GraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;  // Optional
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = nullptr;  // Optional
    pipelineInfo.layout = pipelineLayout.pipelineLayout;
    pipelineInfo.renderPass = state.renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = nullptr;  // Optional
    pipelineInfo.basePipelineIndex = -1;               // Optional

    vk::Pipeline out = device.device.createGraphicsPipeline(nullptr, pipelineInfo);
    graphicsPipelines.add(state, out);
    return { out, pipelineLayout.pipelineLayout };
}

VulkanPipeline VulkanPipelineCache::getOrCreateComputePipeline(const VulkanProgram &program) {
    ZoneScopedN("PipelineCache create compute pipeline")
    auto pipelineLayout = getOrCreatePipelineLayout(program.program.parameterLayout);
    auto it = computePipelines.get(program.program.hash);
    if (it) {
        return { *it, pipelineLayout.pipelineLayout };
    }
    vk::PipelineShaderStageCreateInfo shaderStageInfo{};
    shaderStageInfo.stage = vk::ShaderStageFlagBits::eCompute;
    shaderStageInfo.module = program.shaderModule;
    shaderStageInfo.pName = "main";

    vk::ComputePipelineCreateInfo createInfo;
    createInfo.layout = pipelineLayout.pipelineLayout;
    createInfo.stage = shaderStageInfo;
    vk::Pipeline pipeline = device.device.createComputePipeline(nullptr, createInfo, nullptr);
    computePipelines.add(program.program.hash, pipeline);
    return {
        pipeline, pipelineLayout.pipelineLayout
    };
}

VulkanPipeline VulkanPipelineCache::getOrCreateRTPipeline(const VulkanRTPipelineState& state) {
    auto it = rtPipelines.get(state);
    auto pipelineLayout = getOrCreatePipelineLayout(state.parameterLayout);
    if (it) {
        return { *it, pipelineLayout.pipelineLayout };
    }
    const size_t MAX_RECURSION_DEPTH = 2;
    const auto wrapShaderModule = [](const vk::ShaderModule& shaderModule, const vk::ShaderStageFlagBits& flags) {
        return vk::PipelineShaderStageCreateInfo()
            .setStage(flags)
            .setPName("main")
            .setModule(shaderModule);
    };
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups;
    {
        stages.push_back(wrapShaderModule(state.raygenGroup->shaderModule, vk::ShaderStageFlagBits::eRaygenKHR));
        auto gi = vk::RayTracingShaderGroupCreateInfoKHR()
            .setGeneralShader(stages.size() - 1)
            .setType(vk::RayTracingShaderGroupTypeKHR::eGeneral);
        groups.push_back(gi);
    }
    for (auto& hitGroup : state.hitGroups) {
        stages.push_back(wrapShaderModule(hitGroup->shaderModule, vk::ShaderStageFlagBits::eClosestHitKHR));
        auto gi = vk::RayTracingShaderGroupCreateInfoKHR()
            .setGeneralShader(stages.size() - 1)
            .setType(vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup);
        groups.push_back(gi);
    }
    for (auto& missGroup: state.missGroups) {
        stages.push_back(wrapShaderModule(missGroup->shaderModule, vk::ShaderStageFlagBits::eMissKHR));
        auto gi = vk::RayTracingShaderGroupCreateInfoKHR()
            .setGeneralShader(stages.size() - 1)
            .setType(vk::RayTracingShaderGroupTypeKHR::eGeneral);
        groups.push_back(gi);
    }
    for (auto& callGroup : state.callableGroups) {
        stages.push_back(wrapShaderModule(callGroup->shaderModule, vk::ShaderStageFlagBits::eCallableKHR));
        auto gi = vk::RayTracingShaderGroupCreateInfoKHR()
            .setGeneralShader(stages.size() - 1)
            .setType(vk::RayTracingShaderGroupTypeKHR::eGeneral);
        groups.push_back(gi);
    }
    VkRayTracingPipelineCreateInfoKHR createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    createInfo.stageCount = stages.size();
    createInfo.pStages = stages.data();
    createInfo.groupCount = groups.size();
    createInfo.pGroups = groups.data();
    createInfo.maxPipelineRayRecursionDepth = MAX_RECURSION_DEPTH;
    VkPipeline pipeline;
    auto res = device.dld.vkCreateRayTracingPipelinesKHR(device.device, nullptr, nullptr, 1, &createInfo, nullptr, &pipeline);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("failed to create rt pipeline");
    }
    rtPipelines.add(state, pipeline);
    return {
        pipeline, pipelineLayout.pipelineLayout
    };
}

VulkanShaderBindingTables VulkanPipelineCache::getOrCreateSBT(const VulkanRTPipelineState& state) {
    auto pipeline = getOrCreateRTPipeline(state);
    auto it = rtSBTs.get(state);
    if (it) {
        return *it;
    }
    const uint32_t sbtSize = state.getGroupCount() * device.shaderGroupHandleSizeAligned;

    std::vector<uint8_t> shaderHandleStorage(sbtSize);
    device.dld.vkGetRayTracingShaderGroupHandlesKHR(device.device, pipeline.pipeline, 0, state.getGroupCount(), sbtSize, shaderHandleStorage.data());
    VulkanShaderBindingTables tables{};
    const uint32_t handleSize = device.shaderGroupHandleSize;
    const uint32_t handleSizeAligned = device.shaderGroupHandleSizeAligned;
    tables.raygen = VulkanShaderBindingTable(device, 1);
    memcpy(tables.raygen.buffer->mapped, shaderHandleStorage.data(), handleSize);
    uint32_t index = 1;
    if (!state.hitGroups.empty()) {
        tables.hit = VulkanShaderBindingTable(device, state.hitGroups.size());
        memcpy(tables.hit.buffer->mapped, shaderHandleStorage.data() + handleSizeAligned * index, state.hitGroups.size() * handleSize);
        index += state.hitGroups.size();
    }
    if (!state.missGroups.empty()) {
        tables.miss = VulkanShaderBindingTable(device, state.missGroups.size());
        memcpy(tables.miss.buffer->mapped, shaderHandleStorage.data() + handleSizeAligned * index, state.missGroups.size() * handleSize);
        index += state.missGroups.size();
    }
    if (!state.callableGroups.empty()) {
        tables.callable = VulkanShaderBindingTable(device, state.callableGroups.size());
        memcpy(tables.callable.buffer->mapped, shaderHandleStorage.data() + handleSizeAligned * index, state.callableGroups.size() * handleSize);
        index += state.callableGroups.size();
    }
    rtSBTs.add(state, tables);
    return tables;
}

VkRenderPass VulkanPipelineCache::getOrCreateRenderPass(VulkanSwapChain swapchain, VulkanRenderTarget *renderTarget) {
    ZoneScopedN("PipelineCache create renderpass")
    auto it = renderpasses.get(renderTarget->attachments);
    if (it) {
        return *it;
    }
    VkRenderPass out;
    std::vector<VkAttachmentDescription> attachments;
    bool hasDepth = false;
    VkAttachmentReference depthAttachmentRef{};
    std::vector<VkAttachmentReference> colorAttachmentRef;

    if (renderTarget->surface) {
        attachments.push_back({ .format = (VkFormat)swapchain.format,
                                .samples = VK_SAMPLE_COUNT_1_BIT,
                                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR });
        colorAttachmentRef.push_back({ .attachment = 0,
                                       .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
    } else {
        for (size_t i = 0; i < RenderAttachments::MAX_MRT_NUM; ++i) {
            auto &color = renderTarget->attachments.colors[i];
            if (color.valid) {
                attachments.push_back({ .format = color.format,
                                        .samples = VK_SAMPLE_COUNT_1_BIT,
                                        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                                        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                        .finalLayout = VK_IMAGE_LAYOUT_GENERAL });

                colorAttachmentRef.push_back({ .attachment = (uint32_t)i,
                                               .layout = VK_IMAGE_LAYOUT_GENERAL });
            }
        }
        if (renderTarget->attachments.depth.valid) {
            depthAttachmentRef.attachment = attachments.size();
            attachments.push_back({ .format = renderTarget->attachments.depth.format,
                                    .samples = VK_SAMPLE_COUNT_1_BIT,
                                    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                                    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                    .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                    .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                    .finalLayout = VK_IMAGE_LAYOUT_GENERAL });

            depthAttachmentRef.layout = VK_IMAGE_LAYOUT_GENERAL;
            hasDepth = true;
        }
    }

    VkSubpassDescription subpass{ .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  .colorAttachmentCount = (uint32_t)colorAttachmentRef.size(),
                                  .pColorAttachments = colorAttachmentRef.data(),
                                  .pDepthStencilAttachment = hasDepth ? &depthAttachmentRef : nullptr };

    VkRenderPassCreateInfo renderPassInfo{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = (uint32_t)attachments.size(),
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
    };

    if (vkCreateRenderPass(device.device, &renderPassInfo, nullptr, &out) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }

    renderpasses.add(renderTarget->attachments, out);
    return out;
}

vk::Framebuffer VulkanPipelineCache::getOrCreateFrameBuffer(vk::RenderPass renderPass,
                                                        VulkanSwapChain swapchain, 
                                                          VulkanRenderTarget *renderTarget) {
    ZoneScopedN("PipelineCache create frame buffer")
    auto it = framebuffers.get(
        std::make_pair(renderPass, renderTarget->surface ? swapchain.view : nullptr));
    if (it) {
        std::cout << renderPass << ":" << renderTarget->surface << swapchain.view << " " << *it << std::endl;
        return *it;
    }

    std::vector<VkImageView> attachments;
    if (renderTarget->surface) {
        attachments.push_back(swapchain.view);
    } else {
        for (size_t i = 0; i < RenderAttachments::MAX_MRT_NUM; ++i) {
            auto &color = renderTarget->attachments.colors[i];
            if (color.valid) {
                attachments.push_back(color.view);
            }
        }
        if (renderTarget->attachments.depth.valid) {
            attachments.push_back(renderTarget->attachments.depth.view);
        }
    }
    VkFramebuffer framebuffer;
    auto extent = renderTarget->getExtent(device);
    VkFramebufferCreateInfo framebufferInfo{ .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                                             .renderPass = renderPass,
                                             .attachmentCount = (uint32_t)attachments.size(),
                                             .pAttachments = attachments.data(),
                                             .width = extent.width,
                                             .height = extent.height,
                                             .layers = 1 };

    if (vkCreateFramebuffer(device.device, &framebufferInfo, nullptr, &framebuffer) !=
        VK_SUCCESS) {
        throw std::runtime_error("failed to create framebuffer!");
    }

    framebuffers.add(std::make_pair(renderPass, renderTarget->surface ? swapchain.view : nullptr),
                     framebuffer);
    return framebuffer;
}


