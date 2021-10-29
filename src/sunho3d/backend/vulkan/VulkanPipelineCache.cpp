#include "VulkanPipelineCache.h"
#include <iostream>


VulkanPipelineCache::VulkanPipelineCache(VulkanDevice &device) : device(device) {
    dummyImage = std::make_unique<VulkanTexture>(device, SamplerType::SAMPLER2D, TextureUsage::UPLOADABLE | TextureUsage::SAMPLEABLE | TextureUsage::COLOR_ATTACHMENT | TextureUsage::INPUT_ATTACHMENT, 1, TextureFormat::RGBA8, 1,1);
    
    VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = 16,
        .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
    };
    dummyBuffer = new VulkanBufferObject(device, 16, BufferUsage::UNIFORM | BufferUsage::STORAGE);
    dummyBufferInfo.buffer = dummyBuffer->buffer;
    dummyBufferInfo.range = bufferInfo.size;

    dummyBufferWriteInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    dummyBufferWriteInfo.pNext = nullptr;
    dummyBufferWriteInfo.dstArrayElement = 0;
    dummyBufferWriteInfo.descriptorCount = 1;
    dummyBufferWriteInfo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    dummyBufferWriteInfo.pImageInfo = nullptr;
    dummyBufferWriteInfo.pBufferInfo = &dummyBufferInfo;
    dummyBufferWriteInfo.pTexelBufferView = nullptr;

    dummySamplerInfo.imageView = dummyImage->view;
    dummySamplerInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkSamplerCreateInfo samplerInfo{ .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                        .magFilter = VK_FILTER_NEAREST,
                                        .minFilter = VK_FILTER_NEAREST,
                                        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                        .anisotropyEnable = VK_FALSE,
                                        .maxAnisotropy = 1,
                                        .compareEnable = VK_FALSE,
                                        .compareOp = VK_COMPARE_OP_ALWAYS,
                                        .minLod = 0.0f,
                                        .maxLod = 1.0f,
                                        .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
                                        .unnormalizedCoordinates = VK_FALSE };
    vkCreateSampler(device.device, &samplerInfo, nullptr, &dummySamplerInfo.sampler);

    dummySamplerWriteInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    dummySamplerWriteInfo.pNext = nullptr;
    dummySamplerWriteInfo.dstArrayElement = 0;
    dummySamplerWriteInfo.descriptorCount = 1;
    dummySamplerWriteInfo.pImageInfo = &dummySamplerInfo;
    dummySamplerWriteInfo.pBufferInfo = nullptr;
    dummySamplerWriteInfo.pTexelBufferView = nullptr;
    dummySamplerWriteInfo.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
   
    dummyTargetInfo.imageView = dummyImage->view;
    dummyTargetInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    dummyTargetWriteInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    dummyTargetWriteInfo.pNext = nullptr;
    dummyTargetWriteInfo.dstArrayElement = 0;
    dummyTargetWriteInfo.descriptorCount = 1;
    dummyTargetWriteInfo.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    dummyTargetWriteInfo.pImageInfo = &dummyTargetInfo;
    dummyTargetWriteInfo.pBufferInfo = nullptr;
    dummyTargetWriteInfo.pTexelBufferView = nullptr;
    setupLayouts(graphicsLayout, false);

    
    pipelines.setDestroyer([=](VkPipeline pipeline) {
        vkDestroyPipeline(this->device.device, pipeline, nullptr);
    });
    framebuffers.setDestroyer([=](VkFramebuffer framebuffer) {
        vkDestroyFramebuffer(this->device.device, framebuffer, nullptr);
    });
    descriptorSets.setDestroyer([=](std::array<VkDescriptorSet, 4> descriptors) {
        vkFreeDescriptorSets(this->device.device, descriptorPool, 4, descriptors.data());
    });
    renderpasses.setDestroyer([=](VkRenderPass renderPass) {
        vkDestroyRenderPass(this->device.device, renderPass, nullptr);
    });
}

void VulkanPipelineCache::tick() {
    pipelines.tick();
    framebuffers.tick();
    descriptorSets.tick();
    renderpasses.tick();
}

void VulkanPipelineCache::setupLayouts(PipelineLayout& layout, bool compute) {
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

    pipelineLayoutInfo.pushConstantRangeCount = 0;     // Optional
    pipelineLayoutInfo.pPushConstantRanges = nullptr;  // Optional

    VkDescriptorSetLayoutBinding binding = {};
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS;

    std::array<VkDescriptorSetLayoutBinding, VulkanDescriptor::UBUFFER_BINDING_COUNT> ubindings;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    for (uint32_t i = 0; i < VulkanDescriptor::UBUFFER_BINDING_COUNT; i++) {
        binding.binding = i;
        ubindings[i] = binding;
    }
    VkDescriptorSetLayoutCreateInfo dlinfo = {};
    dlinfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlinfo.bindingCount = ubindings.size();
    dlinfo.pBindings = ubindings.data();
    vkCreateDescriptorSetLayout(device.device, &dlinfo, nullptr, &layout.descriptorSetLayout[0]);

    std::array<VkDescriptorSetLayoutBinding, VulkanDescriptor::SAMPLER_BINDING_COUNT> sbindings;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    for (uint32_t i = 0; i < VulkanDescriptor::SAMPLER_BINDING_COUNT; i++) {
        binding.binding = i;
        sbindings[i] = binding;
    }
    dlinfo.bindingCount = sbindings.size();
    dlinfo.pBindings = sbindings.data();
    vkCreateDescriptorSetLayout(device.device, &dlinfo, nullptr, &layout.descriptorSetLayout[1]);
    
    if (compute) {
        std::array<VkDescriptorSetLayoutBinding, VulkanDescriptor::TARGET_BINDING_COUNT> bindings;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        for (uint32_t i = 0; i < VulkanDescriptor::TARGET_BINDING_COUNT; i++) {
            binding.binding = i;
            bindings[i] = binding;
        }
        dlinfo.bindingCount = bindings.size();
        dlinfo.pBindings = bindings.data();
        vkCreateDescriptorSetLayout(device.device, &dlinfo, nullptr, &layout.descriptorSetLayout[2]);
    } else {
        std::array<VkDescriptorSetLayoutBinding, VulkanDescriptor::STORAGE_BINDING_COUNT> bindings2;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        for (uint32_t i = 0; i < VulkanDescriptor::STORAGE_BINDING_COUNT; i++) {
            binding.binding = i;
            bindings2[i] = binding;
        }
        dlinfo.bindingCount = bindings2.size();
        dlinfo.pBindings = bindings2.data();
        vkCreateDescriptorSetLayout(device.device, &dlinfo, nullptr, &layout.descriptorSetLayout[3]);

        std::array<VkDescriptorSetLayoutBinding, VulkanDescriptor::TARGET_BINDING_COUNT> bindings;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        for (uint32_t i = 0; i < VulkanDescriptor::TARGET_BINDING_COUNT; i++) {
            binding.binding = i;
            bindings[i] = binding;
        }
        dlinfo.bindingCount = bindings.size();
        dlinfo.pBindings = bindings.data();
        vkCreateDescriptorSetLayout(device.device, &dlinfo, nullptr, &layout.descriptorSetLayout[2]);
    }

    pipelineLayoutInfo.setLayoutCount = 4;
    pipelineLayoutInfo.pSetLayouts = layout.descriptorSetLayout.data();

    layout.pipelineLayout = device.device.createPipelineLayout(pipelineLayoutInfo, nullptr);
    VkDescriptorPoolSize poolSizes[4] = {};
    VkDescriptorPoolCreateInfo poolInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                         .pNext = nullptr,
                                         .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                                         .maxSets = 1024 * 3,
                                         .poolSizeCount = 3,
                                         .pPoolSizes = poolSizes };
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = poolInfo.maxSets * VulkanDescriptor::UBUFFER_BINDING_COUNT;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = poolInfo.maxSets * VulkanDescriptor::SAMPLER_BINDING_COUNT;
    poolSizes[2].type = compute ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    poolSizes[2].descriptorCount = poolInfo.maxSets * VulkanDescriptor::TARGET_BINDING_COUNT;
    poolSizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[3].descriptorCount = poolInfo.maxSets * VulkanDescriptor::STORAGE_BINDING_COUNT;
    if (vkCreateDescriptorPool(device.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }
}

void VulkanPipelineCache::getOrCreateDescriptors(const VulkanDescriptor &key,
                                                 std::array<VkDescriptorSet, 4> &descriptors) {
    auto it = descriptorSets.get(key);
    if (it) {
        descriptors = *it;
        return;
    }
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 4;
    allocInfo.pSetLayouts = graphicsLayout.descriptorSetLayout.data();
    if (vkAllocateDescriptorSets(device.device, &allocInfo, descriptors.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    VkDescriptorBufferInfo descriptorBuffers[VulkanDescriptor::UBUFFER_BINDING_COUNT];
    VkDescriptorImageInfo descriptorSamplers[VulkanDescriptor::SAMPLER_BINDING_COUNT];
    VkDescriptorImageInfo descriptorInputAttachments[VulkanDescriptor::TARGET_BINDING_COUNT];
    VkDescriptorBufferInfo descriptorStorageBuffers[VulkanDescriptor::STORAGE_BINDING_COUNT];
    VkWriteDescriptorSet
        descriptorWrites[VulkanDescriptor::UBUFFER_BINDING_COUNT + VulkanDescriptor::SAMPLER_BINDING_COUNT + VulkanDescriptor::TARGET_BINDING_COUNT + VulkanDescriptor::STORAGE_BINDING_COUNT];

    uint32_t nwrites = 0;
    VkWriteDescriptorSet *writes = descriptorWrites;
    nwrites = 0;
    for (uint32_t binding = 0; binding < VulkanDescriptor::UBUFFER_BINDING_COUNT; binding++) {
        VkWriteDescriptorSet &writeInfo = writes[nwrites++];
        if (key.uniformBuffers[binding]) {
            VkDescriptorBufferInfo &bufferInfo = descriptorBuffers[binding];
            bufferInfo.buffer = key.uniformBuffers[binding];
            bufferInfo.offset = key.uniformBufferOffsets[binding];
            bufferInfo.range = key.uniformBufferSizes[binding];
            writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeInfo.pNext = nullptr;
            writeInfo.dstArrayElement = 0;
            writeInfo.descriptorCount = 1;
            writeInfo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writeInfo.pImageInfo = nullptr;
            writeInfo.pBufferInfo = &bufferInfo;
            writeInfo.pTexelBufferView = nullptr;
        } else {
            writeInfo = dummyBufferWriteInfo;
        }
        writeInfo.dstSet = descriptors[0];
        writeInfo.dstBinding = binding;
    }
    for (uint32_t binding = 0; binding < VulkanDescriptor::SAMPLER_BINDING_COUNT; binding++) {
        VkWriteDescriptorSet &writeInfo = writes[nwrites++];
        if (key.samplers[binding].sampler) {
            VkDescriptorImageInfo &imageInfo = descriptorSamplers[binding];
            imageInfo = key.samplers[binding];
            writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeInfo.pNext = nullptr;
            writeInfo.dstArrayElement = 0;
            writeInfo.descriptorCount = 1;
            writeInfo.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writeInfo.pImageInfo = &imageInfo;
            writeInfo.pBufferInfo = nullptr;
            writeInfo.pTexelBufferView = nullptr;
        } else {
            writeInfo = dummySamplerWriteInfo;
        }
        writeInfo.dstSet = descriptors[1];
        writeInfo.dstBinding = binding;
    }
    for (uint32_t binding = 0; binding < VulkanDescriptor::TARGET_BINDING_COUNT; binding++) {
        VkWriteDescriptorSet &writeInfo = writes[nwrites++];
        if (key.inputAttachments[binding].imageView) {
            VkDescriptorImageInfo &imageInfo = descriptorInputAttachments[binding];
            imageInfo = key.inputAttachments[binding];
            writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeInfo.pNext = nullptr;
            writeInfo.dstArrayElement = 0;
            writeInfo.descriptorCount = 1;
            writeInfo.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
            writeInfo.pImageInfo = &imageInfo;
            writeInfo.pBufferInfo = nullptr;
            writeInfo.pTexelBufferView = nullptr;
        } else {
            writeInfo = dummyTargetWriteInfo;
        }
        writeInfo.dstSet = descriptors[2];
        writeInfo.dstBinding = binding;
    }
    for (uint32_t binding = 0; binding < VulkanDescriptor::STORAGE_BINDING_COUNT; binding++) {
        VkWriteDescriptorSet &writeInfo = writes[nwrites++];
        if (key.storageBuffers[binding]) {
            VkDescriptorBufferInfo &bufferInfo = descriptorStorageBuffers[binding];
            bufferInfo.buffer = key.storageBuffers[binding];
            bufferInfo.offset = 0;
            bufferInfo.range = key.storageBufferSizes[binding];
            writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeInfo.pNext = nullptr;
            writeInfo.dstArrayElement = 0;
            writeInfo.descriptorCount = 1;
            writeInfo.pImageInfo = nullptr;
            writeInfo.pBufferInfo = &bufferInfo;
            writeInfo.pTexelBufferView = nullptr;
        } else {
            writeInfo = dummyBufferWriteInfo;
        }
        writeInfo.dstSet = descriptors[3];
        writeInfo.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeInfo.dstBinding = binding;
    }
    vkUpdateDescriptorSets(device.device, nwrites, writes, 0, nullptr);
    descriptorSets.add(key, descriptors);
}

void VulkanPipelineCache::bindDescriptor(vk::CommandBuffer cmd, const VulkanDescriptor &key) {
    std::array<VkDescriptorSet, 4> descriptors;
    getOrCreateDescriptors(key, descriptors);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsLayout.pipelineLayout,
                            0, 4, descriptors.data(), 0, nullptr);
}

VkPipeline VulkanPipelineCache::getOrCreatePipeline(const VulkanPipelineKey &key) {
    auto it = pipelines.get(key);
    if (it) {
        return *it;
    }

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = key.program->vertex;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = key.program->fragment;
    fragShaderStageInfo.pName = "main";
    VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = key.attributeCount;
    std::vector<VkVertexInputAttributeDescription> attributes(key.attributeCount);
    std::vector<VkVertexInputBindingDescription> bindings(key.attributeCount);

    for (uint32_t i = 0; i < key.attributeCount; ++i) {
        Attribute attrib = key.attributes[i];
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
    vertexInputInfo.vertexAttributeDescriptionCount = key.attributeCount;
    vertexInputInfo.pVertexAttributeDescriptions = attributes.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.x = (float)key.viewport.left;
    viewport.y = (float)key.viewport.top;
    viewport.width = (float)key.viewport.width;
    viewport.height = (float)key.viewport.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = { .width = (uint32_t)std::numeric_limits<int32_t>::max(), .height = (uint32_t)std::numeric_limits<int32_t>::max() };

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = key.depthTest.enabled ? VK_TRUE : VK_FALSE;
    depthStencil.depthWriteEnable = key.depthTest.write ? VK_TRUE : VK_FALSE;
    depthStencil.depthCompareOp = (VkCompareOp)translateCompareOp(key.depthTest.compareOp);
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f;  // Optional
    depthStencil.maxDepthBounds = 1.0f;  // Optional
    depthStencil.stencilTestEnable = VK_FALSE;
    depthStencil.front = {};  // Optional
    depthStencil.back = {};   // Optional

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f;  // Optional
    rasterizer.depthBiasClamp = 0.0f;           // Optional
    rasterizer.depthBiasSlopeFactor = 0.0f;     // Optional

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f;           // Optional
    multisampling.pSampleMask = nullptr;             // Optional
    multisampling.alphaToCoverageEnable = VK_FALSE;  // Optional
    multisampling.alphaToOneEnable = VK_FALSE;       // Optional

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;   // Optional
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;  // Optional
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;              // Optional
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;   // Optional
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;  // Optional
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;              // Optional

    colorBlendAttachment.blendEnable = VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;  // Optional
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;  // Optional
    colorBlending.blendConstants[1] = 0.0f;  // Optional
    colorBlending.blendConstants[2] = 0.0f;  // Optional
    colorBlending.blendConstants[3] = 0.0f;  // Optional

    VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_LINE_WIDTH };

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
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
    pipelineInfo.layout = graphicsLayout.pipelineLayout;
    pipelineInfo.renderPass = key.renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;  // Optional
    pipelineInfo.basePipelineIndex = -1;               // Optional
    VkPipeline out;
    if (vkCreateGraphicsPipelines(device.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                  &out) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    pipelines.add(key, out);
    return out;
}

VkRenderPass VulkanPipelineCache::getOrCreateRenderPass(VulkanSwapChain swapchain, VulkanRenderTarget *renderTarget) {
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
        for (size_t i = 0; i < ColorAttachment::MAX_MRT_NUM; ++i) {
            auto &color = renderTarget->attachments.colors[i];
            if (color.valid) {
                attachments.push_back({ .format = color.format,
                                        .samples = VK_SAMPLE_COUNT_1_BIT,
                                        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                                        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                        .initialLayout = VK_IMAGE_LAYOUT_GENERAL,
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
    auto it = framebuffers.get(
        std::make_pair(renderPass, renderTarget->surface ? swapchain.view : nullptr));
    if (it) {
        return *it;
    }

    std::vector<VkImageView> attachments;
    if (renderTarget->surface) {
        attachments.push_back(swapchain.view);
    } else {
        for (size_t i = 0; i < ColorAttachment::MAX_MRT_NUM; ++i) {
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
