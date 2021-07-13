#include "VulkanPipelineCache.h"

void VulkanPipelineCache::init(VulkanContext& context) {
     dummyBufferWriteInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
     dummyBufferWriteInfo.pNext = nullptr;
     dummyBufferWriteInfo.dstArrayElement = 0;
     dummyBufferWriteInfo.descriptorCount = 1;
     dummyBufferWriteInfo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
     dummyBufferWriteInfo.pImageInfo = nullptr;
     dummyBufferWriteInfo.pBufferInfo = &dummyBufferInfo;
     dummyBufferWriteInfo.pTexelBufferView = nullptr;

     dummySamplerWriteInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
     dummySamplerWriteInfo.pNext = nullptr;
     dummySamplerWriteInfo.dstArrayElement = 0;
     dummySamplerWriteInfo.descriptorCount = 1;
     dummySamplerWriteInfo.pImageInfo = &dummySamplerInfo;
     dummySamplerWriteInfo.pBufferInfo = nullptr;
     dummySamplerWriteInfo.pTexelBufferView = nullptr;
     dummySamplerWriteInfo.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

     dummyTargetInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
     dummyTargetWriteInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
     dummyTargetWriteInfo.pNext = nullptr;
     dummyTargetWriteInfo.dstArrayElement = 0;
     dummyTargetWriteInfo.descriptorCount = 1;
     dummyTargetWriteInfo.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
     dummyTargetWriteInfo.pImageInfo = &dummyTargetInfo;
     dummyTargetWriteInfo.pBufferInfo = nullptr;
     dummyTargetWriteInfo.pTexelBufferView = nullptr;
    setupDescriptorLayout(context);
}

void VulkanPipelineCache::setupDescriptorLayout(VulkanContext &context) {
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

    pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
    pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional
       
    VkDescriptorSetLayoutBinding binding = {};
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS;

    std::array<VkDescriptorSetLayoutBinding, UBUFFER_BINDING_COUNT> ubindings;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    for (uint32_t i = 0; i < UBUFFER_BINDING_COUNT; i++) {
       binding.binding = i;
       ubindings[i] = binding;
    }
    VkDescriptorSetLayoutCreateInfo dlinfo = {};
    dlinfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlinfo.bindingCount = ubindings.size();
    dlinfo.pBindings = ubindings.data();
    vkCreateDescriptorSetLayout(context.device, &dlinfo, nullptr, &descriptorSetLayout[0]);
    
    std::array<VkDescriptorSetLayoutBinding, SAMPLER_BINDING_COUNT> sbindings;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    for (uint32_t i = 0; i < SAMPLER_BINDING_COUNT; i++) {
      binding.binding = i;
      sbindings[i] = binding;
    }
    dlinfo.bindingCount = sbindings.size();
    dlinfo.pBindings = sbindings.data();
    vkCreateDescriptorSetLayout(context.device, &dlinfo, nullptr, &descriptorSetLayout[1]);

    std::array<VkDescriptorSetLayoutBinding, TARGET_BINDING_COUNT> bindings;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
      binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    for (uint32_t i = 0; i < TARGET_BINDING_COUNT; i++) {
      binding.binding = i;
      bindings[i] = binding;
    }
    dlinfo.bindingCount = bindings.size();
    dlinfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(context.device, &dlinfo, nullptr, &descriptorSetLayout[2]);
    
    pipelineLayoutInfo.setLayoutCount = 3;
    pipelineLayoutInfo.pSetLayouts = descriptorSetLayout.data();

    if (vkCreatePipelineLayout(context.device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }
    VkDescriptorPoolSize poolSizes[3] = {};
    VkDescriptorPoolCreateInfo poolInfo {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 1024 * 3,
        .poolSizeCount = 3,
        .pPoolSizes = poolSizes
    };
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = poolInfo.maxSets * UBUFFER_BINDING_COUNT;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = poolInfo.maxSets * SAMPLER_BINDING_COUNT;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    poolSizes[2].descriptorCount = poolInfo.maxSets * TARGET_BINDING_COUNT;
    if (vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }
}


void VulkanPipelineCache::getOrCreateDescriptors(VulkanContext& context, const VulkanDescriptorKey& key, std::array<VkDescriptorSet, 3>& descriptors) {
    auto it = descriptorSets.find(key);
    if (it != descriptorSets.end()) {
        descriptors = it->second;
        return;
    }
    
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 3;
    allocInfo.pSetLayouts = descriptorSetLayout.data();
    if (vkAllocateDescriptorSets(context.device, &allocInfo, descriptors.data()) !=VK_SUCCESS) {
         throw std::runtime_error("failed to allocate descriptor sets!");
    }
    descriptorSets.emplace(key, descriptors);
}

void VulkanPipelineCache::bindDescriptors(VulkanContext& context, const VulkanDescriptorKey& key) {
    std::array<VkDescriptorSet, 3> descriptors;
    getOrCreateDescriptors(context, key, descriptors);
    
    dummySamplerInfo.imageLayout = dummyTargetInfo.imageLayout;
    dummySamplerInfo.imageView = dummyImageView;
    dummySamplerInfo.imageView = dummyImageView;

    if (dummySamplerInfo.sampler == VK_NULL_HANDLE) {
       VkSamplerCreateInfo samplerInfo {
           .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
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
           .unnormalizedCoordinates = VK_FALSE
       };
       vkCreateSampler(context.device, &samplerInfo, nullptr, &dummySamplerInfo.sampler);
    }

       VkDescriptorBufferInfo descriptorBuffers[UBUFFER_BINDING_COUNT];
       VkDescriptorImageInfo descriptorSamplers[SAMPLER_BINDING_COUNT];
       VkDescriptorImageInfo descriptorInputAttachments[TARGET_BINDING_COUNT];
       VkWriteDescriptorSet descriptorWrites[UBUFFER_BINDING_COUNT + SAMPLER_BINDING_COUNT +
               TARGET_BINDING_COUNT];
       uint32_t nwrites = 0;
       VkWriteDescriptorSet* writes = descriptorWrites;
       nwrites = 0;
       for (uint32_t binding = 0; binding < UBUFFER_BINDING_COUNT; binding++) {
           VkWriteDescriptorSet& writeInfo = writes[nwrites++];
           if (key.uniformBuffers[binding]) {
               VkDescriptorBufferInfo& bufferInfo = descriptorBuffers[binding];
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
       for (uint32_t binding = 0; binding < SAMPLER_BINDING_COUNT; binding++) {
           VkWriteDescriptorSet& writeInfo = writes[nwrites++];
           if (key.samplers[binding].sampler) {
               VkDescriptorImageInfo& imageInfo = descriptorSamplers[binding];
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
       for (uint32_t binding = 0; binding < TARGET_BINDING_COUNT; binding++) {
           VkWriteDescriptorSet& writeInfo = writes[nwrites++];
           if (key.inputAttachments[binding].imageView) {
               VkDescriptorImageInfo& imageInfo = descriptorInputAttachments[binding];
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
    vkUpdateDescriptorSets(context.device, nwrites, writes, 0, nullptr);
    vkCmdBindDescriptorSets(context.commands.get(), VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 3, descriptors.data(), 0, nullptr);
}

VkPipeline VulkanPipelineCache::getOrCreatePipeline(VulkanContext& context, const VulkanPipelineKey& key) {
    auto it = pipelines.find(key);
    if (it != pipelines.end()) {
        return it->second;
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
      VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};
      
      VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
      vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = key.bindings.size();
    vertexInputInfo.pVertexBindingDescriptions = key.bindings.data();
    vertexInputInfo.vertexAttributeDescriptionCount = key.attributes.size();
    vertexInputInfo.pVertexAttributeDescriptions = key.attributes.data();
      
      VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
      inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
      inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      inputAssembly.primitiveRestartEnable = VK_FALSE;

      VkViewport viewport{};
      viewport.x = 0.0f;
      viewport.y = 0.0f;
    viewport.width = (float)  key.viewport.width;
    viewport.height = (float) key.viewport.height;
      viewport.minDepth = 0.0f;
      viewport.maxDepth = 1.0f;
      
      VkRect2D scissor{};
      scissor.offset = {0, 0};
    scissor.extent = {.width = key.viewport.width, .height = key.viewport.height};
      
      VkPipelineViewportStateCreateInfo viewportState{};
      viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
      viewportState.viewportCount = 1;
      viewportState.pViewports = &viewport;
      viewportState.scissorCount = 1;
      viewportState.pScissors = &scissor;
      
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f; // Optional
    depthStencil.maxDepthBounds = 1.0f; // Optional
    depthStencil.stencilTestEnable = VK_FALSE;
    depthStencil.front = {}; // Optional
    depthStencil.back = {}; // Optional

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
      
      
      VkGraphicsPipelineCreateInfo pipelineInfo{};
      pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
      pipelineInfo.stageCount = 2;
      pipelineInfo.pStages = shaderStages;
      pipelineInfo.pVertexInputState = &vertexInputInfo;
      pipelineInfo.pInputAssemblyState = &inputAssembly;
      pipelineInfo.pViewportState = &viewportState;
      pipelineInfo.pRasterizationState = &rasterizer;
      pipelineInfo.pMultisampleState = &multisampling;
      pipelineInfo.pDepthStencilState = &depthStencil; // Optional
      pipelineInfo.pColorBlendState = &colorBlending;
      pipelineInfo.pDynamicState = nullptr; // Optional
      pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = context.currentRenderPass;
      pipelineInfo.subpass = 0;
      pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
      pipelineInfo.basePipelineIndex = -1; // Optional
    VkPipeline out;
      if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &out) != VK_SUCCESS) {
          throw std::runtime_error("failed to create graphics pipeline!");
      }
    
    pipelines.emplace(key, out);
        return out;
}


VkRenderPass VulkanPipelineCache::getOrCreateRenderPass(VulkanContext& context, VulkanRenderTarget* renderTarget) {
    auto it = renderpasses.find(renderTarget);
    if (it != renderpasses.end()) {
        return it->second;
    }
    VkRenderPass out;
    VkAttachmentDescription colorAttachment{
        .format = context.surface->format.format,
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
    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = VK_FORMAT_D32_SFLOAT;
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    
    VkSubpassDescription subpass{
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachmentRef,
        .pDepthStencilAttachment = &depthAttachmentRef
    };

    std::array<VkAttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};

    VkRenderPassCreateInfo renderPassInfo{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = attachments.size(),
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
    };
    
    if (vkCreateRenderPass(context.device, &renderPassInfo, nullptr, &out) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }

    renderpasses.emplace(renderTarget, out);
    return out;
}

VkFramebuffer VulkanPipelineCache::getOrCreateFrameBuffer(VulkanContext& context, VkRenderPass renderPass, VulkanRenderTarget* renderTarget) {
    auto it = framebuffers.find(std::make_pair(renderTarget, context.currentSwapContext->attachment.view));
    if (it != framebuffers.end()) {
        return it->second;
    }
    VkImageView attachments[] = {
        context.currentSwapContext->attachment.view,
        renderTarget->depth.view
    };

    VkFramebuffer framebuffer;
    VkFramebufferCreateInfo framebufferInfo{
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .renderPass = renderPass,
        .attachmentCount = 2,
        .pAttachments = attachments,
        .width = renderTarget->width,
        .height = renderTarget->height,
        .layers = 1
    };

    if (vkCreateFramebuffer(context.device, &framebufferInfo, nullptr, &framebuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create framebuffer!");
    }
    
    
    framebuffers.emplace(std::make_pair(renderTarget, context.currentSwapContext->attachment.view), framebuffer);
    return framebuffer;
}
