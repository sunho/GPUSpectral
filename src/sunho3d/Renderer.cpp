#include "Renderer.h"

#include <tiny_gltf.h>
#include <sunho3d/shaders/triangle_vert.h>
#include <sunho3d/shaders/triangle_frag.h>

#include "Scene.h"
#include "Entity.h"
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/gtc/matrix_transform.hpp>
#include <glm/glm.hpp>

using namespace sunho3d;

Renderer::Renderer(uint32_t id, Window* window, Scene* scene) : IdResource(id), scene(scene), window(window), driver(window) {
}

Renderer::~Renderer() {
}

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

void Renderer::run() {
    for (auto entry : scene->entities) {
        for (auto& prim : entry->primitives) {
            auto tex = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::UPLOADABLE | TextureUsage::SAMPLEABLE, TextureFormat::RGBA8, prim.material.width, prim.material.height);
            BufferDescriptor td;
            td.data = (uint32_t*)prim.material.diffuseImage.data();
            driver.updateTexture(tex, td);
            textures.push_back(tex);
            auto p = driver.createPrimitive(prim.mode);
            auto vertexBuffer = driver.createVertexBuffer(prim.vertexBuffers.size(), prim.elementCount, prim.attributeCount, prim.attibutes);
            for (int i = 0; i < prim.vertexBuffers.size(); ++i) {
                auto buffer = driver.createBufferObject(prim.vertexBuffers[i].size());
                BufferDescriptor vb = {.data = (uint32_t*)prim.vertexBuffers[i].data()};
                driver.updateBufferObject(buffer, vb, 0);
                driver.setVertexBuffer(vertexBuffer, i, buffer);
            }
            auto indexBuffer = driver.createIndexBuffer(prim.elementCount);
            BufferDescriptor ib = {.data = (uint32_t*)prim.indexBuffer.data()};
            driver.updateIndexBuffer(indexBuffer, ib, 0);
            driver.setPrimitiveBuffer(p, vertexBuffer, indexBuffer);
            primitives.push_back(p);
        }
    }
    UniformBufferObject bb;
    bb.model = glm::rotate(glm::identity<glm::mat4>(), glm::radians(180.0f), glm::vec3(0.0,1.0,0.0));
    bb.view = glm::lookAt(glm::vec3(0.0f, 0.9f, 3.0f), glm::vec3(0.0f, 0.9f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
    bb.proj = glm::ortho(-1.0f,1.0f,-1.0f,1.0f,0.0f,100.0f);
    auto ubo = driver.createUniformBuffer(sizeof(UniformBufferObject));
    BufferDescriptor ub = {.data=(uint32_t*)&bb};
    driver.updateUniformBuffer(ubo, ub, 0);
    auto renderTarget = driver.createDefaultRenderTarget();
    RenderPassParams params;
   Program prog;
   prog.codes[0] = std::vector<char>(triangle_vert, triangle_vert+triangle_vert_len);
   prog.codes[1] = std::vector<char>(triangle_frag, triangle_frag+triangle_frag_len);
   auto vv = driver.createProgram(prog);
    window->run([&, renderTarget, params,vv, ubo](){
        driver.beginRenderPass(renderTarget, params);
        driver.bindUniformBuffer(0, ubo);
        PipelineState pipe;
        pipe.program = vv;
        for (int i =0; i< primitives.size(); ++i) {
            driver.bindTexture(0, textures[i]);
            driver.draw(pipe, primitives[i]);
        }
        driver.endRenderPass();
        driver.commit();
    });
}

