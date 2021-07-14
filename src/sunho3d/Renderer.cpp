#include "Renderer.h"

#include <sunho3d/framegraph/shaders/ForwardPhongFrag.h>
#include <sunho3d/framegraph/shaders/ForwardPhongVert.h>
#include <tiny_gltf.h>

#include "Entity.h"
#include "Scene.h"

using namespace sunho3d;

Renderer::Renderer(Window *window, Scene *scene)
    : scene(scene), window(window), driver(window) {
}

Renderer::~Renderer() {
}

void Renderer::run() {
    for (auto entry : scene->entities) {
        for (auto &prim : entry->primitives) {
            auto tex = driver.createTexture(
                SamplerType::SAMPLER2D, TextureUsage::UPLOADABLE | TextureUsage::SAMPLEABLE,
                TextureFormat::RGBA8, prim.material.width, prim.material.height);
            BufferDescriptor td;
            td.data = (uint32_t *)prim.material.diffuseImage.data();
            driver.updateTexture(tex, td);
            textures.push_back(tex);
            auto p = driver.createPrimitive(prim.mode);
            auto vertexBuffer = driver.createVertexBuffer(
                prim.vertexBuffers.size(), prim.elementCount, prim.attributeCount, prim.attibutes);
            for (int i = 0; i < prim.vertexBuffers.size(); ++i) {
                auto buffer = driver.createBufferObject(prim.vertexBuffers[i].size());
                BufferDescriptor vb = { .data = (uint32_t *)prim.vertexBuffers[i].data() };
                driver.updateBufferObject(buffer, vb, 0);
                driver.setVertexBuffer(vertexBuffer, i, buffer);
            }
            auto indexBuffer = driver.createIndexBuffer(prim.elementCount);
            BufferDescriptor ib = { .data = (uint32_t *)prim.indexBuffer.data() };
            driver.updateIndexBuffer(indexBuffer, ib, 0);
            driver.setPrimitiveBuffer(p, vertexBuffer, indexBuffer);
            primitives.push_back(p);
        }
    }
    bb.model =
        glm::rotate(glm::identity<glm::mat4>(), glm::radians(180.0f), glm::vec3(0.0, 1.0, 0.0));
    bb.view = glm::lookAt(glm::vec3(0.0f, 0.9f, 3.0f), glm::vec3(0.0f, 0.9f, 0.0f),
                          glm::vec3(0.0f, -1.0f, 0.0f));
    bb.proj = glm::ortho(-1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 100.0f);
    auto ubo = driver.createUniformBuffer(sizeof(UniformBufferObject));
    BufferDescriptor ub = { .data = (uint32_t *)&bb };
    driver.updateUniformBuffer(ubo, ub, 0);
    auto renderTarget = driver.createDefaultRenderTarget();
    RenderPassParams params;
    Program prog;
    prog.codes[0] = std::vector<char>(ForwardPhongVert, ForwardPhongVert + ForwardPhongVertSize);
    prog.codes[1] = std::vector<char>(ForwardPhongFrag, ForwardPhongFrag + ForwardPhongFragSize);
    auto vv = driver.createProgram(prog);
    window->run([&, renderTarget, params, vv, ubo]() {
        driver.beginRenderPass(renderTarget, params);
        bb.model =
            glm::rotate(glm::identity<glm::mat4>(), glm::radians(10.f), glm::vec3(0.0, 1.0, 0.0)) *
            bb.model;
        BufferDescriptor ub = { .data = (uint32_t *)&bb };
        driver.updateUniformBuffer(ubo, ub, 0);
        driver.bindUniformBuffer(0, ubo);
        PipelineState pipe;
        pipe.program = vv;
        for (int i = 0; i < primitives.size(); ++i) {
            driver.bindTexture(0, textures[i]);
            driver.draw(pipe, primitives[i]);
        }
        driver.endRenderPass();
        driver.commit();
    });
}
