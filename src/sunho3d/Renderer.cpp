#include "Renderer.h"

#include <tiny_gltf.h>
#include <sunho3d/shaders/triangle_vert.h>
#include <sunho3d/shaders/triangle_frag.h>

#include "Scene.h"
#include "Entity.h"
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
    bb.view =  glm::lookAt(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));;
    bb.proj =  glm::ortho(-10.0f,10.0f,-10.0f,10.0f,0.0f,100.0f);
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
        for (auto p : primitives) {
            driver.draw(pipe, p);
        }
        driver.endRenderPass();
        driver.commit();
    });
}

