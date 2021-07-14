#include "Renderer.h"

#include <sunho3d/framegraph/FrameGraph.h>
#include <sunho3d/framegraph/shaders/ForwardPhongFrag.h>
#include <sunho3d/framegraph/shaders/ForwardPhongVert.h>
#include <tiny_gltf.h>

#include "Entity.h"
#include "Scene.h"

using namespace sunho3d;

Renderer::Renderer(Window* window)
    : window(window), driver(window) {
    Program prog;
    prog.codes[0] = std::vector<char>(ForwardPhongVert, ForwardPhongVert + ForwardPhongVertSize);
    prog.codes[1] = std::vector<char>(ForwardPhongFrag, ForwardPhongFrag + ForwardPhongFragSize);
    rpConf.fowradPassProgram = driver.createProgram(prog);
    renderTarget = driver.createDefaultRenderTarget();
}

Renderer::~Renderer() {
}

void Renderer::run(Scene* scene) {
    scene->prepare();
    SceneData& sceneData = scene->getSceneData();
    FrameGraph fg;

    auto sceneDataRef = fg.declareResource<SceneData*>("SceneData");
    fg.defineResource(sceneDataRef, &sceneData);
    auto lightUBufferRef = fg.declareResource<Handle<HwUniformBuffer>>("LightUBuffer");
    auto transformUBufferRef = fg.declareResource<Handle<HwUniformBuffer>>("TransformUBuffer");
    fg.addRenderPass(RenderPass("prepare buffers", { sceneDataRef }, { lightUBufferRef, transformUBufferRef }, [&](FrameGraph& fg) {
        SceneData* sceneData = fg.getResource<SceneData*>(sceneDataRef);
        RenderPassParams params;
        //        driver.beginRenderPass(renderTarget, params);
        auto lb = driver.createUniformBuffer(sizeof(LightBuffer));
        auto tb = driver.createUniformBuffer(sizeof(TransformBuffer));
        driver.updateUniformBuffer(lb, { .data = (uint32_t*)&sceneData->lightBuffer }, 0);
        //        driver.endRenderPass();
        fg.defineResource(lightUBufferRef, lb);
        fg.defineResource(transformUBufferRef, tb);
    }));
    fg.addRenderPass(RenderPass("forward", { sceneDataRef, lightUBufferRef, transformUBufferRef }, {}, [&](FrameGraph& fg) {
        SceneData* sceneData = fg.getResource<SceneData*>(sceneDataRef);
        auto& camera = scene->getCamera();
        glm::mat4 vp = camera.proj * camera.view;
        auto lb = fg.getResource<Handle<HwUniformBuffer>>(lightUBufferRef);
        auto tb = fg.getResource<Handle<HwUniformBuffer>>(transformUBufferRef);
        TransformBuffer transformBuffer;
        transformBuffer.MVP = vp * glm::identity<glm::mat4>();
        transformBuffer.invModelT = glm::transpose(glm::inverse(transformBuffer.MVP));
        transformBuffer.model = glm::identity<glm::mat4>();

        PipelineState pipe;
        pipe.program = rpConf.fowradPassProgram;
        RenderPassParams params;
        driver.beginRenderPass(renderTarget, params);
        driver.updateUniformBuffer(tb, { .data = (uint32_t*)&transformBuffer }, 0);
        driver.bindUniformBuffer(0, tb);
        driver.bindUniformBuffer(1, lb);
        for (size_t i = 0; i < sceneData->geometries.size(); ++i) {
            auto& geom = sceneData->geometries[i];
            auto& model = sceneData->worldTransforms[i];
            driver.bindTexture(0, geom.material->diffuseMap);
            driver.draw(pipe, geom.primitive);
        }
        driver.endRenderPass();
        driver.commit();
    }));
    fg.defineResource(sceneDataRef, &sceneData);
    fg.compile();
    fg.run();
}
