#include "Renderer.h"

#include <sunho3d/framegraph/shaders/DisplayTextureFrag.h>
#include <sunho3d/framegraph/shaders/DisplayTextureVert.h>
#include <sunho3d/framegraph/shaders/ForwardPhongFrag.h>
#include <sunho3d/framegraph/shaders/ForwardPhongVert.h>
#include <tiny_gltf.h>

#include "Entity.h"
#include "Scene.h"

using namespace sunho3d;

RenderGraph::RenderGraph(Renderer& renderer)
    : parent(renderer), driver(renderer.getDriver()) {
}

RenderGraph::~RenderGraph() {
    for (auto d : destroyers) {
        d();
    }
}
void RenderGraph::reset() {
    for (auto d : destroyers) {
        d();
    }
    destroyers.clear();
    fg = FrameGraph();
}

void RenderGraph::addRenderPass(const std::string& name, std::vector<FGResource> inputs, std::vector<FGResource> outputs, RenderPass::RenderPassFunc func) {
    fg.addRenderPass(RenderPass(name, inputs, outputs, func));
}

void RenderGraph::submit() {
    fg.compile();
    fg.run();
}

Renderer::Renderer(Window* window)
    : window(window), driver(window) {
    for (size_t i = 0; i < VULKAN_COMMANDS_SIZE; ++i) {
        renderGraphs.push_back(RenderGraph(*this));
    }

    Program prog;
    prog.codes[0] = std::vector<char>(ForwardPhongVert, ForwardPhongVert + ForwardPhongVertSize);
    prog.codes[1] = std::vector<char>(ForwardPhongFrag, ForwardPhongFrag + ForwardPhongFragSize);
    fowradPassProgram = driver.createProgram(prog);
    surfaceRenderTarget = driver.createDefaultRenderTarget();

    Program prog2;
    prog2.codes[0] = std::vector<char>(DisplayTextureVert, DisplayTextureVert + DisplayTextureVertSize);
    prog2.codes[1] = std::vector<char>(DisplayTextureFrag, DisplayTextureFrag + DisplayTextureFragSize);
    quadDrawProgram = driver.createProgram(prog2);

    Primitive primitive;
    std::vector<float> v = {
        -1,
        -1,
        1,
        1,
        -1,
        1,
        1,
        -1,
        1,
        1,
        -1,
        -1,
    };
    std::vector<uint16_t> indices = { 0, 1, 2, 3, 4, 5 };
    primitive.attibutes[0] = {
        .name = "position", .offset = 0, .index = 0, .type = ElementType::FLOAT2, .stride = 8
    };
    auto buffer0 = driver.createBufferObject(4 * v.size(), BufferUsage::VERTEX);
    driver.updateBufferObject(buffer0, { .data = (uint32_t*)v.data() }, 0);
    auto vbo = driver.createVertexBuffer(1, 6, 1, primitive.attibutes);
    driver.setVertexBuffer(vbo, 0, buffer0);

    auto ibo = driver.createIndexBuffer(indices.size());
    driver.updateIndexBuffer(ibo, { .data = (uint32_t*)indices.data() }, 0);
    primitive.indexBuffer = ibo;
    primitive.vertexBuffer = vbo;
    quadPrimitive = driver.createPrimitive(PrimitiveMode::TRIANGLES);
    driver.setPrimitiveBuffer(quadPrimitive, vbo, ibo);
}

Renderer::~Renderer() {
}

void Renderer::run(Scene* scene) {
    const uint32_t cmdIndex = driver.acquireCommandBuffer();
    RenderGraph& renderGraph = renderGraphs[cmdIndex];
    renderGraph.reset();

    scene->prepare();
    SceneData& sceneData = scene->getSceneData();

    auto sceneDataRef = renderGraph.declareResource<SceneData*>("SceneData");
    renderGraph.defineResource(sceneDataRef, &sceneData);

    auto lightUB = renderGraph.createUniformBuffer(sizeof(LightBuffer));
    driver.updateUniformBuffer(lightUB, { .data = (uint32_t*)&sceneData.lightBuffer }, 0);

    renderGraph.addRenderPass("forward", { sceneDataRef }, {}, [this, scene, lightUB, sceneDataRef, &renderGraph](FrameGraph& fg) {
        auto color = renderGraph.createTexture(SamplerType::SAMPLER2D, TextureUsage::COLOR_ATTACHMENT, TextureFormat::RGBA8, 1200u, 1200u);
        auto depth = renderGraph.createTexture(SamplerType::SAMPLER2D, TextureUsage::DEPTH_ATTACHMENT, TextureFormat::DEPTH32F, 1200u, 1200u);
        ColorAttachment att = {};
        att.colors[0] = {
            .handle = color
        };
        att.targetNum = 1;
        auto renderTarget = renderGraph.createRenderTarget(1200u, 1200u, att, TextureAttachment{ .handle = depth });
        std::map<Material*, Handle<HwUniformBuffer>> materialBuffers;

        SceneData* sceneData = fg.getResource<SceneData*>(sceneDataRef);
        auto& camera = scene->getCamera();
        glm::mat4 vp = camera.proj * camera.view;

        const auto crateTransformBuffer = [&](glm::mat4 model) {
            auto tb = renderGraph.createUniformBuffer(sizeof(TransformBuffer));
            TransformBuffer transformBuffer;
            transformBuffer.MVP = vp * model;
            transformBuffer.invModelT = glm::transpose(glm::inverse(model));
            transformBuffer.model = model;
            transformBuffer.cameraPos = glm::vec4(camera.pos(), 1.0f);
            driver.updateUniformBuffer(tb, { .data = (uint32_t*)&transformBuffer }, 0);
            return tb;
        };

        const auto getOrCreateMaterialBuffer = [&](Material* material) {
            auto it = materialBuffers.find(material);
            if (it != materialBuffers.end()) {
                return it->second;
            }
            MaterialBuffer materialBuffer;
            materialBuffer.phong = 100;
            materialBuffer.specular = glm::vec4(0.2, 0.2, 0.2, 1.0);
            auto mb = renderGraph.createUniformBuffer(sizeof(MaterialBuffer));
            driver.updateUniformBuffer(mb, { .data = (uint32_t*)&materialBuffer }, 0);
            materialBuffers.emplace(material, mb);
            return mb;
        };

        PipelineState pipe;
        pipe.program = this->fowradPassProgram;
        RenderPassParams params;
        driver.beginRenderPass(renderTarget, params);
        for (size_t i = 0; i < sceneData->geometries.size(); ++i) {
            auto& geom = sceneData->geometries[i];
            auto& model = sceneData->worldTransforms[i];
            driver.bindUniformBuffer(0, crateTransformBuffer(model));
            driver.bindUniformBuffer(1, lightUB);
            driver.bindUniformBuffer(2, getOrCreateMaterialBuffer(geom.material));
            driver.bindTexture(0, geom.material->diffuseMap);
            driver.draw(pipe, geom.primitive);
        }
        driver.endRenderPass();
        pipe.program = this->quadDrawProgram;

        driver.beginRenderPass(this->surfaceRenderTarget, params);
        driver.bindTexture(0, color);
        driver.draw(pipe, quadPrimitive);
        driver.endRenderPass();
        driver.commit();
    });
    renderGraph.submit();
}
