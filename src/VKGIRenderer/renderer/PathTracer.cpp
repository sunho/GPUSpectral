#include "PathTracer.h"

using namespace VKGIRenderer;

void PathTracer::setup() {
    accumulateBuffer = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::STORAGE | TextureUsage::SAMPLEABLE, TextureFormat::RGBA16F, 1, 2048, 2048, 1);
}

void PathTracer::render(InflightContext& ctx, const Scene& scene) {
    auto tlas = ctx.data->tlas;
    prepareScene(*ctx.rg, scene);
    auto rsBuf = ctx.rg->createTempUniformBuffer(&renderState, sizeof(RenderState));

    ctx.rg->addFramePass({
        .textures = {
            {{accumulateBuffer}, ResourceAccessType::RTWrite},
        },
        .func = [this, tlas, rsBuf](FrameGraph& rg) {
            RTPipeline pipeline = {};
            pipeline.raygenGroup = renderer.getShaderProgram("RayGen");
            pipeline.missGroups.push_back(renderer.getShaderProgram("RayMiss"));
            pipeline.hitGroups.push_back(renderer.getShaderProgram("RayHit"));
            pipeline.bindTLAS(0, 0, tlas);
            pipeline.bindStorageImage(0, 1, accumulateBuffer);
            pipeline.bindUniformBuffer(0, 2, rsBuf);
            driver.traceRays(pipeline, 2048, 2048);
        },
    });

    ctx.rg->addFramePass({
        .textures = {
            {{accumulateBuffer}, ResourceAccessType::FragmentRead},
        },
        .func = [this, tlas](FrameGraph& rg) {
            GraphicsPipeline pipe = {};
            pipe.vertex = renderer.getShaderProgram("DrawTextureVert");
            pipe.fragment= renderer.getShaderProgram("DrawTextureFrag");
            RenderPassParams params;
            driver.beginRenderPass(renderer.surfaceRenderTarget, params);
            pipe.bindTexture(0, 0, accumulateBuffer);
            driver.draw(pipe, renderer.quadPrimitive);
            driver.endRenderPass();
        },
    });

}

void PathTracer::prepareScene(FrameGraph& rg, const Scene& scene) {
    std::vector<RenderState::Instance> rinstances;
    for (auto& obj : scene.renderObjects) {
        RenderState::Instance ri;
        ri.transformInvT = glm::inverse(glm::transpose(obj.transform));
        ri.positionBuffer = driver.getDeviceAddress(obj.mesh->positionBuffer);
        ri.normalBuffer = driver.getDeviceAddress(obj.mesh->normalBuffer);
        const Material& material = scene.getMaterial(obj.material);
        ri.emission = material.emission;
        ri.bsdf = material.bsdf;
        rinstances.push_back(ri);
    }
    
    auto instanceBuffer = rg.createTempStorageBuffer(rinstances.data(), rinstances.size() * sizeof(RenderState::Instance));
    renderState.scene.instances = driver.getDeviceAddress(instanceBuffer);
#define BSDFDefinition(BSDFNAME, BSDFFIELD, BSDFTYPE) {\
            if (!scene.BSDFFIELD##s.empty()) { \
                auto bsdfBuffer = rg.createTempStorageBuffer(scene.BSDFFIELD##s.data(), scene.BSDFFIELD##s.size() * sizeof(BSDFNAME)); \
                renderState.scene.BSDFFIELD##s = driver.getDeviceAddress(bsdfBuffer); \
            }\
        }
#include "../assets/shaders/BSDF.inc"
#undef BSDFDefinition

    auto lightBuffer = rg.createTempStorageBuffer(scene.triangleLights.data(), scene.triangleLights.size() * sizeof(TriangleLight));
    renderState.scene.triangleLights = lightBuffer;
    renderState.camera.eye = scene.camera.pos();
    renderState.camera.view = scene.camera.toWorld;
    renderState.camera.fov = scene.camera.fov;
    renderState.params.timestamp = timestamp;
    timestamp++;
}
