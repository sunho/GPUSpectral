#include "PathTracer.h"

using namespace GPUSpectral;

void PathTracer::setup() {
    accumulateBuffer = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::STORAGE | TextureUsage::SAMPLEABLE, TextureFormat::RGBA32F, 1, driver.getFrameSize().width, driver.getFrameSize().height, 1);
}

void PathTracer::createRenderPass(FrameGraph &fg, const Scene& scene) {
    std::unordered_map<uint32_t, uint32_t> primitiveIdToVB;
    std::vector<RTInstance> instances;
    for (auto& obj: scene.renderObjects) {
        RTInstance instance;
        instance.blas = renderer.getOrCreateBLAS(obj.mesh);
        instance.transfom = obj.transform;
        instances.push_back(instance);
    }
    auto tlas = driver.createTLAS({ .instances=instances.data(), .count = (uint32_t)instances.size()});
    fg.queueDispose([tlas, this]() { driver.destroyTLAS(tlas); });

    prepareScene(fg, scene);
    auto rsBuf = fg.createTempUniformBuffer(&renderState, sizeof(RenderState));

    fg.addFramePass({
        .textures = {
            {{accumulateBuffer}, ResourceAccessType::RTWrite},
        },
        .func = [this, tlas, rsBuf](FrameGraph& rg) {
            RTPipeline pipeline = {};
            pipeline.raygenGroup = renderer.getShaderProgram("raygen.rgen");
            pipeline.missGroups.push_back(renderer.getShaderProgram("miss.rmiss"));
            pipeline.missGroups.push_back(renderer.getShaderProgram("shadowmiss.rmiss"));
            pipeline.hitGroups.push_back(renderer.getShaderProgram("rayhit.rchit"));
            pipeline.bindTLAS(0, 0, tlas);
            pipeline.bindStorageImage(0, 1, accumulateBuffer);
            pipeline.bindUniformBuffer(0, 2, rsBuf);
            driver.traceRays(pipeline, driver.getFrameSize().width, driver.getFrameSize().height);
        },
    });

    fg.addFramePass({
        .textures = {
            {{accumulateBuffer}, ResourceAccessType::FragmentRead},
        },
        .func = [this, tlas](FrameGraph& rg) {
            GraphicsPipeline pipe = {};
            pipe.vertex = renderer.getShaderProgram("DrawTexture.vert");
            pipe.fragment= renderer.getShaderProgram("DrawTexture.frag");
            RenderPassParams params;
            driver.beginRenderPass(renderer.getSurfaceRenderTarget(), params);
            pipe.bindTexture(0, 0, accumulateBuffer);
            driver.draw(pipe, renderer.getQuadPrimitive());
            driver.endRenderPass();
        },
    });

}

void PathTracer::prepareScene(FrameGraph& rg, const Scene& scene) {
    std::vector<RenderState::Instance> rinstances;
    for (auto& obj : scene.renderObjects) {
        RenderState::Instance ri;
        ri.transformInvT = glm::inverse(glm::transpose(obj.transform));
        ri.positionBuffer = driver.getDeviceAddress(obj.mesh->getPositionBuffer());
        ri.normalBuffer = driver.getDeviceAddress(obj.mesh->getNormalBuffer());
        const Material& material = scene.getMaterial(obj.material);
        ri.emission = glm::vec4(material.emission,0.0);
        ri.bsdf = material.bsdf;
        ri.twofaced = material.twofaced;
        rinstances.push_back(ri);
    }
    
    int size = sizeof(RenderState::Instance);
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
    renderState.scene.triangleLights = driver.getDeviceAddress(lightBuffer);
    renderState.scene.numLights = scene.triangleLights.size();
    renderState.camera.eye = scene.camera.getPosition();
    renderState.camera.view = scene.camera.getToWorld();
    renderState.camera.fov = scene.camera.getFov();
    renderState.params.timestamp = timestamp;
    timestamp++;

}
