#pragma once
#include "Renderer.h"
#include "../Scene.h"

namespace VKGIRenderer {

using DevicePtr = uint64_t;

struct RenderState {
    struct Camera {
        alignas(16) glm::vec3 eye;
        alignas(16) glm::mat4 view;
        alignas(4) float fov;
        // int pad[3];
    };

    struct Scene {
        alignas(8) DevicePtr instances;
        alignas(8) DevicePtr triangleLights;
#define BSDFDefinition(BSDFNAME, BSDFFIELD, BSDFTYPE) alignas(8) DevicePtr BSDFFIELD##s;
#include "../assets/shaders/BSDF.inc"
#undef BSDFDefinition
        int numLights;
        int pad[3];
    };

    struct Instance {
        alignas(16) glm::mat4 transformInvT;
        alignas(8) uint64_t positionBuffer;
        alignas(8) uint64_t normalBuffer;
        alignas(16) glm::vec3 emission;
        alignas(4) BSDFHandle bsdf;
    };

    struct RenderParams {
        int spp;
        int toneMap;
        int nee;
        int timestamp;
    };

    alignas(16) Camera camera;
    alignas(16) Scene scene;
    alignas(16) RenderParams params;
};

class PathTracer : public RendererImpl {
public:
    PathTracer(Renderer& renderer) : renderer(renderer), driver(renderer.getDriver()) {
        setup();
    }
    ~PathTracer() { }

    void setup();
    void render(InflightContext& ctx, const Scene& scene) override;
private:
    void prepareScene(FrameGraph& rg, const Scene& scene);
    Handle<HwTexture> accumulateBuffer;

    RenderState renderState;
    int timestamp{ 0 };
    Renderer& renderer;
    VulkanDriver& driver;
};

}
