#pragma once
#include "Renderer.h"

namespace VKGIRenderer {

class PathTracer {
public:
    PathTracer(Renderer& renderer) : renderer(renderer) {}
    ~PathTracer() { }

    void setup();
    void render(InflightContext& ctx);
private:
    Handle<HwTexture> accumulateBuffer;

    Renderer& renderer;
};

}
