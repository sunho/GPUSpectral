#pragma once

#include "Window.h"
#include "backend/vulkan/VulkanDriver.h"
#include "utils/ResourceList.h"

namespace sunho3d {

class Scene;
class Renderer : public IdResource {
  public:
    Renderer(Window *window);
    ~Renderer();

    VulkanDriver &getDriver() {
        return driver;
    }
    void run(Scene *scene);

  private:
    struct RenderPassConfig {
        Handle<HwProgram> fowradPassProgram;
    };

    RenderPassConfig rpConf;
    Handle<HwRenderTarget> renderTarget;
    VulkanDriver driver;
    Window *window;
};

}  // namespace sunho3d
