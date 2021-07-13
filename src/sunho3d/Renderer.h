#pragma once

#include "backend/vulkan/VulkanDriver.h"
#include "Window.h"
#include "ResourceList.h"

namespace sunho3d {

class Scene;
class Renderer : public IdResource {
public:
    Renderer(uint32_t id, Window* window, Scene* scene);
    ~Renderer();
    
    void run();
private:
    VulkanDriver driver;
    Window* window;
    Scene* scene;
    std::vector<Handle<HwPrimitive>> primitives;
    std::vector<Handle<HwTexture>> textures;
};

}
