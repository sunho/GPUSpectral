#pragma once

#include "Window.h"
#include "backend/vulkan/VulkanDriver.h"
#include "utils/ResourceList.h"
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace sunho3d {
struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

class Scene;
class Renderer : public IdResource {
  public:
    Renderer(Window *window, Scene *scene);
    ~Renderer();

    void run();

  private:
    VulkanDriver driver;
    Window *window;
    Scene *scene;
    std::vector<Handle<HwPrimitive>> primitives;
    std::vector<Handle<HwTexture>> textures;
    UniformBufferObject bb;
};

}  // namespace sunho3d
