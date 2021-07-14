#pragma once

#include <stdlib.h>

#include <list>
#include <memory>

#include "Entity.h"
#include "Renderer.h"
#include "Scene.h"
#include "Window.h"
#include "utils/ResourceList.h"

namespace sunho3d {
class Engine {
  public:
    Engine();
    ~Engine();

    Window *createWindow(size_t width, size_t height);
    Entity *createEntity();
    Renderer *createRenderer(Window *window);
    Scene *createScene(Renderer *renderer);
    Material *createMaterial();

  private:
    ResourceList<Window> windows;
    ResourceList<Renderer> renderers;
    ResourceList<Entity> entities;
    ResourceList<Scene> scenes;
    ResourceList<Material> materials;
};

}  // namespace sunho3d
