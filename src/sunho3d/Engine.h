#pragma once

#include <stdlib.h>

#include <list>
#include <memory>
#include <filesystem>

#include "Entity.h"
#include "renderer/Renderer.h"
#include "Scene.h"
#include "Window.h"
#include "utils/ResourceList.h"

namespace sunho3d {
class Engine {
  public:
    Engine(const std::filesystem::path& assetBasePath);
    ~Engine();

    Window *createWindow(size_t width, size_t height);
    Entity *createEntity();
    Mesh *createMesh();
    Renderer *createRenderer(Window *window);
    Scene *createScene(Renderer *renderer);
    Material *createMaterial();
    std::string assetPath(const std::string& assetName);


  private:
    ResourceList<Window> windows;
    ResourceList<Renderer> renderers;
    ResourceList<Entity> entities;
    ResourceList<Scene> scenes;
    ResourceList<Material> materials;
    ResourceList<Mesh> meshes;

    std::filesystem::path assetBasePath;
};

}  // namespace sunho3d
