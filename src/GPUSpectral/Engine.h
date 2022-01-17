#pragma once

#include <stdlib.h>

#include <list>
#include <memory>
#include <filesystem>

#include "Mesh.h"
#include "renderer/Renderer.h"
#include "Scene.h"
#include "Window.h"
#include "utils/ResourceList.h"

namespace GPUSpectral {
class Engine {
  public:
    Engine(const std::filesystem::path& basePath, const std::filesystem::path& assetBasePath);
    ~Engine();

    Window *createWindow(size_t width, size_t height);
    Mesh *createMesh();
    Renderer *createRenderer(Window *window);
    std::string assetPath(const std::string& assetName);
    std::filesystem::path getBasePath() { return basePath;  }



  private:
    ResourceList<Window> windows;
    ResourceList<Renderer> renderers;
    ResourceList<Mesh> meshes;
    
    std::filesystem::path basePath;
    std::filesystem::path assetBasePath;
};

}  // namespace GPUSpectral
