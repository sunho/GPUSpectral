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

using MeshPtr = std::shared_ptr<Mesh>;
class Engine {
  public:
    Engine(const std::filesystem::path& basePath, const std::filesystem::path& assetBasePath);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    Engine(Engine&&) = delete;
    Engine& operator=(Engine&&) = delete;

    void init(size_t width, size_t height);

    MeshPtr createMesh(const std::span<Mesh::Vertex> vertices, const std::span<uint32_t>& indices);

    [[nodiscard]] std::string assetPath(const std::string& assetName) const noexcept;

    [[nodiscard]] std::filesystem::path getBasePath() const noexcept;

    [[nodiscard]] const Renderer& getRenderer() const noexcept;

    [[nodiscard]] const Window& getWindow() const noexcept;

    [[nodiscard]] Renderer& getRenderer() noexcept;

    [[nodiscard]] Window& getWindow() noexcept;

   private:
    std::unique_ptr<Window> window;
    std::unique_ptr<Renderer> renderer;
    
    std::filesystem::path basePath;
    std::filesystem::path assetBasePath;

    uint32_t nextMeshId{ 1 };
};

}  // namespace GPUSpectral
