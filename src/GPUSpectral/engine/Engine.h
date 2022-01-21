#pragma once

#include <stdlib.h>

#include <filesystem>
#include <list>
#include <memory>

#include "../renderer/Mesh.h"
#include "../renderer/Renderer.h"
#include "Window.h"

namespace GPUSpectral {

class Engine {
  public:
    Engine(const std::filesystem::path& basePath, const std::filesystem::path& assetBasePath);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    Engine(Engine&&) = delete;
    Engine& operator=(Engine&&) = delete;

    void init(size_t width, size_t height);

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
};

}  // namespace GPUSpectral
