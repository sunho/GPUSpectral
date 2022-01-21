#include "Engine.h"

using namespace GPUSpectral;

Engine::Engine(const std::filesystem::path& basePath, const std::filesystem::path& assetBasePath)
    : assetBasePath(assetBasePath), basePath(basePath) {
}

Engine::~Engine() {
}

std::string GPUSpectral::Engine::assetPath(const std::string& assetName) const noexcept {
    auto path = assetBasePath / assetName;
    return path.string();
}

std::filesystem::path GPUSpectral::Engine::getBasePath() const noexcept {
    return assetBasePath;
}

const Renderer& GPUSpectral::Engine::getRenderer() const noexcept {
    return *renderer;
}

const Window& GPUSpectral::Engine::getWindow() const noexcept {
    return *window;
}

Renderer& GPUSpectral::Engine::getRenderer() noexcept {
    return *renderer;
}

Window& GPUSpectral::Engine::getWindow() noexcept {
    return *window;
}

void GPUSpectral::Engine::init(size_t width, size_t height) {
    window = std::make_unique<Window>(width, height);
    renderer = std::make_unique<Renderer>(*this, window.get());
}
