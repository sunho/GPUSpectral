#pragma once
#include <string>
#include "../renderer/Renderer.h"
#include "../renderer/Scene.h"

Scene loadScene(Renderer& renderer, const std::string& path);
TextureId loadTexture(Renderer& renderer, const std::string& path);
Mesh loadMesh(const std::string& path);
