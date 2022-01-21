#pragma once

#include <GPUSpectral/backend/Handles.h>
#include <filesystem>
#include <map>
#include <string>

namespace tinygltf {
class Model;
class Node;
class Mesh;
};  // namespace tinygltf

namespace pugi {
class xml_node;
};

namespace tinyparser_mitsuba {
class Object;
};

namespace GPUSpectral {
class Scene;
class Engine;
class Entity;
class Mesh;
class Renderer;
struct Material;
Scene loadScene(Engine& engine, Renderer& renderer, const std::string& path);
}  // namespace GPUSpectral
