#pragma once

#include <string>
#include <filesystem>
#include <map>
#include <VKGIRenderer/backend/Handles.h>

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

namespace VKGIRenderer {
class Scene;
class Engine;
class Entity;
class Mesh;
class Renderer;
struct Material;

class Loader {
  public:
    explicit Loader(Engine &engine, Renderer &renderer);
    Scene *loadGLTF(const std::string &path);
    Scene *loadMitsuba(const std::string &path);
    Mesh *loadObj(const std::string &path, bool twosided=false);
    Handle<HwTexture> loadTexture(const std::string &path);
    

  private:
    Mesh *loadOrGetMesh(const std::string &path);
    Mesh *createQuad();
    Handle<HwTexture> loadOrGetTexture(const std::string &path);
    void loadGLTFNode(Scene *scene, tinygltf::Model &model, tinygltf::Node &node,
                      Entity *parent = nullptr);
    void loadGLTFMesh(Entity *entity, tinygltf::Model &model, tinygltf::Mesh &mesh);
    void loadMaterial(Material *material, tinyparser_mitsuba::Object &obj, const std::filesystem::path &basepath);
    Engine &engine;
    std::map<std::string, Mesh *> meshCache;
    std::map<std::string, Handle<HwTexture>> textureCache;
    Renderer &renderer;
};

}  // namespace VKGIRenderer
