#pragma once

#include <string>

namespace tinygltf {
class Model;
class Node;
class Mesh;
};  // namespace tinygltf

namespace sunho3d {
class Scene;
class Engine;
class Entity;
class Renderer;
class Loader {
  public:
    explicit Loader(Engine &engine, Renderer &renderer);
    Scene *loadGLTF(const std::string &path);
    Entity *loadObj(const std::string &path);

  private:
    void loadGLTFNode(Scene *scene, tinygltf::Model &model, tinygltf::Node &node,
                      Entity *parent = nullptr);
    void loadGLTFMesh(Entity *entity, tinygltf::Model &model, tinygltf::Mesh &mesh);
    Engine &engine;
    Renderer &renderer;
};

}  // namespace sunho3d
