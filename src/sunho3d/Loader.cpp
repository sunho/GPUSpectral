#include "Loader.h"
#include "Engine.h"
#include "Scene.h"

#include <iostream>
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>
#include <stdexcept>

using namespace sunho3d;

Loader::Loader(Engine& engine) : engine(engine) {
}

Scene* Loader::loadGLTF(const std::string &path) {
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;
    tinygltf::Model model;
    bool res = loader.LoadASCIIFromFile(&model, &err, &warn, path.c_str());
    if (!warn.empty()) {
    std::cout << "WARN: " << warn << std::endl;
    }
    if (!err.empty()) {
    std::cout << "ERR: " << err << std::endl;
    }
    if (!res) {
      throw std::runtime_error("couldn't load model");
    }
    
    Scene* scene = engine.createScene();
    tinygltf::Scene &s = model.scenes[model.defaultScene];
    for (auto node : s.nodes) {
        loadGLTFNode(scene, model, model.nodes[node]);
    }
    return scene;
}

void Loader::loadGLTFNode(Scene* scene, tinygltf::Model &model, tinygltf::Node &node, Entity* parent) {
    if ((node.mesh >= 0) && (node.mesh < model.meshes.size())) {
        Entity* entity = engine.createEntity();
        loadGLTFMesh(entity, model, model.meshes[node.mesh]);
        if (parent) {
            parent->addNode(entity);
        } else {
            scene->addEntity(entity);
        }
        parent = entity;
    }
    
    for (size_t i = 0; i < node.children.size(); i++) {
        assert((node.children[i] >= 0) && (node.children[i] < model.nodes.size()));
        loadGLTFNode(scene, model, model.nodes[node.children[i]], parent);
    }
}

ElementType translateType(int type, int componentType) {
      if (componentType == TINYGLTF_COMPONENT_TYPE_BYTE) {
        if (type == TINYGLTF_TYPE_SCALAR) {
            return ElementType::BYTE;
        } else if (type == TINYGLTF_TYPE_VEC2) {
          return ElementType::BYTE2;
        } else if (type == TINYGLTF_TYPE_VEC3) {
                    return ElementType::BYTE3;
        } else if (type == TINYGLTF_TYPE_VEC4) {
                   return ElementType::BYTE4;
        }
      } else if (componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
        if (type == TINYGLTF_TYPE_SCALAR) {
            return ElementType::UBYTE;
        } else if (type == TINYGLTF_TYPE_VEC2) {
          return ElementType::UBYTE2;
        } else if (type == TINYGLTF_TYPE_VEC3) {
                    return ElementType::UBYTE3;
        } else if (type == TINYGLTF_TYPE_VEC4) {
                   return ElementType::UBYTE4;
        }
      } else if (componentType == TINYGLTF_COMPONENT_TYPE_SHORT) {
                if (type == TINYGLTF_TYPE_SCALAR) {
            return ElementType::SHORT;
        } else if (type == TINYGLTF_TYPE_VEC2) {
          return ElementType::SHORT2;
        } else if (type == TINYGLTF_TYPE_VEC3) {
                    return ElementType::SHORT3;
        } else if (type == TINYGLTF_TYPE_VEC4) {
                   return ElementType::SHORT4;
        }
      } else if (componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                       if (type == TINYGLTF_TYPE_SCALAR) {
            return ElementType::USHORT;
        } else if (type == TINYGLTF_TYPE_VEC2) {
          return ElementType::USHORT2;
        } else if (type == TINYGLTF_TYPE_VEC3) {
                    return ElementType::USHORT3;
        } else if (type == TINYGLTF_TYPE_VEC4) {
                   return ElementType::USHORT4;
        }
      } else if (componentType == TINYGLTF_COMPONENT_TYPE_INT) {
                              if (type == TINYGLTF_TYPE_SCALAR) {
            return ElementType::INT;
        }
      } else if (componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                                      if (type == TINYGLTF_TYPE_SCALAR) {
            return ElementType::UINT;
        }
      } else if (componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
                                            if (type == TINYGLTF_TYPE_SCALAR) {
            return ElementType::FLOAT;
        } else if (type == TINYGLTF_TYPE_VEC2) {
          return ElementType::FLOAT2;
        } else if (type == TINYGLTF_TYPE_VEC3) {
                    return ElementType::FLOAT3;
        } else if (type == TINYGLTF_TYPE_VEC4) {
                   return ElementType::FLOAT4;
        }
      } else {
          throw std::runtime_error("unsupported type");
      }
}

PrimitiveMode translatePrimitiveMode(int mode) {
    if (mode == TINYGLTF_MODE_TRIANGLES) {
        return PrimitiveMode::TRIANGLES;
    } else if (mode == TINYGLTF_MODE_TRIANGLE_FAN) {
        return PrimitiveMode::TRIANGLE_FANS;
    } else if (mode == TINYGLTF_MODE_TRIANGLE_STRIP) {
          return PrimitiveMode::TRIANGLE_STRIPS;
    } else {
        throw std::runtime_error("unsupported mode");
    }
}

void Loader::loadGLTFMesh(Entity* entity, tinygltf::Model &model, tinygltf::Mesh &mesh) {
        for (size_t i = 0; i < mesh.primitives.size(); ++i) {
            Primitive out;
            std::map<int, size_t> offsetMap;
        tinygltf::Primitive primitive = mesh.primitives[i];

        tinygltf::Accessor indexAccessor = model.accessors[primitive.indices];
            
            int j = 0;
        for (auto &attrib : primitive.attributes) {
            tinygltf::Accessor accessor = model.accessors[attrib.second];
            if (offsetMap.find(accessor.bufferView) == offsetMap.end()) {
                auto& bufferView = model.bufferViews[accessor.bufferView];
                auto& data =  model.buffers[bufferView.buffer].data;
                offsetMap.emplace(accessor.bufferView, out.vertexBuffers.size());
                out.vertexBuffers.push_back(std::vector<char>(data.begin() + bufferView.byteOffset, data.begin() + bufferView.byteOffset + bufferView.byteLength));
            }
            size_t index = offsetMap.at(accessor.bufferView);
            int byteStride = accessor.ByteStride(model.bufferViews[accessor.bufferView]);
            int numComps = tinygltf::GetNumComponentsInType(accessor.type);
            int size = tinygltf::GetComponentSizeInBytes(accessor.componentType);
            size_t offset_ = accessor.byteOffset;
            out.attibutes[j] = {
                .offset = (uint32_t)offset_,
                .stride = (uint8_t)byteStride,
                .name = attrib.first,
                .type = translateType(accessor.type, accessor.componentType),
                .index = (uint32_t)index
            };
            if (accessor.normalized) {
                out.attibutes[j].flags |= Attribute::FLAG_NORMALIZED;
            }
            ++j;
        }
            
            out.attributeCount = primitive.attributes.size();
            out.elementCount = indexAccessor.count;
            out.mode = translatePrimitiveMode(primitive.mode);
            auto& bufferView = model.bufferViews[indexAccessor.bufferView];
            auto data = model.buffers[bufferView.buffer].data.data() + bufferView.byteOffset + indexAccessor.byteOffset;
            for (int j = 0; j < indexAccessor.count; ++j) {
                if (tinygltf::GetComponentSizeInBytes(indexAccessor.componentType) == 2) {
                    out.indexBuffer.push_back(*(uint16_t*)(data + j*2));
                } else {
                    out.indexBuffer.push_back(*(uint32_t*)(data + j*4));
                }
            }
            entity->addPrimitive(std::move(out));
    }
}
