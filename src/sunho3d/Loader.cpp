#include "Loader.h"

#include <iostream>

#include "Engine.h"
#include "Renderer.h"
#include "Scene.h"
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

#include <stdexcept>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <filesystem>

using namespace sunho3d;

Loader::Loader(Engine &engine, Renderer &renderer)
    : engine(engine), renderer(renderer) {
}

Scene *Loader::loadGLTF(const std::string &path) {
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

    Scene *scene = engine.createScene(&renderer);
    tinygltf::Scene &s = model.scenes[model.defaultScene];
    for (auto node : s.nodes) {
        loadGLTFNode(scene, model, model.nodes[node]);
    }
    return scene;
}

Entity *Loader::loadObj(const std::string &path) {
    Entity *out = engine.createEntity();
    std::string warn;
    std::string err;
    std::vector<tinyobj::shape_t> shapes;
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::material_t> materials;
    std::map<int, Material *> generatedMaterials;

    auto p = std::filesystem::path(path);

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.c_str(), p.parent_path().string().c_str());
    if (!warn.empty()) {
        std::cout << "WARN: " << warn << std::endl;
    }
    if (!err.empty()) {
        std::cerr << err << std::endl;
    }

    for (size_t m = 0; m < materials.size(); m++) {
        tinyobj::material_t *mp = &materials[m];
    }
    for (size_t s = 0; s < shapes.size(); s++) {
        auto &driver = renderer.getDriver();
        Material *material = nullptr;
        const int matId = shapes[s].mesh.material_ids[0];
        auto it = generatedMaterials.find(matId);
        if (it == generatedMaterials.end()) {
            int width, height, comp;
            unsigned char *data =
                stbi_load((p.parent_path() / materials[matId].diffuse_texname).string().c_str(), &width, &height, &comp, 0);
            if (!data) {
                throw std::runtime_error("FUCK");
            }
            std::vector<char> textureData;
            for (int j = height - 1; j >= 0; --j) {
                for (int i = 0; i < width; ++i) {
                    textureData.push_back(data[(j * width + i) * comp]);
                    textureData.push_back(data[(j * width + i) * comp + 1]);
                    textureData.push_back(data[(j * width + i) * comp + 2]);
                    textureData.push_back(0xFF);
                }
            }
            auto tex = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::UPLOADABLE | TextureUsage::SAMPLEABLE, TextureFormat::RGBA8, width, height);
            driver.updateTexture(tex, { .data = (uint32_t *)textureData.data() });
            material = engine.createMaterial();
            material->diffuseMap = tex;
            stbi_image_free(data);
            generatedMaterials.emplace(matId, material);
        } else {
            material = it->second;
        }

        Primitive primitive;
        primitive.material = material;
        std::vector<float> v;
        std::vector<float> vt;
        std::vector<float> vn;
        for (size_t f = 0; f < shapes[s].mesh.indices.size(); ++f) {
            primitive.indices.push_back(f);
            auto i0 = shapes[s].mesh.indices[f];
            glm::vec3 pos;
            pos.x = attrib.vertices[3 * i0.vertex_index];
            pos.y = attrib.vertices[3 * i0.vertex_index + 1];
            pos.z = attrib.vertices[3 * i0.vertex_index + 2];
            v.push_back(pos.x);
            v.push_back(pos.y);
            v.push_back(pos.z);
            glm::vec2 uv;
            uv.x = attrib.texcoords[2 * i0.texcoord_index];
            uv.y = attrib.texcoords[2 * i0.texcoord_index + 1];
            vt.push_back(uv.x);
            vt.push_back(uv.y);
            glm::vec3 normal;
            normal.x = attrib.normals[3 * i0.normal_index];
            normal.y = attrib.normals[3 * i0.normal_index + 1];
            normal.z = attrib.normals[3 * i0.normal_index + 2];
            vn.push_back(normal.x);
            vn.push_back(normal.y);
            vn.push_back(normal.z);
            primitive.vertices.push_back({.pos = pos, .normal = normal, .uv = uv});
        }
        primitive.attibutes[0] = {
            .name = "position",
            .index = 0,
            .offset = 0,
            .stride = 12,
            .type = ElementType::FLOAT3
        };
        primitive.attibutes[1] = {
            .name = "normal", .index = 1, .offset = 0, .stride = 12, .type = ElementType::FLOAT3
        };
        primitive.attibutes[2] = {
            .name = "texcoord",
            .index = 2,
            .offset = 0,
            .stride = 8,
            .type = ElementType::FLOAT2,
        };
        auto buffer0 = driver.createBufferObject(4 * v.size(), BufferUsage::VERTEX | BufferUsage::STORAGE);
        driver.updateBufferObject(buffer0, { .data = (uint32_t *)v.data() }, 0);
        auto buffer1 = driver.createBufferObject(4 * v.size(), BufferUsage::VERTEX | BufferUsage::STORAGE);
        driver.updateBufferObject(buffer1, { .data = (uint32_t *)vn.data() }, 0);
        auto buffer2 = driver.createBufferObject(4 * vt.size(), BufferUsage::VERTEX | BufferUsage::STORAGE);
        driver.updateBufferObject(buffer2, { .data = (uint32_t *)vt.data() }, 0);

        auto vbo = driver.createVertexBuffer(3, v.size() / 3, 3, primitive.attibutes);
        driver.setVertexBuffer(vbo, 0, buffer0);
        driver.setVertexBuffer(vbo, 1, buffer1);
        driver.setVertexBuffer(vbo, 2, buffer2);

        auto ibo = driver.createIndexBuffer(primitive.indices.size());
        driver.updateIndexBuffer(ibo, { .data = (uint32_t *)primitive.indices.data() }, 0);
        primitive.indexBuffer = ibo;
        primitive.vertexBuffer = vbo;
        primitive.hwInstance = driver.createPrimitive(PrimitiveMode::TRIANGLES);
        driver.setPrimitiveBuffer(primitive.hwInstance, vbo, ibo);
        out->addPrimitive(primitive);
    }
    return out;
}

void Loader::loadGLTFNode(Scene *scene, tinygltf::Model &model, tinygltf::Node &node,
                          Entity *parent) {
    if ((node.mesh >= 0) && (node.mesh < model.meshes.size())) {
        Entity *entity = engine.createEntity();
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

void Loader::loadGLTFMesh(Entity *entity, tinygltf::Model &model, tinygltf::Mesh &mesh) {
    /*for (size_t i = 0; i < mesh.primitives.size(); ++i) {
        Primitive out;
        std::map<int, size_t> offsetMap;
        std::vector<Handle<HwBufferObject>> vertexBuffers;
        
        tinygltf::Primitive primitive = mesh.primitives[i];
        tinygltf::Accessor indexAccessor = model.accessors[primitive.indices];

        int j = 0;
        for (auto &attrib : primitive.attributes) {
            tinygltf::Accessor accessor = model.accessors[attrib.second];
            if (offsetMap.find(accessor.bufferView) == offsetMap.end()) {
                auto &bufferView = model.bufferViews[accessor.bufferView];
                auto &data = model.buffers[bufferView.buffer].data;
                offsetMap.emplace(accessor.bufferView, vertexBuffers.size());
                renderer.getDriver().createBufferObject(bufferView.byteLength);
                
                out.vertexBuffers.push_back(std::vector<char>(data.begin() + bufferView.byteOffset,
                                                              data.begin() + bufferView.byteOffset +
                                                                 ));
            }
            size_t index = offsetMap.at(accessor.bufferView);
            int byteStride = accessor.ByteStride(model.bufferViews[accessor.bufferView]);
            int numComps = tinygltf::GetNumComponentsInType(accessor.type);
            int size = tinygltf::GetComponentSizeInBytes(accessor.componentType);
            size_t offset_ = accessor.byteOffset;
            out.attibutes[j] = { .offset = (uint32_t)offset_,
                                 .stride = (uint8_t)byteStride,
                                 .name = attrib.first,
                                 .type = translateType(accessor.type, accessor.componentType),
                                 .index = (uint32_t)index };
            if (accessor.normalized) {
                out.attibutes[j].flags |= Attribute::FLAG_NORMALIZED;
            }
            ++j;
        }

        out.attributeCount = primitive.attributes.size();
        out.elementCount = indexAccessor.count;
        out.mode = translatePrimitiveMode(primitive.mode);
        auto &bufferView = model.bufferViews[indexAccessor.bufferView];
        auto data = model.buffers[bufferView.buffer].data.data() + bufferView.byteOffset +
                    indexAccessor.byteOffset;
        for (int j = 0; j < indexAccessor.count; ++j) {
            if (tinygltf::GetComponentSizeInBytes(indexAccessor.componentType) == 2) {
                out.indexBuffer.push_back(*(uint16_t *)(data + j * 2));
            } else {
                out.indexBuffer.push_back(*(uint32_t *)(data + j * 4));
            }
        }
        entity->addPrimitive(std::move(out));
    }*/
}
