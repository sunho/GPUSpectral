#include "Loader.h"
#include <assert.h>
#include <stb_image.h>
#include <tiny_obj_loader.h>
#include <tinyparser-mitsuba.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/matrix.hpp>
#include <iostream>
#include "../renderer/Renderer.h"
#include "../renderer/Scene.h"
#include "../utils/Util.h"
#include "Engine.h"

#include <filesystem>
#include <fstream>
using namespace GPUSpectral;

static MeshPtr loadMesh(Renderer& renderer, Engine& engine, const std::string& path) {
    std::string warn;
    std::string err;
    std::vector<tinyobj::shape_t> shapes;
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::material_t> materials;
    std::map<int, Material*> generatedMaterials;

    auto p = std::filesystem::path(path);

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.c_str(), p.parent_path().string().c_str());
    if (!warn.empty()) {
        std::cout << "WARN: " << warn << std::endl;
    }
    if (!err.empty()) {
        std::cerr << err << std::endl;
    }

    for (size_t m = 0; m < materials.size(); m++) {
        tinyobj::material_t* mp = &materials[m];
    }
    std::vector<uint32_t> indices;
    std::vector<Mesh::Vertex> vertices;
    for (size_t s = 0; s < shapes.size(); s++) {
        auto& driver = renderer.getDriver();
        Material* material = nullptr;
        const int matId = shapes[s].mesh.material_ids[0];
        for (size_t f = 0; f < shapes[s].mesh.indices.size(); ++f) {
            indices.push_back(indices.size());
            auto i0 = shapes[s].mesh.indices[f];
            glm::vec3 pos;
            pos.x = attrib.vertices[3 * i0.vertex_index];
            pos.y = attrib.vertices[3 * i0.vertex_index + 1];
            pos.z = attrib.vertices[3 * i0.vertex_index + 2];
            glm::vec2 uv;
            uv.x = attrib.texcoords[2 * i0.texcoord_index];
            uv.y = attrib.texcoords[2 * i0.texcoord_index + 1];
            glm::vec3 normal;
            normal.x = attrib.normals[3 * i0.normal_index];
            normal.y = attrib.normals[3 * i0.normal_index + 1];
            normal.z = attrib.normals[3 * i0.normal_index + 2];
            vertices.push_back({ .pos = pos, .normal = normal, .uv = uv });
        }
    }
    return renderer.createMesh({ vertices.data(), vertices.size() }, { indices.data(), indices.size() });
}

static Handle<HwTexture> loadTexture(Renderer& renderer, const std::string& path) {
    int width, height, comp;
    unsigned char* data =
        stbi_load(path.c_str(), &width, &height, &comp, 0);
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
    auto tex = renderer.getDriver().createTexture(SamplerType::SAMPLER2D, TextureUsage::UPLOADABLE | TextureUsage::SAMPLEABLE, TextureFormat::RGBA8, 1, width, height, 1);
    renderer.getDriver().copyTextureInitialData(tex, { .data = (uint32_t*)textureData.data() });
    stbi_image_free(data);
    return tex;
}

/*TextureId loadHdrTexture(Renderer& renderer, const std::string& path) {
    int width, height, comp;
    std::vector<float> textureData;
    if (path.find(".pfm") != std::string::npos) {
        loadPfm(path, textureData, width, height);
    }
    else {
        stbi_hdr_to_ldr_gamma(1.0f);
        stbi_hdr_to_ldr_scale(1.0f);
        stbi_ldr_to_hdr_gamma(1.0f);
        stbi_ldr_to_hdr_scale(1.0f);
        float* data = stbi_loadf(path.c_str(), &width, &height, &comp, 0);
        if (!data) {
            throw std::runtime_error("FUCK");
        }
        for (int j = height - 1; j >= 0; --j) {
            for (int i = 0; i < width; ++i) {
                textureData.push_back(data[(j * width + i) * comp]);
                textureData.push_back(data[(j * width + i) * comp + 1]);
                textureData.push_back(data[(j * width + i) * comp + 2]);
                textureData.push_back(1.0f);
            }
        }
        stbi_image_free(data);
    }
    auto texId = renderer.createTexture(TextureFormat::RGBA32F, width, height);
    renderer.getTexture(texId)->upload(textureData.data());
    return texId;
}*/

static glm::vec3 convertRgb(tinyparser_mitsuba::Color color) {
    return glm::vec3(color.r, color.g, color.b);
}

static Handle<HwTexture> loadTexture(Renderer& renderer, Scene& scene, tinyparser_mitsuba::Object& obj, const std::filesystem::path& basepath) {
    std::string type = obj.pluginType();
    if (type == "bitmap") {
        /*auto filename = obj.property("filename").getString();
        auto path = (basepath / filename).string();
        return loadTexture(renderer, path);*/
    } else if (type == "checkerboard") {
        /*const uint32_t uSize = obj.property("uscale").getNumber(1);
        const uint32_t vSize = obj.property("vscale").getNumber(1);
        const glm::vec3 colorOn = convertRgb(obj.property("color0").getColor());
        const glm::vec3 colorOff = convertRgb(obj.property("color1").getColor());
        const uint32_t texWidth = uSize * 100 * 2;
        const uint32_t texHeight = vSize * 100 * 2;
        Image image = Procedural::createCheckerborad(2 * uSize, 2 * vSize, texWidth, texHeight, colorOn, colorOff);
        auto buf = image.pack();
        auto texId = renderer.createTexture(TextureFormat::RGBA8, texWidth, texHeight);
        renderer.getTexture(texId)->upload(buf.data());
        return texId;*/
    }
    assert(false);
    return Handle<HwTexture>();
}

static void loadMaterial(Renderer& renderer, Scene& scene, Material* material, tinyparser_mitsuba::Object& obj, const std::filesystem::path& basepath) {
    std::string type = obj.pluginType();
    if (type == "twosided") {
        material->twofaced = true;
    }
    if (type == "diffuse") {
        auto rgb = obj.property("reflectance").getColor();
        glm::vec3 reflectance = glm::vec3(rgb.r, rgb.g, rgb.b);
        auto bsdf = DiffuseBSDF{ reflectance };
        for (auto [name, child] : obj.namedChildren()) {
            if (name == "reflectance") {
                auto tex = loadTexture(renderer, scene, *child, basepath);
                //bsdf.reflectanceTex = renderer.getTexture(tex)->getTextureObject();
                break;
            }
        }
        material->bsdf = scene.addDiffuseBSDF(bsdf);
    } else if (type == "roughplastic") {
        auto rgb = obj.property("diffuse_reflectance").getColor();
        float alpha = obj.property("alpha").getNumber();
        glm::vec3 diffuse = glm::vec3(rgb.r, rgb.g, rgb.b);
        if (obj.property("ext_ior").isValid()) {
            if (abs(obj.property("ext_ior").getNumber() - 1.0f) > 0.001f) {
                std::cout << "unsupported ext ior of plastic" << std::endl;
            }
        }
        float ior = obj.property("int_ior").isValid() ? obj.property("int_ior").getNumber() : 1.3f;
        float R0 = (ior - 1.0f) / (ior + 1.0f);
        R0 *= R0;
        auto bsdf = RoughPlasticBSDF{
            .diffuse = diffuse,
            .iorIn = ior,
            .iorOut = 1.0f,
            .R0 = R0,
            .alpha = (float)sqrt(2.0f) * alpha,
        };
        for (auto [name, child] : obj.namedChildren()) {
            if (name == "diffuse_reflectance") {
                auto tex = loadTexture(renderer, scene, *child, basepath);
                //bsdf.diffuseTex = renderer.getTexture(tex)->getTextureObject();
                break;
            }
        }
        material->bsdf = scene.addRoughPlasticBSDF(bsdf);
    } else if (type == "dielectric") {
        float intIOR = obj.property("int_ior").getNumber();
        float extIOR = obj.property("ext_ior").getNumber();
        material->bsdf = scene.addSmoothDielectricBSDF(SmoothDielectricBSDF{ intIOR, extIOR });
    } else if (type == "conductor") {
        float ior = obj.property("eta").isValid() ? obj.property("eta").getNumber() : 0.0f;
        material->bsdf = scene.addSmoothConductorBSDF(SmoothConductorBSDF{ ior, 1.0f });
    } else if (type == "plastic") {
        auto rgb = obj.property("diffuse_reflectance").getColor();
        glm::vec3 diffuse = glm::vec3(rgb.r, rgb.g, rgb.b);
        if (obj.property("ext_ior").isValid()) {
            if (abs(obj.property("ext_ior").getNumber() - 1.0f) > 0.001f) {
                std::cout << "unsupported ext ior of plastic" << std::endl;
            }
        }
        float ior = obj.property("int_ior").isValid() ? obj.property("int_ior").getNumber() : 1.3f;
        float R0 = (ior - 1.0f) / (ior + 1.0f);
        R0 *= R0;
        material->bsdf = scene.addSmoothPlasticBSDF(SmoothPlasticBSDF{
            .diffuse = diffuse,
            .iorIn = ior,
            .iorOut = 1.0f,
            .R0 = R0,
        });
    } else if (type == "roughconductor") {
        auto eta_ = obj.property("eta").getColor();
        auto k_ = obj.property("k").getColor();
        auto reflectance_ = obj.property("specular_reflectance").getColor();
        glm::vec3 eta = glm::vec3(eta_.r, eta_.g, eta_.b);
        glm::vec3 k = glm::vec3(k_.r, k_.g, k_.b);
        glm::vec3 reflectance = glm::vec3(reflectance_.r, reflectance_.g, reflectance_.b);
        float alpha = obj.property("alpha").getNumber();
        material->bsdf = scene.addRoughConductorBSDF(RoughConductorBSDF{
            .eta = eta,
            .k = k,
            .reflectance = reflectance,
            .alpha = (float)sqrt(2) * alpha,
        });
    }

    for (auto child : obj.anonymousChildren()) {
        if (child->type() == tinyparser_mitsuba::OT_BSDF) {
            loadMaterial(renderer, scene, material, *child, basepath);
        }
    }
}

void loadPfm(const std::string& path, std::vector<float>& data, int& width, int& height) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file");
    }
    std::string header;
    file >> header;
    if (header != "PF") {
        throw std::runtime_error("invalid pfm file");
    }
    file >> width >> height;
    data.resize(width * height * 4);
    for (uint32_t i = 0; i < width * height; ++i) {
        file >> data[i * 4 + 0] >> data[i * 4 + 1] >> data[i * 4 + 2];
    }
}

Scene GPUSpectral::loadScene(Engine& engine, Renderer& renderer, const std::string& path) {
    std::unordered_map<std::string, MeshPtr> meshCache;
    auto loadOrGetMesh = [&](const std::string& objPath) {
        if (meshCache.find(objPath) != meshCache.end()) {
            return meshCache.at(objPath);
        }
        auto mesh = loadMesh(renderer, engine, objPath);
        meshCache.emplace(objPath, mesh);
        return mesh;
    };

    Scene outScene = {};

    auto parentPath = std::filesystem::path(path).parent_path();
    tinyparser_mitsuba::SceneLoader loader;
    auto scene = loader.loadFromFile(path.c_str());
    for (auto obj : scene.anonymousChildren()) {
        if (obj->type() == tinyparser_mitsuba::OT_SHAPE) {
            std::string filename;
            if (obj->pluginType() == "obj") {
                filename = (parentPath / obj->property("filename").getString()).string();
            } else if (obj->pluginType() == "rectangle") {
                filename = engine.assetPath("rect.obj");
            } else if (obj->pluginType() == "cube") {
                filename = engine.assetPath("box.obj");
            } else if (obj->pluginType() == "disk") {
                filename = engine.assetPath("disk.obj");
            } else {
                filename = (parentPath / filename).string();
            }
            auto mesh = loadOrGetMesh(filename);
            auto transform = obj->property("to_world").getTransform();
            bool faceNormals = obj->property("face_normals").getBool(false);
            auto matrix = glm::transpose(glm::make_mat4(transform.matrix.data()));
            if (obj->property("center").isValid()) {
                auto point = obj->property("center").getVector();
                matrix[3][0] = point.x;
                matrix[3][1] = point.y;
                matrix[3][2] = point.z;
                matrix[3][3] = 1.0f;
            }

            bool emitting = false;
            RenderObject renderObject = {};
            Material material = {};
            for (auto child : obj->anonymousChildren()) {
                if (child->type() == tinyparser_mitsuba::OT_BSDF) {
                    loadMaterial(renderer, outScene, &material, *child, parentPath);
                } else if (child->type() == tinyparser_mitsuba::OT_EMITTER) {
                    if (child->pluginType() == "area") {
                        auto col = child->property("radiance").getColor();
                        material.emission = glm::vec3(col.r, col.g, col.b);
                        emitting = true;
                    }
                }
            }

            renderObject.mesh = mesh;
            renderObject.transform = matrix;
            renderObject.material = outScene.addMaterial(material);
            outScene.getMaterial(renderObject.material).facenormals = faceNormals;
            outScene.addRenderObject(renderObject);

            if (emitting) {
                auto m = renderObject.mesh;
                auto vertices = m->getVertices();
                for (size_t i = 0; i < vertices.size(); i += 3) {
                    TriangleLight light = {};
                    glm::vec3 pos0 = vertices[i].pos;
                    glm::vec3 pos1 = vertices[i + 1].pos;
                    glm::vec3 pos2 = vertices[i + 2].pos;
                    light.positions[0] = (renderObject.transform * glm::vec4(pos0.x, pos0.y, pos0.z, 1.0f));
                    light.positions[1] = (renderObject.transform * glm::vec4(pos1.x, pos1.y, pos1.z, 1.0f));
                    light.positions[2] = (renderObject.transform * glm::vec4(pos2.x, pos2.y, pos2.z, 1.0f));
                    light.radiance = glm::vec4(material.emission, 1.0);
                    outScene.addTriangleLight(light);
                }
            }
        } else if (obj->type() == tinyparser_mitsuba::OT_SENSOR) {
            auto transform = obj->property("to_world").getTransform();
            float fov = obj->property("fov").getNumber();
            outScene.camera.setFov(fov * M_PI / 180.f, 1.0f, 0.001f, 1000.0f);
            auto matrix = glm::make_mat4(transform.matrix.data());
            matrix = glm::transpose(matrix);
            outScene.camera.setToWorld(matrix);
        } else if (obj->type() == tinyparser_mitsuba::OT_EMITTER) {
            /*auto filename = obj->property("filename").getString();
            auto path = (parentPath / filename).string();
            auto tex = loadHdrTexture(renderer, path);
            auto transform = obj->property("to_world").getTransform();
            auto matrix = glm::make_mat4(transform.matrix.data());
            outScene.envMap = tex;
            outScene.envMapTransform = matrix;*/
        }
    }
    return outScene;
}
