#include "SceneLoader.h"
#include <tinyparser-mitsuba.h>
#include <tiny_obj_loader.h>
#include <iostream>

#include <filesystem>

static void loadMaterial(Scene& scene, Material* material, tinyparser_mitsuba::Object& obj, const std::filesystem::path& basepath) {
    std::string type = obj.pluginType();
    if (type == "twosided") {
        material->twofaced = true;
    }
    if (type == "diffuse") {
        bool found = false;
        /*for (auto [name, child] : obj.namedChildren()) {
            if (name == "reflectance") {
                found = true;
                auto filename = child->property("filename").getString();
                auto path = (basepath / filename).string();
                auto tex = loadOrGetTexture(path);
                material->materialData = DiffuseTextureMaterialData{ tex };
                break;
            }
        }*/
        if (!found) {
            auto rgb = obj.property("reflectance").getColor();
            float3 reflectance = make_float3(rgb.r, rgb.g, rgb.b);
            material->bsdf = scene.addDiffuseBSDF(DiffuseBSDF{ reflectance });
        }
    } else if (type == "roughplastic") {
        bool found = false;
        /*for (auto [name, child] : obj.namedChildren()) {
            if (name == "diffuse_reflectance") {
                found = true;
                auto filename = child->property("filename").getString();
                auto path = (basepath / filename).string();
                auto tex = loadOrGetTexture(path);
                material->materialData = DiffuseTextureMaterialData{ tex };
                break;
            }
        }*/
        if (!found) {
            auto rgb = obj.property("diffuse_reflectance").getColor();
            //material->color = make_float3(rgb.r, rgb.g, rgb.b);
        }
    }
    else if (type == "dielectric") {
        float intIOR = obj.property("int_ior").getNumber();
        float extIOR = obj.property("ext_ior").getNumber();
        material->bsdf = scene.addSmoothDielectricBSDF(SmoothDielectricBSDF{ intIOR, extIOR });
    }

    for (auto child : obj.anonymousChildren()) {
        if (child->type() == tinyparser_mitsuba::OT_BSDF) {
            loadMaterial(scene, material, *child, basepath);
        }
    }
}

Scene loadScene(Renderer& renderer, const std::string& path) {
    std::unordered_map<std::string, int> meshCache;
    auto loadOrGetMesh = [&](const std::string& objPath) {
        if (meshCache.find(objPath) != meshCache.end()) {
            return meshCache.at(objPath);
        }
        auto obj = loadMesh(objPath);
        int meshId = renderer.addMesh(obj);
        meshCache.emplace(objPath, meshId);
        return meshId;
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
            }
            else if (obj->pluginType() == "rectangle") {
                filename = renderer.assetPath("rect.obj");
            }
            else if (obj->pluginType() == "cube") {
                filename = renderer.assetPath("box.obj");
            }
            else {
                filename = (parentPath / filename).string();
            }
            auto mesh = loadOrGetMesh(filename);
            auto transform = obj->property("to_world").getTransform();
            auto matrix = mat4(transform.matrix.data());
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
                    loadMaterial(outScene, &material, *child, parentPath);
                } else if (child->type() == tinyparser_mitsuba::OT_EMITTER) {
                    auto col = child->property("radiance").getColor();
                    material.emission = make_float3(col.r, col.g, col.b);
                    emitting = true;
                }
            }

            renderObject.meshId = mesh;
            renderObject.transform = matrix;
            renderObject.material = outScene.addMaterial(material);
            outScene.addRenderObject(renderObject);

            if (emitting) {
                auto m = renderer.getMesh(mesh);
                for (size_t i = 0; i < m->positions.size(); i+=3) {
                    TriangleLight light = {};
                    float3 pos0 = m->positions[i];
                    float3 pos1 = m->positions[i+1];
                    float3 pos2 = m->positions[i+2];
                    light.positions[0] = make_float3(renderObject.transform * float4(pos0.x, pos0.y, pos0.z, 1.0f));
                    light.positions[1] = make_float3(renderObject.transform * float4(pos1.x, pos1.y, pos1.z, 1.0f));
                    light.positions[2] = make_float3(renderObject.transform * float4(pos2.x, pos2.y, pos2.z, 1.0f));
                    light.radinace = material.emission;
                    outScene.addTriangleLight(light);
                }
            }
        } else if (obj->type() == tinyparser_mitsuba::OT_SENSOR) {
            auto transform = obj->property("to_world").getTransform();
            float fov = obj->property("fov").getNumber();
            outScene.camera.fov = fov * M_PI / 180.f;
            auto matrix = mat4(transform.matrix.data());
            float3 eye = make_float3(matrix[3][0], matrix[3][1], matrix[3][2]);
            outScene.camera.eye = eye;
            outScene.camera.u = make_float3(matrix[0][0], matrix[0][1], matrix[0][2]);
            outScene.camera.v = make_float3(matrix[1][0], matrix[1][1], matrix[1][2]);
            outScene.camera.w = make_float3(matrix[2][0], matrix[2][1], matrix[2][2]);
        }
    }
    return outScene;
}

Mesh loadMesh(const std::string& path) {
    Mesh outMesh = {};
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
    for (size_t s = 0; s < shapes.size(); s++) {
        Material* material = nullptr;
        const int matId = shapes[s].mesh.material_ids[0];
        for (size_t f = 0; f < shapes[s].mesh.indices.size(); ++f) {
            auto i0 = shapes[s].mesh.indices[f];
            float3 pos;
            pos.x = attrib.vertices[3 * i0.vertex_index];
            pos.y = attrib.vertices[3 * i0.vertex_index + 1];
            pos.z = attrib.vertices[3 * i0.vertex_index + 2];
            outMesh.positions.push_back(pos);
            float2 uv;
            uv.x = attrib.texcoords[2 * i0.texcoord_index];
            uv.y = attrib.texcoords[2 * i0.texcoord_index + 1];
            outMesh.uvs.push_back(uv);
            float3 normal;
            normal.x = attrib.normals[3 * i0.normal_index];
            normal.y = attrib.normals[3 * i0.normal_index + 1];
            normal.z = attrib.normals[3 * i0.normal_index + 2];
            outMesh.normals.push_back(normal);
        }
    }
    return outMesh;
}

