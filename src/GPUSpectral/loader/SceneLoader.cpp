#include "SceneLoader.h"
#include <tinyparser-mitsuba.h>

#include <filesystem>

static void loadMaterial(Material* material, tinyparser_mitsuba::Object& obj, const std::filesystem::path& basepath) {
    std::string type = obj.pluginType();
    /*if (type == "twosided") {
        material->twosided = true;
    }*/
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
            material->materialData = DiffuseColorMaterialData{ glm::vec3(rgb.r, rgb.g, rgb.b) };
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
            material->materialData = DiffuseColorMaterialData{ glm::vec3(rgb.r, rgb.g, rgb.b) };
        }
    }

    for (auto child : obj.anonymousChildren()) {
        if (child->type() == tinyparser_mitsuba::OT_BSDF) {
            loadMaterial(material, *child, basepath);
        }
    }
}

Scene loadScene(Renderer& renderer, const std::string& path) {
    std::unordered_map<std::string, int> meshCache;
    auto loadOrGetMesh = [&](const std::string& objPath) {
        if (meshCache.find(objPath) == meshCache.end()) {
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
    int i = 0;
    for (auto obj : scene.anonymousChildren()) {
        if (obj->type() == tinyparser_mitsuba::OT_SHAPE) {
            std::string filename;
            if (obj->pluginType() == "obj") {
                filename = obj->property("filename").getString();
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
            auto matrix = glm::make_mat4(transform.matrix.data());
            matrix = glm::transpose(matrix);
            if (obj->property("center").isValid())
            {
                auto point = obj->property("center").getVector();
                matrix[3] = glm::vec4(point.x, point.y, point.z, 1.0);
            }

            auto material = engine.createMaterial();
            entity->setMaterial(material);
            for (auto child : obj->anonymousChildren()) {
                if (child->type() == tinyparser_mitsuba::OT_BSDF) {
                    material->materialData = DiffuseColorMaterialData{ glm::vec3(1.0f) };
                    loadMaterial(material, *child, p);
                }
                else if (child->type() == tinyparser_mitsuba::OT_EMITTER) {
                    auto col = child->property("radiance").getColor();
                    glm::vec3 radiance = glm::vec3(col.r, col.g, col.b);
                    auto l = new sunho3d::Light(sunho3d::Light::Type::POINT);
                    auto pos = matrix[3];
                    l->setTransform({ .x = pos.x, .y = pos.y, .z = pos.z });
                    l->setRadiance(radiance);
                    outScene->addLight(l);
                    material->materialData = EmissionMaterialData{ radiance };
                }
            }


            entity->setTransformMatrix(matrix);
            entity->setMesh(mesh);

            outScene->addEntity(entity);
        } else if (obj->type() == tinyparser_mitsuba::OT_SENSOR) {
            auto transform = obj->property("to_world").getTransform();
            float fov = obj->property("fov").getNumber();
            auto matrix = glm::make_mat4(transform.matrix.data());
            matrix = glm::transpose(matrix);
            glm::vec4 affine = matrix[3];


            //matrix = glm::inverse(matrix); // TODO: need this?
            matrix[3] = -1.0f * affine;
            matrix[3][3] *= -1.0f;
            //matrix[2][2] *= -1.0f;
            //matrix[1][1] *= -1.0f;



            outScene->getCamera().view = matrix;
            outScene->getCamera().setProjectionFov(glm::radians(fov), 1.0, 0.01f, 25.0f);
            //outScene->getCamera().proj[2][2] *= -1.0f;
            //outScene->getCamera().proj[1][1] *= -1.0f;
        }
    }
}

Mesh loadMesh(const std::string& path) {
       
}

