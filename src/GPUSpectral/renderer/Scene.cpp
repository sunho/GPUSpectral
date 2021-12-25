#include "Scene.h"
#include "Renderer.h"

void Scene::prepare(Renderer& renderer) {
    for (auto& obj : renderObjects) {
        auto mesh = renderer.getMesh(obj.meshId);
        for (auto& pos : mesh->positions) {
            sceneData.positions.push_back(obj.transform * float4(pos.x, pos.y, pos.z, 1.0f));
        }

        auto invT = obj.transform.transpose().inverse();
        for (auto& nor : mesh->normals) {
            sceneData.normals.push_back(invT * float4(nor.x, nor.y, nor.z, 0.0f));
        }
        
        for (auto& uv : mesh->uvs) {
            sceneData.uvs.push_back(make_float2(uv.x, uv.y));
        }

        for (size_t i = 0; i < mesh->positions.size() / 3; ++i) {
            sceneData.matIndices.push_back(obj.material);
        }
    }
}
