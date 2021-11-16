#include "Scene.h"
#include "backend/vulkan/VulkanDriver.h"
#include "renderer/Renderer.h"
#include <Tracy.hpp>
using namespace sunho3d;


GVetexBufferContainer::GVetexBufferContainer(VulkanDriver& driver) : driver(driver) {
    gpuBuffer = driver.createBufferObject(maxSize, BufferUsage::STORAGE);
}

GVetexBufferContainer::~GVetexBufferContainer() {
    if (gpuBuffer) {
        driver.destroyBufferObject(gpuBuffer);
    }
}

uint32_t GVetexBufferContainer::getVertexStart(const Primitive& primitive) {
    const uint32_t handle = primitive.hwInstance.getId();
    return vertexStarts.at(handle);
}

void GVetexBufferContainer::registerPrimitiveIfNeccesary(const Primitive& primitive) {
    const uint32_t handle = primitive.hwInstance.getId();
    const size_t psize = primitive.vertices.size() * sizeof(Vertex);;
    auto it = vertexStarts.find(handle);
    if (it == vertexStarts.end()) {
        buffer.insert(buffer.end(), primitive.vertices.begin(), primitive.vertices.end());
        vertexStarts.emplace(handle, currentSize / sizeof(Vertex));
        currentSize += psize;
        if (maxSize < currentSize) {
            growBuffer();
        }
        uploadBuffer();
    }
}

Handle<HwBufferObject> GVetexBufferContainer::getGPUBuffer() const {
    return gpuBuffer;
}

void GVetexBufferContainer::growBuffer() {
    maxSize = currentSize * 2;
    driver.destroyBufferObject(gpuBuffer);
    gpuBuffer = driver.createBufferObject(maxSize, BufferUsage::STORAGE);
}

void GVetexBufferContainer::uploadBuffer() {
    BufferDescriptor desc = {};
    desc.data = (uint32_t*)buffer.data();
    desc.size = currentSize;
    driver.updateBufferObjectSync(gpuBuffer, desc, 0);
}

Scene::Scene(Renderer* renderer)
    : renderer(renderer), globalVertexBufferContainer(renderer->getDriver()) {
}

void Scene::addEntity(Entity* entity) {
    for (auto& p : entity->getMesh()->getPrimitives()) {
        globalVertexBufferContainer.registerPrimitiveIfNeccesary(p);
    }
    entities.push_back(entity);
}

void Scene::prepare() {
    ZoneScopedN("Scene perpare")
    sceneData.geometries.clear();
    sceneData.worldTransforms.clear();
    sceneData.lightBuffer.lightNum = 0;
    sceneData.globalVertexBuffer = globalVertexBufferContainer.getGPUBuffer();

    for (auto& entity : entities) {
        visitEntity(entity, glm::identity<glm::mat4>(), glm::identity<glm::mat4>());
    }

    for (auto& light : lights) {
        auto& transform = light->getTransform();
        glm::vec4 pos = glm::vec4(transform.x, transform.y, transform.z, 1.0f);
        glm::vec4 dir = glm::vec4(light->getDirection(), 0.0f);
        glm::vec4 RI = glm::vec4(light->getRadius(), light->getIntensity(), 0.0f, 0.0f);
        sceneData.lightBuffer.lights[sceneData.lightBuffer.lightNum++] = {
            .pos = pos,
            .dir = dir,
            .RI = RI
        };
    }
}

void Scene::visitEntity(Entity* entity, const glm::mat4& currentTransform, const glm::mat4& currentTransformInvT) {
    glm::mat4 nextTransform = entity->getTransform() * currentTransform;
    glm::mat4 nextTransformInvT = entity->getTransformInvT() * currentTransformInvT;
    for (auto& prim : entity->getMesh()->getPrimitives()) {
        sceneData.geometries.push_back({
            .material = entity->getMaterial(), 
            .primitive = prim.hwInstance, 
            .vertexStart = globalVertexBufferContainer.getVertexStart(prim)
                                         });
        sceneData.worldTransforms.push_back(nextTransform);
        sceneData.worldTransformsInvT.push_back(nextTransformInvT);
    }
    for (auto& entity : entity->getNodes()) {
        visitEntity(entity, nextTransform, nextTransformInvT);
    }
}

void Scene::addLight(sunho3d::Light* light) {
    lights.push_back(light);
}
