#include <GPUSpectral/Engine.h>
#include <GPUSpectral/Loader.h>
#include <GPUSpectral/renderer/PathTracer.h>   

#include <Windows.h>
#include <filesystem>

std::filesystem::path basePath() {
    char path[1024];
    GetModuleFileNameA(0, path, 1024);
    auto out = std::filesystem::path(path);
    return out.parent_path();
}


int main() {

    GPUSpectral::Engine engine(basePath(), basePath() / "assets");
    GPUSpectral::Window* window = engine.createWindow(600, 600);
    GPUSpectral::Renderer* renderer = engine.createRenderer(window);
    auto pathTracer = std::make_unique<GPUSpectral::PathTracer>(*renderer);
    renderer->setRendererImpl(std::move(pathTracer));

    auto scene = loadScene(engine, *renderer, (basePath() / "assets" / "scenes" / "test2" / "scene.xml").string());
    glm::vec3 cameraPos = glm::vec3(0.0, 1.0, 13.0f);
    glm::vec3 origin = glm::vec3(0.0, 1.0, 0.0);

    window->run([&]() {
        //t.x += 0.1;
        //neptune->setTransform(t);
        int a = glfwGetKey(window->window, GLFW_KEY_A);
        if (a == GLFW_PRESS)
        {
            cameraPos.z -= 0.05f;
        }
        int d = glfwGetKey(window->window, GLFW_KEY_D);
        if (d == GLFW_PRESS)
        {
            cameraPos.x += 0.05f;
        }
        int w = glfwGetKey(window->window, GLFW_KEY_W);
        if (w == GLFW_PRESS)
        {
            cameraPos.z += 0.05f;
        }
        int s = glfwGetKey(window->window, GLFW_KEY_S);
        if (s == GLFW_PRESS)
        {
            cameraPos.x -= 0.05f;
        }
        //scene.camera.lookAt(cameraPos, origin, glm::vec3(0.0, 1.0, 0.0));
        renderer->run(scene);
    });
    
    


    return 0;
}
