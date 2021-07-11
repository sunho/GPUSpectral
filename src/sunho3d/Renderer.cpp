#include "Renderer.h"


#include <sunho3d/shaders/triangle_vert.h>
#include <sunho3d/shaders/triangle_frag.h>

#include <glm/glm.hpp>

using namespace sunho3d;

Renderer::Renderer(Window* window) : window(window), driver(window) {
}

Renderer::~Renderer() {
}

void Renderer::run() {
    window->run([&](){
        auto primitive = driver.createPrimitive();
        AttributeArray attrs = { Attribute{.offset=0, .stride=2} };
        auto vertexBuffer = driver.createVertexBuffer(3, 1, attrs);
        auto buffer = driver.createBufferObject(3*2*4);
        auto indexBufer = driver.createIndexBuffer(3);
        BufferDescriptor descriptor;
        uint16_t* index = new uint16_t[3];
        index[0] = 0;
         index[1] = 1;
         index[2] = 2;
        descriptor.data = (uint32_t*)index;
        
        driver.updateIndexBuffer(indexBufer, descriptor, 0);
        glm::vec2* verts = new glm::vec2[3];
        verts[0] = glm::vec2(0.0, -0.5);
        verts[1] = glm::vec2(0.5, 0.5);
        verts[2] = glm::vec2(-0.5, 0.5);
        descriptor.data = (uint32_t*) verts;
        driver.updateBufferObject(buffer, descriptor, 0);
        driver.setVertexBuffer(vertexBuffer, buffer);
        driver.setPrimitiveBuffer(primitive, vertexBuffer, indexBufer);
        auto renderTarget = driver.createDefaultRenderTarget();
        RenderPassParams params;
        Program prog;
        prog.codes[0] = std::vector<char>(triangle_vert, triangle_vert+triangle_vert_len);
        prog.codes[1] = std::vector<char>(triangle_frag, triangle_frag+triangle_frag_len);
        auto vv = driver.createProgram(prog);
        driver.beginRenderPass(renderTarget, params);
        PipelineState pipe;
        pipe.program = vv;
        driver.draw(pipe, primitive);
    });
}

