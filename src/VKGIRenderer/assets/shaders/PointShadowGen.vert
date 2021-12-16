#version 450
#pragma shader_stage(vertex)

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTex;

layout(location = 0) out vec3 pos;

layout(binding = 0) uniform TransformUniformBuffer {
    mat4 MVP;
    mat4 model;
    mat4 invModelT;
    vec3 cameraPos;
} transform;

layout(binding = 1) uniform ShadowGenUniformBuffer {
    mat4 lightVP;
    vec3 lightPos;
    float farPlane;
} unifromBuffer;

void main() {
    gl_Position = unifromBuffer.lightVP*transform.model*vec4(inPosition, 1.0);
    pos = vec3(transform.model*vec4(inPosition,1.0));
}
