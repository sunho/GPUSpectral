#version 450

layout(location = 0) in vec3 inPos;

layout(binding = 1) uniform ShadowGenUniformBuffer {
    mat4 lightMVP;
    vec3 lightPos;
    float farPlane;
} unifromBuffer;

void main() {
    float lightDistance = length(inPos - unifromBuffer.lightPos);
    lightDistance = lightDistance / unifromBuffer.farPlane;
    gl_FragDepth = lightDistance;
}
