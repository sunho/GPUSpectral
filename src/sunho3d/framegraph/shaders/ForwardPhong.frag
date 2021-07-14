#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 pos;
layout(location = 0) out vec4 outColor;

vec3 lightPos = vec3(1.0, 1.0, 1.0);

layout(binding = 0) uniform sampler2D texSampler;
void main() {
    vec3 lightV = lightPos - pos;
    vec3 light = normalize(lightPos - pos);
    vec4 diffuse = texture(texSampler, uv);
    float dis = length(lightV);
    float lightI = 1.0/(dis*dis);
    outColor = diffuse * 0.2 +  lightI* diffuse * dot(light,normalize(normal));
    //outColor = vec4(normalize(normal),1);
    //outColor = vec4(light,1);
}
