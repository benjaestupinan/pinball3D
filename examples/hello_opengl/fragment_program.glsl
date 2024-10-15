#version 330

// in vec3 fragColor;
in vec3 textureDir;

uniform samplerCube cubemap;

out vec3 outColor;

void main()
{
    // vec3 I = normalize(fragPosition + fragNormal);
    outColor = vec3(texture(cubemap, textureDir));
}