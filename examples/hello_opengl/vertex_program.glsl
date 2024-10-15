#version 330
in vec3 position;
in vec3 normal;
//in vec3 color;

// out vec3 fragColor;
out vec3 textureDir;

void main()
{
    textureDir = vec3(normalize(normal));
    gl_Position = vec4(position, 1.0f);
}