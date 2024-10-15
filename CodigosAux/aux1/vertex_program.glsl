#version 330
in vec3 position;
in vec3 color;

uniform float time;

out vec3 fragColor;

void main()
{
    fragColor = vec3(color.x * time, color.y * time, color.z * time);
    gl_Position = vec4(position.x, position.y, position.z, 1.0f);
}