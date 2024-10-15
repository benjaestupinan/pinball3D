import pyglet
from pyglet.gl import *
from PIL import Image
import numpy as np
import ctypes

# Tamaño del framebuffer
FRAMEBUFFER_SIZE = 512

# Crear y configurar el framebuffer
framebuffer = GLuint()
glGenFramebuffers(1, framebuffer)
glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)

# Crear y configurar la textura
texture = GLuint()
glGenTextures(1, texture)
glBindTexture(GL_TEXTURE_2D, texture)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, FRAMEBUFFER_SIZE, FRAMEBUFFER_SIZE, 0, GL_RGB, GL_UNSIGNED_BYTE, None)

glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

# Crear y configurar el renderbuffer para el depth buffer
renderbuffer = GLuint()
glGenRenderbuffers(1, renderbuffer)
glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer)
glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, FRAMEBUFFER_SIZE, FRAMEBUFFER_SIZE)

# Adjuntar el renderbuffer y la textura al framebuffer
glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, renderbuffer)
glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)

# Verificar que el framebuffer esté completo
if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
    raise Exception("Framebuffer no está completo.")

glBindFramebuffer(GL_FRAMEBUFFER, 0)

# Vertices y colores para el triángulo
vertices = np.array([
    -0.5, -0.5, 0.0, 1.0, 0.0, 0.0, # Vértice 1: posición y color (rojo)
     0.5, -0.5, 0.0, 0.0, 1.0, 0.0, # Vértice 2: posición y color (verde)
     0.0,  0.5, 0.0, 0.0, 0.0, 1.0  # Vértice 3: posición y color (azul)
], dtype=np.float32)

# Crear y configurar el VAO y VBO
VAO = GLuint()
glGenVertexArrays(1, VAO)
glBindVertexArray(VAO)

VBO = GLuint()
glGenBuffers(1, VBO)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), GL_STATIC_DRAW)

# Especificar el layout de los datos en el buffer
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * ctypes.sizeof(GLfloat), ctypes.c_void_p(0))
glEnableVertexAttribArray(0)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * ctypes.sizeof(GLfloat), ctypes.c_void_p(3 * ctypes.sizeof(GLfloat)))
glEnableVertexAttribArray(1)

glBindBuffer(GL_ARRAY_BUFFER, 0)
glBindVertexArray(0)

# Shader de vértices
vertex_shader_source = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
out vec3 ourColor;
void main()
{
    gl_Position = vec4(position, 1.0);
    ourColor = color;
}
"""

# Shader de fragmentos
fragment_shader_source = """
#version 330 core
in vec3 ourColor;
out vec4 color;
void main()
{
    color = vec4(ourColor, 1.0);
}
"""

def create_shader_program():
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex_shader, vertex_shader_source)
    glCompileShader(vertex_shader)
    if not glGetShaderiv(vertex_shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(vertex_shader))
    
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragment_shader, fragment_shader_source)
    glCompileShader(fragment_shader)
    if not glGetShaderiv(fragment_shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(fragment_shader))
    
    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)
    if not glGetProgramiv(shader_program, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(shader_program))
    
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    
    return shader_program

shader_program = create_shader_program()

def render_to_texture():
    # Enlazar el framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)
    glViewport(0, 0, FRAMEBUFFER_SIZE, FRAMEBUFFER_SIZE)
    
    # Limpiar el framebuffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # Usar el shader program
    glUseProgram(shader_program)
    
    # Enlazar el VAO y dibujar
    glBindVertexArray(VAO)
    glDrawArrays(GL_TRIANGLES, 0, 3)
    glBindVertexArray(0)
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

def save_texture(filename):
    # Leer los datos de la textura desde la GPU
    glBindTexture(GL_TEXTURE_2D, texture)
    data = (GLubyte * (FRAMEBUFFER_SIZE * FRAMEBUFFER_SIZE * 3))()
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, data)
    
    # Convertir los datos a un array de numpy
    image = np.frombuffer(data, dtype=np.uint8).reshape(FRAMEBUFFER_SIZE, FRAMEBUFFER_SIZE, 3)
    
    # Guardar la imagen usando Pillow
    img = Image.fromarray(image)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)  # Flip para corregir la orientación
    img.save(filename)
    print(f"Texture saved to {filename}")

window = pyglet.window.Window(visible=False)

@window.event
def on_draw():
    render_to_texture()
    save_texture('output.png')
    pyglet.app.exit()

pyglet.app.run()
