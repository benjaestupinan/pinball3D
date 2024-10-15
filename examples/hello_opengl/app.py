import pyglet
import pyglet.gl as GL
import trimesh as tm
import numpy as np
import os
from pathlib import Path
import ctypes
from PIL import Image

if __name__ == "__main__":
    # esta es una ventana de pyglet.
    # le damos la resolución como parámetro
    window = pyglet.window.Window(960, 960)

    def create_vao(vertices):
        vao = GL.GLuint()
        GL.glGenVertexArrays(1, vao)
        GL.glBindVertexArray(vao)

        vbo = GL.GLuint()
        GL.glGenBuffers(1, vbo)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)

        vertices = np.array(vertices, dtype=np.float32)
        arrayType = (GL.GLfloat * len(vertices)).from_buffer(vertices)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, arrayType, GL.GL_STATIC_DRAW)

        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 6 * ctypes.sizeof(GL.GLfloat), ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(0)

        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 6 * ctypes.sizeof(GL.GLfloat), ctypes.c_void_p(3 * ctypes.sizeof(GL.GLfloat)))
        GL.glEnableVertexAttribArray(1)

        GL.glBindVertexArray(0)
        return vao

    # cargaremos el modelo del conejo de Stanford.
    # esta es la versión descargable desde Wikipedia
    # el formato STL es binario!

    ball = tm.load("Tarea3/assets/BallModel/Ball.obj")

    # no sabemos de qué tamaño es el conejo.
    # y solo podemos dibujar en nuestro cubo de referencia
    # cuyas esquinas son [-1, -1, -1] y [1, 1, 1]
    # trimesh nos facilita manipular la geometría del modelo 3D
    # en este caso:
    # 1) lo movemos hacia el origen, es decir, le restamos el centroide
    ball.apply_translation(-ball.centroid)
    # 2) puede ser que sea muy grande (o muy pequeño) para verse en la ventana.
    # así que lo escalamos de acuerdo al tamaño
    # (de acuerdo a la documentación de trimesh, el valor scale es el largo de la arista
    # más grande de la caja que contiene al conejo)
    ball.apply_scale(2.0 / ball.scale)

    # el shader de vértices solo lee la posición de cada vértice
    # cada vértice es pintado de color blanco en este shader.
    with open(Path(os.path.dirname(__file__)) / "vertex_program.glsl") as f:
        vertex_source_code = f.read()

    # y el shader de píxeles solo lee el color correspondiente al píxel
    with open(Path(os.path.dirname(__file__)) / "fragment_program.glsl") as f:
        fragment_source_code = f.read()

    vert_shader = pyglet.graphics.shader.Shader(vertex_source_code, "vertex")
    frag_shader = pyglet.graphics.shader.Shader(fragment_source_code, "fragment")
    pipeline = pyglet.graphics.shader.ShaderProgram(vert_shader, frag_shader)

    # ahora le entregaremos los datos de nuestro objeto a la GPU
    # afortunadamente trimesh tiene una función que convierte el modelo 3D
    # a la representación adecuada para uso con OpenGL :)
    ball_vertex_list = tm.rendering.mesh_to_vertexlist(ball)

    # queremos dibujar al conejo con nuestro pipeline.
    # como dibujarlo dependerá de lo que contenga cada shader del pipeline,
    # tenemos que pedirle al pipeline que reserve espacio en la GPU
    # para copiar nuestro conejo a la memoria gráfica
    ball_gpu = pipeline.vertex_list_indexed(
        # ¿cuántos vértices contiene? 
        # su quinto elemento contiene el tipo y las posiciones de los vértices.
        # vertex_list[4][0] es vec3f (sabemos que es una posición en 3D de punto flotante)
        # vertex_list[4][1] contiene las posiciones (x, y, z) de cada vértice
        # todas concatenadas en una única lista.
        # por eso el total de vértices es el largo de la lista dividido por 3.
        len(ball_vertex_list[4][1]) // 3,
        # ¿cómo se dibujarán? en este caso, como triángulos
        GL.GL_TRIANGLES,
        # su cuarto elemento contiene los índices de los vértices, es decir,
        # cuales vértices de la lista conforman cada triángulo
        ball_vertex_list[3]
    )

    # en el código anterior no se copió la información a la GPU,
    # sino que se reservó espacio en la memoria para almacenar esos datos.
    # aquí los copiamos: directamente asignamos nuestra lista de vértices
    # al contenido del atributo position que recibe el pipeline.
    # este atributo se recibe a nivel de vértice
    # más adelante veremos que hay otro
    ball_gpu.position[:] = ball_vertex_list[4][1]
    ball_gpu.normal[:] = ball_vertex_list[5][1]

    def create_cubemap(faces): # Funcion que crea el cubemap
        textureID =GL.GLuint()
        GL.glGenTextures(1, textureID)
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, textureID)

        for i, face in enumerate(faces):
            img = Image.open(face)
            img = img.convert("RGB")
            img_data = np.array(list(img.getdata()), np.uint8).flatten()
            imgWidth, imgHeight = img.size
            img_data = (GL.GLubyte * len(img_data))(*img_data)
            GL.glTexImage2D(GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL.GL_RGB, imgWidth, imgHeight, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, img_data)

        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE)

        GL.glGenerateMipmap(GL.GL_TEXTURE_CUBE_MAP)
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, textureID)

        return textureID
    
    faces = [
        "izq.jpg",
        "der.jpg",
        "front.jpg",
        "back.jpg",
        "up.jpg",
        "down.jpg"
    ]

    cubemap_texture = create_cubemap(faces)

    vao =  create_vao(ball.vertices)


    # GAME LOOP
    @window.event
    def on_draw():
        # esta función define el color con el que queda una ventana vacía
        # noten que esto es algo de OpenGL, no de pyglet
        GL.glClearColor(0.5, 0.5, 0.5, 1.0)
        # GL_LINE => Wireframe
        # GL_FILL => pinta el interior de cada triángulo
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glLineWidth(1.0)

        # lo anterior no ejecuta esos cambios en la ventana,
        # más bien, configura OpenGL para cuando se ejecuten instrucciones
        # como la siguiente:
        window.clear()

        # del mismo modo, activamos el pipeline
        # activarlo no implica graficar nada de manera inmediata
        pipeline.use()
        # hasta que le pedimos a ball_gpu que grafique sus triángulos
        # utilizando el pipeline activo
        GL.glActiveTexture(GL.GL_TEXTURE_CUBE_MAP)
        print(cubemap_texture)
        GL.glBindVertexArray(vao)
        # GL.glBindTexture(GL.GL_TEXTURE_3D, cubemap_texture)
        print("###################################")
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(ball.vertices // 6))

    # aquí comienza pyglet a ejecutar su loop.
    pyglet.app.run()
    # ¡cuando ejecutemos el programa veremos al conejo en Wireframe!
