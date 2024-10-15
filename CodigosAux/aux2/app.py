import pyglet
import pyglet.gl as GL
import trimesh as tm
import numpy as np
import os
from pathlib import Path
from time import time
from pyglet.window import key

def frustum(left, right, bottom, top, near, far):
    r_l = right - left
    t_b = top - bottom
    f_n = far - near
    return np.array([
        [ 2 * near / r_l,
        0,
        (right + left) / r_l,
        0],
        [ 0,
        2 * near / t_b,
        (top + bottom) / t_b,
        0],
        [ 0,
        0,
        -(far + near) / f_n,
        -2 * near * far / f_n],
        [ 0,
        0,
        -1,
        0]], dtype = np.float32)

def perspective(fovy, aspect, near, far):
    halfHeight = np.tan(np.pi * fovy / 360) * near
    halfWidth = halfHeight * aspect
    return frustum(-halfWidth, halfWidth, -halfHeight, halfHeight, near, far)

def lookAt(eye, at, up):

    forward = (at - eye)
    forward = forward / np.linalg.norm(forward)

    side = np.cross(forward, up)
    side = side / np.linalg.norm(side)

    newUp = np.cross(side, forward)
    newUp = newUp / np.linalg.norm(newUp)

    return np.array([
            [side[0],       side[1],    side[2], -np.dot(side, eye)],
            [newUp[0],     newUp[1],   newUp[2], -np.dot(newUp, eye)],
            [-forward[0], -forward[1], -forward[2], np.dot(forward, eye)],
            [0,0,0,1]
        ], dtype = np.float32)

def translate(tx, ty, tz):
    return np.array([
        [1,0,0,tx],
        [0,1,0,ty],
        [0,0,1,tz],
        [0,0,0,1]], dtype = np.float32)

def scale(sx, sy, sz):
    return np.array([
        [sx,0,0,0],
        [0,sy,0,0],
        [0,0,sz,0],
        [0,0,0,1]], dtype = np.float32)

def rotationX(theta):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    return np.array([
        [1,0,0,0],
        [0,cos_theta,-sin_theta,0],
        [0,sin_theta,cos_theta,0],
        [0,0,0,1]], dtype = np.float32)

class Controller:
    def __init__(self):
        self.scaleZ = 1
        self.currentColor = 0
        self.colors = [
            [1,0,0],[0,1,0],[0,0,1]
        ]
        self.rotating = False
        self.alpha = 0
        self.x = 0
        self.y = 0
        self.z = 0

    def scaleUp(self):
        self.scaleZ += 0.3
    def scaleDown(self):
        self.scaleZ -= 0.3
    def getColor(self):
        return self.colors[self.currentColor]
    def changeColor(self):
        self.currentColor = (self.currentColor+1)%len(self.colors)
    def startRotation(self):
        self.rotating = True
    def stopRotation(self):
        self.rotating = False
    def moveLeft(self):
        self.y -= 0.2
    def moveRight(self):
        self.y += 0.2


if __name__ == "__main__":
    # esta es una ventana de pyglet.
    # le damos la resolución como parámetro
    # try:
    #     # si queremos más calidad visual (a un costo, claro)
    #     # podemos activar antialiasing en caso de que esté disponible
    #     config = pyglet.gl.Config(sample_buffers=1, samples=4)
    #     window = pyglet.window.Window(960, 960, config=config)
    # except pyglet.window.NoSuchConfigException:
    #     # si no está disponible, hacemos una ventana normal
    window = pyglet.window.Window(960, 960)

    controller = Controller()

    @window.event
    def on_key_press(symbol, modifier):
        # print(symbol)
        if(key.A == symbol):
            print('aprete la A')
        if(key.LEFT == symbol):
            controller.moveLeft()
        if(key.RIGHT == symbol):
            controller.moveRight()
        if(key.UP == symbol):
            controller.scaleUp()
        if(key.DOWN == symbol):
            controller.scaleDown()
        if(key.ENTER ==  symbol):
            controller.changeColor()
        if(key.SPACE == symbol):
            controller.startRotation()
    @window.event
    def on_key_release(symbol, modifier):
        # print(symbol)
        if(key.A == symbol):
            print('solté la A')
        if(key.SPACE):
            controller.stopRotation()

    # cargaremos el modelo del conejo de Stanford.
    # esta es la versión descargable desde Wikipedia
    # el formato STL es binario!
    bunny = tm.load("assets/Stanford_Bunny.stl")

    # no sabemos de qué tamaño es el conejo.
    # y solo podemos dibujar en nuestro cubo de referencia
    # cuyas esquinas son [-1, -1, -1] y [1, 1, 1]
    # trimesh nos facilita manipular la geometría del modelo 3D
    # en este caso:
    # 1) lo movemos hacia el origen, es decir, le restamos el centroide
    bunny.apply_translation(-bunny.centroid)
    # 2) puede ser que sea muy grande (o muy pequeño) para verse en la ventana.
    # así que lo escalamos de acuerdo al tamaño
    # (de acuerdo a la documentación de trimesh, el valor scale es el largo de la arista
    # más grande de la caja que contiene al conejo)
    bunny.apply_scale(2.0 / bunny.scale)

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
    bunny_vertex_list = tm.rendering.mesh_to_vertexlist(bunny)

    # queremos dibujar al conejo con nuestro pipeline.
    # como dibujarlo dependerá de lo que contenga cada shader del pipeline,
    # tenemos que pedirle al pipeline que reserve espacio en la GPU
    # para copiar nuestro conejo a la memoria gráfica
    bunny_gpu = pipeline.vertex_list_indexed(
        # ¿cuántos vértices contiene? 
        # su quinto elemento contiene el tipo y las posiciones de los vértices.
        # vertex_list[4][0] es vec3f (sabemos que es una posición en 3D de punto flotante)
        # vertex_list[4][1] contiene las posiciones (x, y, z) de cada vértice
        # todas concatenadas en una única lista.
        # por eso el total de vértices es el largo de la lista dividido por 3.
        len(bunny_vertex_list[4][1]) // 3,
        # ¿cómo se dibujarán? en este caso, como triángulos
        GL.GL_TRIANGLES,
        # su cuarto elemento contiene los índices de los vértices, es decir,
        # cuales vértices de la lista conforman cada triángulo
        bunny_vertex_list[3]
    )

    # en el código anterior no se copió la información a la GPU,
    # sino que se reservó espacio en la memoria para almacenar esos datos.
    # aquí los copiamos: directamente asignamos nuestra lista de vértices
    # al contenido del atributo position que recibe el pipeline.
    # este atributo se recibe a nivel de vértice
    # más adelante veremos que hay otro
    bunny_gpu.position[:] = bunny_vertex_list[4][1]

    # GAME LOOP
    @window.event
    def on_draw():
        # esta función define el color con el que queda una ventana vacía
        # noten que esto es algo de OpenGL, no de pyglet
        GL.glClearColor(0.5, 0.5, 0.5, 1.0)
        # GL_LINE => Wireframe
        # GL_FILL => pinta el interior de cada triángulo
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        GL.glLineWidth(1.0)

        # lo anterior no ejecuta esos cambios en la ventana,
        # más bien, configura OpenGL para cuando se ejecuten instrucciones
        # como la siguiente:
        window.clear()

        # del mismo modo, activamos el pipeline
        # activarlo no implica graficar nada de manera inmediata
        pipeline.use()
        # hasta que le pedimos a bunny_gpu que grafique sus triángulos
        # utilizando el pipeline activo

        if(controller.rotating):
            controller.alpha += 0.1

        pipeline['translate'] = translate(controller.x, controller.y, controller.z).reshape(16, 1, order='F')
        pipeline['scale'] = rotationX(controller.alpha).reshape(16, 1, order='F')
        pipeline['rotation'] = scale(1, 1, controller.scaleZ).reshape(16, 1, order='F')

        pipeline['view'] = lookAt(
            np.array([3, 3, 3]),
            np.array([0, 0, 0]), 
            np.array([0, 0, 1])
        ).reshape(16, 1, order='F')
        pipeline['projection'] = perspective(45, window.width/window.height, 0.001, 100).reshape(16, 1, order='F')
        pipeline['color'] = controller.getColor()
        bunny_gpu.draw(pyglet.gl.GL_TRIANGLES)
        # 
        # pipeline['transform'] = identity().reshape(16, 1, order='F')
        # pared.draw()
        # transf=rot
        # palito.draw()

    # aquí comienza pyglet a ejecutar su loop.
    pyglet.app.run()
    # ¡cuando ejecutemos el programa veremos al conejo en Wireframe!
