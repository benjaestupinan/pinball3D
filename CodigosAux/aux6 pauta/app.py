import pyglet
import pyglet.gl as GL
import trimesh as tm
import numpy as np
import os
from pathlib import Path
import sys

if sys.path[0] != "":
    sys.path.insert(0, "")

# una función auxiliar para cargar shaders
from grafica.utils import load_pipeline

from grafica.arcball import Arcball
from grafica.textures import texture_2D_setup

# funciones de graficas.transformations
def translate(tx, ty, tz):
    return np.array([
        [1,0,0,tx],
        [0,1,0,ty],
        [0,0,1,tz],
        [0,0,0,1]], dtype = np.float32)

def rotationZ(theta):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    return np.array([
        [cos_theta,-sin_theta,0,0],
        [sin_theta,cos_theta,0,0],
        [0,0,1,0],
        [0,0,0,1]], dtype = np.float32)


# código de arcball.py para leer un mesh
def setupMesh(file_path, tex_pipeline, notex_pipeline, scale): 
    # dependiendo de lo que contenga el archivo a cargar,
    # trimesh puede entregar una malla (mesh)
    # o una escena (compuesta de mallas)
    # con esto forzamos que siempre entregue una escena
    asset = tm.load(file_path, force="scene")


    # de acuerdo a la documentación de trimesh, esto centra la escena
    # no es igual a trabajar con una malla directamente
    asset.rezero()

    # y esto la escala con lo que decidamos, printea tambien un cubo en el que cae todo el modelo para tener como referencia
    asset = asset.scaled(scale / asset.scale)
    print(file_path)
    print(asset.bounds)

    # aquí guardaremos las mallas del modelo que graficaremos
    vertex_lists = {}

    # con esto iteramos sobre las mallas
    for object_id, object_geometry in asset.geometry.items():
        mesh = {}

        # por si acaso, para que la malla tenga normales consistentes
        object_geometry.fix_normals(True)

        object_vlist = tm.rendering.mesh_to_vertexlist(object_geometry)

        n_triangles = len(object_vlist[4][1]) // 3

        # el pipeline a usar dependerá de si el objeto tiene textura
        # OJO: asumimos que si tiene material, tiene textura
        # pero no siempre es así.
        # print(object_id)
        # print(dir(object_geometry.visual.material))
        # print(object_geometry.visual.material.image)
        if object_geometry.visual.material.image != None:
            print('has texture')
            mesh["pipeline"] = tex_pipeline
            has_texture = True
        else:
            print('no texture')
            mesh["pipeline"] = notex_pipeline
            has_texture = False

        # inicializamos los datos en la GPU
        mesh["gpu_data"] = mesh["pipeline"].vertex_list_indexed(
            n_triangles, GL.GL_TRIANGLES, object_vlist[3]
        )

        # copiamos la posición de los vértices
        mesh["gpu_data"].position[:] = object_vlist[4][1]

        # las normales vienen en vertex_list[5]
        # las manipulamos del mismo modo que los vértices
        mesh["gpu_data"].normal[:] = object_vlist[5][1]

        # print(asset)
        # print(dir(mesh['gpu_data']))

        # con (o sin) textura es diferente el procedimiento
        # aunque siempre en vertex_list[6] viene la información de material
        if has_texture:
            # copiamos la textura
            # trimesh ya la cargó, solo debemos copiarla a la GPU
            # si no se usa trimesh, el proceso es el mismo,
            # pero se debe usar Pillow para cargar la imagen
            mesh["texture"] = texture_2D_setup(object_geometry.visual.material.image)
            # copiamos las coordenadas de textura en el parámetro uv
            mesh["gpu_data"].uv[:] = object_vlist[6][1]
        else:
            # usualmente el color viene como c4B/static en vlist[6][0], lo que significa "color de 4 bytes". idealmente eso debe verificarse
            mesh["gpu_data"].color[:] = object_vlist[6][1]
        mesh['id'] = object_id[0:-4]
        vertex_lists[object_id] = mesh
    return vertex_lists

if __name__ == "__main__":
    width = 960
    height = 960
    window = pyglet.window.Window(width, height)


    # como no todos los archivos que carguemos tendrán textura,
    # tendremos dos pipelines
    tex_pipeline = load_pipeline(   
        Path(os.path.dirname(__file__)) / "vertex_program.glsl",
        Path(os.path.dirname(__file__)) / "fragment_program.glsl",
    )

    notex_pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "vertex_program_notex.glsl",
        Path(os.path.dirname(__file__)) / "fragment_program_notex.glsl",
    )

    # vertex_lists['asset name'] = setupMesh(asset)
    vertex_lists = {}
    vertex_lists = setupMesh('./CodigosAux/aux6_assets/baseball.obj', tex_pipeline, notex_pipeline, 0.05/0.406014)
    vertex_lists = vertex_lists | setupMesh('./CodigosAux/aux6_assets/baseball bat.obj', tex_pipeline, notex_pipeline, 0.5/0.497)

    # como setear la escena
    import pymunk

    # primero generamos un espacio en el cual estarán nuestros objetos, sin esto no podremos moverlos ni hacerlos interactuar
    # la gravedad es 3 para que no caiga tan rápido
    space = pymunk.Space()
    space.gravity = (0, -3)

    # la pelota
    # la masa la googlie
    ball_mass= 0.142
    # un body en pymunk es donde se almacenan y controlan las fisicas del objeto, es aqui donde manipulamos sus variables
    ball_body = pymunk.Body(ball_mass, pymunk.moment_for_circle(ball_mass, 0.35, 0.35, (0,0)))

    from random import random
    # posición aleatoria para inicializar el "juego"
    ball_body.position = (0, 2 + 3*random())

    # el shape es lo que administra las coliciones, en este caso en un circulo
    # el tamaño lo recuperamos de los bounds del modelo
    # un argumento necesario para cada shape es el body al cual se conectará
    ball_shape = pymunk.Circle(ball_body, 0.035, (0,0))

    # aqui se agregan al espacio para que interactue con los demas elementos del sistema
    space.add(ball_body, ball_shape)

    # el bate
    # este será un polígono cuyos puntos recuperamos de los bounds del modelo
    print(vertex_lists)
    vertices = [(0.5, 0.033), (-0.5, 0.033), (-0.5, -0.033), (0.5, -0.033)]
    # es necesario cambiar el centro de masa así el bate rota desde el mango y no desde el centro
    centro_de_masa = (0.5, 0)

    bat_mass = 1
    # creamos el cuerpo igual que con la la esfera, pero considerando que esta vez es un polígono
    print(type(vertices))
    bat_body = pymunk.Body(bat_mass, pymunk.moment_for_poly(bat_mass, vertices=vertices, offset=centro_de_masa))
    bat_body.center_of_gravity = centro_de_masa
    bat_body.position = (0.4, 0)
    bat_body.angle = np.pi/2
    # bat_body.angular_velocity = -2
    bat_shape = pymunk.Poly(bat_body,
                            vertices=vertices)
    
    space.add(bat_body, bat_shape)

    # aquí iremos guardando la altura máxima para ver el puntaje
    max_h = ball_body.position[1]

    # este diccionario lo ocupamos para recuperar los cuerpos en el for del on_draw(), 
    # tiene el mismo nombre que los mesh para poder recuperarlos después
    bodies = {}
    bodies['baseball'] = ball_body
    bodies['baseball bat'] = bat_body


    # instanciamos nuestra Arcball
    arcball = Arcball(
        np.identity(4),
        np.array((width, height), dtype=float),
        1.5,
        np.array([0.0, 0.0, 0.0]),
    )

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        # print("press", x, y, button, modifiers)
        arcball.down((x, y))

    @window.event
    # def on_mouse_release(x, y, button, modifiers):
    #     print("release", x, y, button, modifiers)
    #     print(arcball.pose)

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        # print("drag", x, y, dx, dy, buttons, modifiers)
        arcball.drag((x, y))

    @window.event
    def on_key_press(key, modifier):
        # aquí activamos la rotación del bate
        if key == pyglet.window.key.SPACE:
            bat_body.angular_velocity = -15
            # bat_body.position = (0.3, 0)
    @window.event
    def on_key_release(key, modifier):
        # aquí la desactivamos para que deje de rotar
        if key == pyglet.window.key.SPACE:
            bat_body.angular_velocity = 0
            


    @window.event
    def on_draw():
        global max_h
        GL.glClearColor(0.5, 0.5, 0.5, 1.0)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glLineWidth(1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        window.clear()

        for i in range(10):
            # tenemos una fuerza opuesta a la gravedad para que el bate flote
            bat_body.force = (0, 3)
            # aquí avanzamos el tiempo dentro de la simulación, lo hacemos 10 veces por cuadro, de los cuales hay 60 por segundo
            space.step(1.0/60 / 10)

        if max_h < ball_body.position[1]:
            # actualizamos la altura máxima
            max_h = ball_body.position[1]
        
        

        for object_geometry in vertex_lists.values():
            # dibujamos cada una de las mallas con su respectivo pipeline
            pipeline = object_geometry["pipeline"]
            pipeline.use()

            pipeline["light_position"] = np.array([-1.0, 1.0, -1.0])

            # recuperamos las variables de los bodys así podemos usarlas para la transform que dibujará los modelos
            x,y = bodies[object_geometry['id']].position
            alpha = bodies[object_geometry['id']].angle
            pipeline["transform"] = (arcball.pose @ translate(x, y, 0) @ rotationZ(alpha)).reshape(16, 1, order="F")

            if "texture" in object_geometry:
                GL.glBindTexture(GL.GL_TEXTURE_2D, object_geometry["texture"])
            else:
                # esto "activa" una textura nula
                GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

            object_geometry["gpu_data"].draw(pyglet.gl.GL_TRIANGLES)

        # si la pelota baja del piso, se acaba el juego
        if(ball_body.position[1] < -1):
            print('max height: ' + str(max_h))
            pyglet.app.exit()
    pyglet.app.run()
