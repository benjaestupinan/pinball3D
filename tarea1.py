import os.path
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pyglet
import pyglet.gl as GL
import trimesh as tm
from pyglet.window import key

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
)

if sys.path[0] != '':
    sys.path.insert(0, '')

import grafica.transformations as tr

# Se crea el grafo de escena

def create_pinball(meshPinball, meshFlipper, meshObstaculo1, meshObstaculo2, meshTieFighter, meshXwing, mesh_pipeline):
    graph = nx.DiGraph(root = "pinball")

    graph.add_node(
        "pinball",
        transform = tr.rotationZ(-np.pi/2))
    graph.add_node(
        "pinball_geometry", 
        mesh = meshPinball, 
        pipeline = mesh_pipeline, 
        transform = tr.uniformScale(2.1),
        color = np.array((1.0, 0.73, 0.03))
        )
    
    graph.add_edge("pinball", "pinball_geometry")

    graph.add_node(
        "obstaculo1",
        transform = tr.translate(0, -.5, .125))
    graph.add_node(
        "obstaculo1_geometry",
        mesh = meshObstaculo1,
        pipeline = mesh_pipeline,
        transform = tr.uniformScale(.25),
        color = np.array((0.7, 0.0, 0.7))
    )

    graph.add_edge("pinball", "obstaculo1")
    graph.add_edge("obstaculo1", "obstaculo1_geometry")

    graph.add_node(
        "obstaculo1.2",
        transform = tr.translate(0, .5, .125))
    graph.add_node(
        "obstaculo1.2_geometry",
        mesh = meshObstaculo1,
        pipeline = mesh_pipeline,
        transform = tr.uniformScale(.25),
        color = np.array((0.7, 0.0, 0.7))
    )

    graph.add_edge("pinball", "obstaculo1.2")
    graph.add_edge("obstaculo1.2", "obstaculo1.2_geometry")


    graph.add_node(
        "obstaculo2",
        transform = tr.translate(-0.75, 0, .175) @ tr.rotationZ(3*np.pi/4))
    graph.add_node(
        "obstaculo2_geometry",
        mesh = meshObstaculo2,
        pipeline = mesh_pipeline,
        transform = tr.uniformScale(.6),
        color = np.array((0.0, 0.7, 0.0))
    )

    graph.add_edge("pinball", "obstaculo2")
    graph.add_edge("obstaculo2", "obstaculo2_geometry")

    graph.add_node(
        "TieFighter",
        transform = tr.translate(2.5,0,0))
    graph.add_node(
        "TieFighter_geometry",
        transform = tr.uniformScale(0.3),
        mesh = meshTieFighter,
        pipeline = mesh_pipeline,
        color = np.array((.3, .3, .3)))
    
    graph.add_edge("pinball", "TieFighter")
    graph.add_edge("TieFighter", "TieFighter_geometry")

    graph.add_node(
        "X-Wing",
        transform = tr.translate(-2.5, 0, 0) @ tr.rotationZ(np.pi/2))
    graph.add_node(
        "x-wing_geometry",
        transform = tr.uniformScale(0.3),
        mesh = meshXwing,
        pipeline = mesh_pipeline,
        color = np.array((.9, .9, .9)))
    
    graph.add_edge("pinball", "X-Wing")
    graph.add_edge("X-Wing", "x-wing_geometry")

    graph.add_node(
        "flipper_der",
        transform = tr.translate(1.15, .5, .125) @ tr.scale(1.0, -1.0, 1.0) @ tr.rotationZ(np.pi))
    graph.add_node(
        "flipper_der_geometry",
        transform = tr.uniformScale(0.3),
        mesh = meshFlipper,
        pipeline = mesh_pipeline,
        color = np.array((0.0, 0.59, 0.78))
        )
    
    graph.add_edge("pinball", "flipper_der")
    graph.add_edge("flipper_der", "flipper_der_geometry")

    graph.add_node(
        "flipper_izq",
        transform = tr.translate(1.15, -.5, 0.125) @ tr.rotationZ(-np.pi))
    graph.add_node(
        "flipper_izq_geometry",
        transform = tr.uniformScale(0.3),
        mesh = meshFlipper,
        pipeline = mesh_pipeline,
        color = np.array((0.0, 0.59, 0.78))
        )
    
    graph.add_edge("pinball", "flipper_izq")
    graph.add_edge("flipper_izq", "flipper_izq_geometry")

    return graph

def update_pinball(dt, window):
    window.program_state["total_time"] += dt
    total_time = window.program_state["total_time"]

    graph = window.program_state["scene_graph"]

    graph.nodes["TieFighter"]["transform"] = tr.rotationZ(-total_time*2.5) @ tr.translate(2.5, 0, 0)
    graph.nodes["X-Wing"]["transform"] = tr.rotationZ(total_time*3.5) @ tr.translate(-2.5, 0, 1) @ tr.rotationZ(-np.pi/2)

if __name__ == "__main__":    
    
    width = 960
    height = 960

    window = pyglet.window.Window(width, height)

    meshPinball = tm.load("Tarea1/assets/Pinball.stl")
    model_scale_pinball = tr.uniformScale(2.0 / meshPinball.scale)
    model_translate_pinball = tr.translate(*-meshPinball.centroid)
    meshPinball.apply_transform(model_scale_pinball @ model_translate_pinball)

    meshObstaculo1 = tm.load("Tarea1/assets/obstaculo1.stl")
    model_scale_obstaculo1 = tr.uniformScale(2.0 / meshObstaculo1.scale)
    model_translate_obstaculo1 = tr.translate(*-meshObstaculo1.centroid)
    meshObstaculo1.apply_transform(model_scale_obstaculo1 @ model_translate_obstaculo1)

    meshObstaculo2 = tm.load("Tarea1/assets/obstaculo2.stl")
    model_scale_obstaculo2 = tr.uniformScale(2.0 / meshObstaculo2.scale)
    model_translate_obstaculo2 = tr.translate(*-meshObstaculo2.centroid)
    meshObstaculo2.apply_transform(model_scale_obstaculo2 @ model_translate_obstaculo2)

    meshTieFighter = tm.load("Tarea1/assets/TieFighter.stl")
    model_scale_tiefighter = tr.uniformScale(2.0 / meshTieFighter.scale)
    model_translate_tiefighter = tr.translate(*-meshTieFighter.centroid)
    meshTieFighter.apply_transform(model_scale_tiefighter @ model_translate_tiefighter)

    meshXwing = tm.load("Tarea1/assets/x-wing.stl")
    model_scale_xwing = tr.uniformScale(2.0 / meshXwing.scale)
    model_translate_xwing = tr.translate(*-meshXwing.centroid)
    meshXwing.apply_transform(model_scale_xwing @ model_translate_xwing)

    meshFlipper = tm.load("Tarea1/assets/flipper.stl")
    model_scale_flipper = tr.uniformScale(2.0 / meshFlipper.scale)
    meshFlipper.apply_transform(model_scale_flipper)

    with open(Path(os.path.dirname(__file__)) / "Tarea1/vertex_program.glsl") as f:
        vertex_source_code = f.read()
    
    with open(Path(os.path.dirname(__file__)) / "Tarea1/fragment_program.glsl") as f:
        fragment_source_code = f.read()
    
    vert_shader = pyglet.graphics.shader.Shader(vertex_source_code, "vertex")
    frag_shader = pyglet.graphics.shader.Shader(fragment_source_code, "fragment")
    pipeline = pyglet.graphics.shader.ShaderProgram(vert_shader,frag_shader)

    pinball_vertex_list = tm.rendering.mesh_to_vertexlist(meshPinball)
    pinball_mesh_gpu = pipeline.vertex_list_indexed(
        len(pinball_vertex_list[4][1]) // 3,
        GL.GL_TRIANGLES,
        pinball_vertex_list[3]
    )
    pinball_mesh_gpu.position[:] = pinball_vertex_list[4][1]

    obstaculo1_vertex_list = tm.rendering.mesh_to_vertexlist(meshObstaculo1)
    obstaculo1_mesh_gpu = pipeline.vertex_list_indexed(
        len(obstaculo1_vertex_list[4][1]) // 3,
        GL.GL_TRIANGLES,
        obstaculo1_vertex_list[3]
    )
    obstaculo1_mesh_gpu.position[:] = obstaculo1_vertex_list[4][1]

    obstaculo2_vertex_list = tm.rendering.mesh_to_vertexlist(meshObstaculo2)
    obstaculo2_mesh_gpu = pipeline.vertex_list_indexed(
        len(obstaculo2_vertex_list[4][1]) // 3,
        GL.GL_TRIANGLES,
        obstaculo2_vertex_list[3]
    )
    obstaculo2_mesh_gpu.position[:] = obstaculo2_vertex_list[4][1]

    tiefighter_vertex_list = tm.rendering.mesh_to_vertexlist(meshTieFighter)
    tiefighter_mesh_gpu = pipeline.vertex_list_indexed(
        len(tiefighter_vertex_list[4][1]) // 3,
        GL.GL_TRIANGLES,
        tiefighter_vertex_list[3]
    )
    tiefighter_mesh_gpu.position[:] = tiefighter_vertex_list[4][1]

    xwing_vertex_list = tm.rendering.mesh_to_vertexlist(meshXwing)
    xwing_mesh_gpu = pipeline.vertex_list_indexed(
        len(xwing_vertex_list[4][1]) // 3,
        GL.GL_TRIANGLES,
        xwing_vertex_list[3]
    )
    xwing_mesh_gpu.position[:] = xwing_vertex_list[4][1]

    flipper_vertex_list = tm.rendering.mesh_to_vertexlist(meshFlipper)
    flipper_mesh_gpu = pipeline.vertex_list_indexed(
        len(flipper_vertex_list[4][1]) // 3,
        GL.GL_TRIANGLES,
        flipper_vertex_list[3]
    )
    flipper_mesh_gpu.position[:] = flipper_vertex_list[4][1]

    graph = create_pinball(pinball_mesh_gpu,flipper_mesh_gpu,obstaculo1_mesh_gpu,obstaculo2_mesh_gpu,tiefighter_mesh_gpu,xwing_mesh_gpu,pipeline)


    #Se definen las vistas
    top_view = tr.lookAt(np.array([0, 0, 5]), np.array([0, 0, 0]), np.array([0, 1, 0]))
    pov =  tr.lookAt(np.array([0, -5, 5]), np.array([0, 0, 0]), np.array([0, 0, 1]))
    side_view = tr.lookAt(np.array([0, -5, 0]), np.array([0, 0, 0]), np.array([0, 1, 1]))

    ortogonal = tr.ortho(-3, 3, -3, 3, 0.1, 100)
    perspective = tr.perspective(45, float(width) / float(height), 0.1, 100)

    TopView = "TopView"
    POV = "POV"
    SideView = "side_view"

    window.program_state = {
        "scene_graph": graph,
        "total_time": 0.0,
        "view": tr.lookAt(np.array([0, 0, 5]), np.array([0, 0, 0]), np.array([0, 1, 0])),
        "projection": ortogonal,
        "view_name": TopView,
        "flipper_der": {"angulo": 0,
                        "moving": False},
        "flipper_izq": {"angulo": 0,
                        "moving": False},
        }
    
    def ChangeView():
        if window.program_state["view_name"] == TopView:
            window.program_state["view"] = pov
            window.program_state["view_name"] = POV
            window.program_state["projection"] = perspective
        elif window.program_state["view_name"] == POV:
            window.program_state["view"] = side_view
            window.program_state["view_name"] = SideView
            window.program_state["projection"] = ortogonal
        elif window.program_state["view_name"] == SideView:
            window.program_state["view"] = top_view
            window.program_state["view_name"] = TopView
            window.program_state["projection"] = ortogonal             
        
    def MoveFlipperIzq():
        window.program_state["flipper_izq"]["moving"] = True            

    def MoveFlipperDer():
        window.program_state["flipper_der"]["moving"] = True

    def ReturnFlipperIzq():
        window.program_state["flipper_izq"]["moving"] = False

    def ReturnFlipperDer():
        window.program_state["flipper_der"]["moving"] = False

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == key.C:
            ChangeView()
        if symbol == key.A:
            MoveFlipperIzq()
        if symbol == key.D:
            MoveFlipperDer()
            

    @window.event
    def on_key_release(symbol, modifiers):
        if symbol == key.A:
            ReturnFlipperIzq()
        elif symbol == key.D:
             ReturnFlipperDer()
            

    @window.event
    def on_draw():
        GL.glClearColor(0.1, 0.1, 0.1, 0.1)
        GL.glLineWidth(2.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)


        window.clear()
        
        pipeline.use()

        pipeline["view"] = window.program_state["view"].reshape(16, 1, order="F")
        pipeline["projection"] = window.program_state["projection"].reshape(16, 1, order="F")

        graph = window.program_state["scene_graph"]

        if window.program_state["flipper_izq"]["moving"] and window.program_state["flipper_izq"]["angulo"] < np.pi/3:
            graph.nodes["flipper_izq"]["transform"] @= tr.rotationZ(np.pi/16)
            window.program_state["flipper_izq"]["angulo"] += np.pi/16

        if not window.program_state["flipper_izq"]["moving"] and window.program_state["flipper_izq"]["angulo"] > 0:
            graph.nodes["flipper_izq"]["transform"] @= tr.rotationZ(-np.pi/16)
            window.program_state["flipper_izq"]["angulo"] -= np.pi/16

        if window.program_state["flipper_der"]["moving"] and window.program_state["flipper_der"]["angulo"] < np.pi/3:
            graph.nodes["flipper_der"]["transform"] @= tr.rotationZ(np.pi/16)
            window.program_state["flipper_der"]["angulo"] += np.pi/16

        if not window.program_state["flipper_der"]["moving"] and window.program_state["flipper_der"]["angulo"] > 0:
            graph.nodes["flipper_der"]["transform"] @= tr.rotationZ(-np.pi/16)
            window.program_state["flipper_der"]["angulo"] -= np.pi/16


        root_key = graph.graph["root"]
        edges = list(nx.edge_dfs(graph, source=root_key))

        transformations = {root_key: graph.nodes[root_key]["transform"]}

        for src, dst in edges:
            current_node = graph.nodes[dst]

            if not dst in transformations:
                dst_transform = current_node["transform"]
                transformations[dst] = transformations[src] @ dst_transform

            if "mesh" in current_node:

                pipeline["transform"] = transformations[dst].reshape(16, 1, order="F")

                for attr in current_node.keys():
                    if attr in ("mesh", "pipeline", "transform"):
                        continue

                    current_attr = current_node[attr]
                    current_size = current_node[attr].shape[0]

                    if len(current_node[attr].shape) > 1:
                        current_size = current_size * current_node[attr].shape[1]

                    pipeline[attr] = current_node[attr].reshape(
                        current_size, 1, order="F"
                    )

                current_node["mesh"].draw(pyglet.gl.GL_TRIANGLES)
                print(current_node["mesh"])

    pyglet.clock.schedule_interval(update_pinball, 1 / 60.0, window)
    pyglet.app.run()