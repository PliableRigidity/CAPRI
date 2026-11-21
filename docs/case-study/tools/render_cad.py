"""Render the chassis CAD exports (assets/*.stl) with MuJoCo's offscreen renderer.

Geometry only: colours and lighting are chosen here and say nothing about the printed robot.
Views are written to tools/.cache/ and composed into figures/cad-chassis.png by make_figures.py.
Run from the repository root:  python docs/case-study/tools/render_cad.py
"""
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
CACHE = Path(__file__).resolve().parent / ".cache"

XML = """
<mujoco>
  <compiler meshdir="{meshdir}"/>
  <visual>
    <global offwidth="1600" offheight="1000"/>
    <quality shadowsize="4096" offsamples="8"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.55 0.55 0.55" specular="0.1 0.1 0.1"/>
    <rgba haze="1 1 1 1"/>
    <map shadowclip="0.22" znear="0.02"/>
  </visual>
  <asset>
    <mesh name="bot" file="{file}" scale="0.001 0.001 0.001"/>
    <texture name="sky" type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="64" height="64"/>
    <material name="body" rgba="{rgba}" specular="0.25" shininess="0.4"/>
    <material name="floor" rgba="0.93 0.94 0.95 1" specular="0" reflectance="0"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 1" dir="-0.25 0.35 -0.9" diffuse="0.75 0.75 0.75" castshadow="true"/>
    <geom type="plane" size="2 2 0.01" pos="0 0 {floor_z}" material="floor"/>
    <geom type="mesh" mesh="bot" material="body" pos="0 0 0"/>
  </worldbody>
</mujoco>
"""


def render(stl, rgba, views, out):
    probe = mujoco.MjModel.from_xml_string(XML.format(meshdir=(ROOT / "assets").as_posix(), file=stl, rgba=rgba, floor_z=0))
    # mesh vertices in the mesh frame; MuJoCo recentres meshes, so read the geom's bounding box instead
    data0 = mujoco.MjData(probe)
    mujoco.mj_forward(probe, data0)
    verts = probe.mesh_vert.reshape(-1, 3) @ data0.geom_xmat[1].reshape(3, 3).T + data0.geom_xpos[1]   # world frame
    low = float(verts[:, 2].min())
    model = mujoco.MjModel.from_xml_string(XML.format(meshdir=(ROOT / "assets").as_posix(), file=stl, rgba=rgba, floor_z=low - 0.0005))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=1000, width=1600)
    images = []
    for azimuth, elevation, distance in views:
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = data.geom_xpos[1]
        camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
        renderer.update_scene(data, camera)
        images.append(renderer.render().copy())
    renderer.close()
    for index, image in enumerate(images):
        Image.fromarray(image).save(CACHE / f"{out}_view{index}.png", optimize=True)
    print(out, len(images), "views; extent mm", np.round((verts.max(0) - verts.min(0)) * 1000, 1))


if __name__ == "__main__":
    CACHE.mkdir(exist_ok=True)
    render("BattleBot v3.2.stl", "0.16 0.45 0.80 1", [(55, -24, 0.34), (90, -89, 0.40)], "cad_autobot")
    render("Opp_Chassis.stl", "0.80 0.25 0.20 1", [(55, -24, 0.30)], "cad_last_year")
