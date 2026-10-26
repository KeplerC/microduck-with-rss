"""Home desk set, metres/kg. No robot and no routine. See REPORT.md for uncertainty.

World z=0.75 is the desk surface; x is roughly image-right and +y points
away from the seated viewer. The phone camera is held at the fitted pose.
"""
from pathlib import Path
import json
import numpy as np
import robosimstudio as rss
from robosimstudio.core.render.finish import Finish
from robosimstudio.core.render.spec import RenderSpec

ROOT = Path(__file__).resolve().parent
A = ROOT / 'assets'
LAYOUT = json.loads((A / 'layout.json').read_text())
CALIBRATION = json.loads((A / 'calibration.json').read_text())
SETTLED = json.loads((A / 'settled_poses.json').read_text())
DESK_Z = 0.75
MAT_Z = DESK_Z + 0.003

def world(p):
    return (float(p[0]), float(p[1]), float(p[2]) + DESK_Z)

def asset(name, *, photo=False, source=False, colliders=()):
    return rss.AssetSpec(
        path=str(A / (name + '.obj')),
        texture=str(A / 'can_albedo_source.png') if source else (str(ROOT / 'ref' / 'plate_home.png') if photo else ''),
        material_colors=(name == 'pen'),
        collision_paths=tuple(str(A / f) for f in colliders),
    )

# Same 50 Hz control convention as Microduck, with extra substeps for thin props.
PHYSICS = rss.ArrangementSpec(
    control_hz=50, substeps=10,
    backends=(('*', rss.backend('mujoco', nconmax=8192, njmax=16384)),),
    collision=rss.CollisionSpec(rigid_gap=0.0002, contact_ke=100000, contact_kd=1000, rigid_contact_max=16384),
)
scene = rss.Scene(cell='void', name='home_desk', physics=PHYSICS,
    render=RenderSpec(environment='lab', scenery='', quality='final',
        key_intensity=1.1, ambient=(0.8,0.8,0.8), light_direction=(0,-0.25,-1), enable_textures=True))
# 1.30 x 0.80 m; extends beyond all visible near and side edges.
desk = scene.add(rss.box(name='desk_top', half=(.65,.40,.0125),
    at=(0,-.245,DESK_Z-.0125), static=True, friction=.7,
    color=(.68,.65,.59), finish=Finish(roughness=.78)))
mat = scene.add(rss.body(name='cat_mat', asset=asset('cat_mat',photo=True,colliders=('mat_contact_hull.obj',)),
    at=world((0,0,0)), static=True, friction=1.0, color=(1,1,1), finish=Finish(roughness=.93)))
pad = scene.add(rss.body(name='legal_pad',asset=asset('legal_pad',photo=True),
    at=world((0,0,0)),static=True,friction=.65,color=(1,1,1),finish=Finish(roughness=.9)))
rolled_pages = scene.add(rss.body(name='rolled_pages',asset=asset('rolled_pages',photo=True),
    at=world((0,0,0)),static=True,friction=.65,color=(1,1,1),finish=Finish(roughness=.9)))
tablet = scene.add(rss.body(name='tablet',asset=asset('tablet',photo=True),
    at=world((0,0,0)),static=True,friction=.7,color=(1,1,1),finish=Finish(roughness=.7)))
can_x,can_y,can_r,can_h = LAYOUT['can']
can = scene.add(rss.body(name='can',asset=asset('can',photo=True,source=True,colliders=('can_contact.obj',)),
    at=world((can_x,can_y,.0032)),mass=.20,friction=.55,contact_gap=.0002,
    color=(1,1,1),finish=Finish(roughness=.42,metallic=.15)))
bowl = scene.add(rss.body(name='bowl',asset=asset('bowl',colliders=LAYOUT['bowl_colliders']),
    at=tuple(SETTLED['bowl'][:3]),quat=tuple(SETTLED['bowl'][3:]),mass=.220,
    friction=.65,contact_gap=.0002,color=(.36,.365,.34),finish=Finish(roughness=.28)))
pen = scene.add(rss.body(name='pen',asset=asset('pen',colliders=('pen_contact.obj','pen_clip_contact.obj','pen_head_contact.obj')),
    at=world(LAYOUT['pen_origin']),quat=tuple(LAYOUT['pen_quat']),mass=.012,
    friction=.55,contact_gap=.00015,finish=Finish(roughness=.53)))
phone = rss.CameraSpec(name='phone',width=1824,height=1368,
    fov_y=CALIBRATION['fov_y'],eye=world(CALIBRATION['eye']),
    look_at=world(CALIBRATION['look_at']),up=tuple(CALIBRATION['up']),
    background=str(ROOT/'ref/plate_home.png'))
scene.add(phone)
scene.add(rss.CameraSpec(name='overview',width=1280,height=960,fov_y=50,
    eye=(.7,-1.10,1.65),look_at=(0,-.12,.75)))

# Keep the inside wall visible beside the rim in camera-plate composition.
from compositing import use_shell_segmentation
if __name__ != 'home_twin':  # home.py adds the duck and a ball to `scene` and builds it itself
    sim = use_shell_segmentation(scene.build(control_mode='joint_position'))
