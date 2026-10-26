# SPDX-License-Identifier: Apache-2.0
"""Play with the Microduck at home

The duck in a real-to-sim twin of a desk at home: pick it up, walk it with the keys, kick a ball into the bowl.

The desk is a twin Codex built on RoboSimStudio's live page from one phone
photo (``home_desk/``, its ``REPORT.md`` says how): the photo's camera solved
from known points, every thing on the desk a collider at its real size, the
bowl hollow and lying on its side as it was. This page loads that twin's own
``scene.py`` and puts the duck and a ball on the desk.

Click the duck and drag it where you like: the handle takes it by the head, and
it squawks while its feet are off the desk. **W A S D** walk and turn it,
**Space** lines it up on the ball and kicks it where it faces (into the bowl,
when the bowl is roughly that way), **E** walks it to the sock and picks the sock
up in its beak (and puts it down again), and **Q** quacks. It watches the ball while
it stands, cheers when the ball ends up in the bowl, and hangs its head when a
kick stops short. Pass another twin's folder to play somewhere else:
``uv run home.py path/to/twin``, and ``--ducks=4`` for a flock, each duck in its own colours and with a routine
of its own (flock.py).
"""

import math
import runpy
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import robosimstudio as rss
from robosimstudio.specs.physics import backend

from duck import DuckSpec, _rot
from keys import keyboard
from play import Play
from sock import sock

HERE = Path(__file__).resolve().parent
TWIN = HERE / "home_desk"
DUCK_AT, DUCK_YAW = (0.24, 0.07), -math.pi / 2  # at the back of the desk, behind the notepad, facing the camera
FLOCK = (
    # where it starts and potters about [m], heading [rad], colours (assets.COLOURS), what it does on its own
    # (flock.KINDS), how far it strays [m]; all on the mat (flock.DESK)
    ((-0.15, -0.25), 0.3, "classic", "wander", 0.04),  # the one the keys steer, at the front of the mat
    ((-0.14, -0.12), -math.pi / 2, "mint", "peck", 0.07),  # beside the bowl
    ((0.21, -0.21), -math.pi / 2, "charcoal", "sit", 0.07),  # on the notepad: sitting puts a duck 4-6 cm back
    # rolls towards the camera from the middle of the mat, then walks back; that roll lands on its feet every time,
    # where one that sets off along the desk, or from the bare desk, comes out short, askew or on its back
    ((-0.02, -0.05), -math.pi / 2, "lavender", "roll", 0.06),
)
"""A flock (``--ducks N``, up to four): each in its own colours, each with its own way of passing the time."""
FLOCK_BALL_AT = (0.33, -0.20)  # on the mat past the notepad, out of the ducks' way
BALL_AT = (0.0, -0.02)  # on the mat, at its back, in line with the end of the notepad's roll
BALL_R = 0.0225
ROLLING = 4e-4  # the ball's rolling resistance on the mat [N m]
SOCK_AT, SOCK_YAW = (-0.03, -0.20), math.pi / 2  # a folded sock on the mat by the notepad's edge, lying along y

SOCK_THROUGH = ("cat_mat",)  # the desk mat's shapes, a 3 mm shell over the desk: the sock lies on the desk itself


def sock_lies_on_the_desk(sim, names) -> None:
    """Let the sock's particles pass through thin shells (shapes whose label starts with one of ``names``).

    The duck and the ball still stand on them; MuJoCo's own contacts are untouched. The sock rests
    on the solid under them instead, 25 mm of desk, 3 mm lower. On the 3 mm mat, a bite that pressed
    the sock flat pushed particles out under it, where they stuck between the mat and the desk and
    held the sock down while the beak lifted it.
    """
    import newton

    flags = sim.model.shape_flags.numpy()
    bit = int(newton.ShapeFlags.COLLIDE_PARTICLES)
    for i, label in enumerate(sim.model.shape_label):
        if str(label).startswith(tuple(names)):
            flags[i] &= ~bit
    sim.model.shape_flags.assign(flags)


def sock_collides_soft_only(sim) -> None:
    """Give the sock's one-way coupler a collision pass that finds the sock's contacts and nothing else.

    The coupler collides the whole model every substep, every rigid pair on the desk included: the
    bowl's 288 hull pieces against the mat, the notepad against its roll, the duck's feet against the
    mat and the roll. Nothing reads those rigid contacts (MuJoCo finds its own, and VBD beside an
    external rigid solver reads only the soft ones), and Newton 1.7's mesh-against-mesh kernel can
    hang the GPU on them: a warp still checking its loop reads the counter thread 0 has already reset
    for the next pair, and spins on it for good. Dragging the duck about hung the page 3 times in 3,
    and headless drags 6 times in 33; with this, 0 in 80. The soft pass pairs every particle with
    every shape whatever the pair list, so an empty list gives the sock the same contacts, record for
    record over 30,000 substeps of walking and biting, and the step runs at 121 control steps a
    second against 64.
    """
    import newton
    import warp as wp

    solver = sim.pipeline.program.solver
    full = solver.collision
    soft = newton.CollisionPipeline(
        sim.model,
        broad_phase="explicit",
        shape_pairs_filtered=wp.zeros(0, dtype=wp.vec2i, device=sim.model.device),
        soft_contact_gap=float(full.soft_contact_gap),
        enable_rigid_soft_full_surface_contact=bool(full.enable_rigid_soft_full_surface_contact),
    )
    # It writes into the coupler's own buffer, the one the soft solver reads; the step is captured anew.
    if soft.soft_contact_max > solver.contacts.soft_contact_max:
        raise RuntimeError("the soft-only collision pass needs a larger contact buffer than the coupler's")
    solver.collision = soft
    sim.pipeline.invalidate_graph()


SOCK_SOLVER = backend("vbd", iterations=15, friction_epsilon=1e-3, particle_enable_self_contact=False)
"""The sock's solver: Newton's VBD, one step per 5 ms MuJoCo step. Friction is smoothed only
below 1 mm/s, so the sock lies still instead of creeping. Self-contact is off: a
folded bundle does not fold onto itself here, and its per-step collision pass was most of
what the sock cost (36 steps/s with it, 53 without). Fifteen iterations and thirty held the
sock in the same bites, 5 of 5 each."""


def rolling_resistance(sim, ref, torque: float) -> None:
    """Give a ball the drag of rolling on a desk mat: a friction torque [N m] on its three turning axes.

    MuJoCo's own rolling friction acts only on a six-dimensional contact, which is fixed when
    the model is built; a joint's friction loss can be set on a built model. Without either a
    ball rolls on as if on glass, off the end of the desk.
    """
    import newton
    from robosimstudio.core.tuning import ArrayWriter

    class Dofs(ArrayWriter):
        flags = int(newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES)

    body = int(ref.handle.body)
    joint = int(np.nonzero(sim.model.joint_child.numpy() == body)[0][0])
    first = int(sim.model.joint_qd_start.numpy()[joint])
    writer = Dofs(sim.model, sim.pipeline)
    writer.landed(writer._write("joint_friction", [first + 3, first + 4, first + 5], torque))


def look_through(studio, camera) -> None:
    """Open every browser through ``camera`` (a ``CameraSpec``): its place, aim and lens. Reset View returns there."""
    initial = studio.server.initial_camera
    initial.up = (0.0, 0.0, 1.0)  # the world's up, which the browser orbits about; the lens's own roll is the render's
    initial.position = tuple(camera.eye)
    initial.look_at = tuple(camera.look_at)
    initial.fov = math.radians(camera.fov_y)


@dataclass
class Home:
    """The twin with the duck and a ball on its desk."""

    sim: Any
    duck: Any  # the first duck
    ball: Any
    sock: Any  # the folded sock, or None
    ground: float  # the desk mat's top, which the duck, the ball and the sock stand on [m]
    in_bowl: Callable[[Any], bool]  # the ball's centre is inside the bowl
    mouth: Callable[[], np.ndarray]  # the middle of the bowl's mouth, in world [m]
    camera: Any  # the photo's own camera, as the twin solved it, or None
    ducks: list = None  # every duck, the first one first
    kinds: list = None  # each duck's routine settings (flock.Routine's kind, home, leash, lane), or None


def build(twin_dir: Path = TWIN, ducks: int = 1, *, with_sock: bool | None = None) -> Home:
    """Load the twin's ``scene.py``, put the duck (or ``ducks`` of them), a ball and a sock on its desk, and build it.

    The sock comes with one duck only (``with_sock`` None), unless asked for: it costs the step 40% (121
    control steps a second against 206, headless).
    """
    sys.path.insert(0, str(twin_dir))  # a twin's scene.py may import helpers from its own folder
    twin = runpy.run_path(str(twin_dir / "scene.py"), run_name="home_twin")
    scene = twin["scene"]
    # what the duck's policies were trained at -- 50 Hz control over a 5 ms MuJoCo step -- with the
    # twin's own contact settings. The sock is a soft body on VBD, led one way by MuJoCo: the beak, the
    # feet and the desk push it and feel nothing back, which is what lets a beak pinch it.
    with_sock = ducks <= 1 if with_sock is None else with_sock
    soft = {"backends": (("soft", SOCK_SOLVER),), "coupling": rss.coupler("one_way")} if with_sock else {"backends": ()}
    scene.physics = rss.ArrangementSpec(
        control_hz=50.0,
        substeps=4,
        backends=(("*", backend("mujoco", nconmax=8192, njmax=16384)), *soft.pop("backends")),
        collision=twin["PHYSICS"].collision,
        **soft,
    )
    top = float(twin.get("MAT_Z", 0.75))
    members = [(DUCK_AT, DUCK_YAW, "classic", "", 0.0)] if ducks <= 1 else list(FLOCK[: min(ducks, len(FLOCK))])

    def ground_at(xy) -> float:  # the notepad is 4 mm higher than the mat, and the bare desk behind the mat 3 mm lower
        return top + (0.004 if xy[0] > 0.07 and xy[1] < -0.03 else -0.003 if xy[1] > 0.0 else 0.0)

    flock, kinds = [], []
    for i, (at, yaw, colours, kind, leash) in enumerate(members):
        spec = DuckSpec(
            name="duck" if i == 0 else f"duck{i + 1}", at=at, yaw=yaw, ground=ground_at(at), colours=colours
        )
        flock.append(scene.add(spec))
        kinds.append({"kind": kind, "home": at, "leash": leash, "lane": yaw} if kind else None)
    duck = flock[0]
    ball = scene.add(
        rss.sphere(
            name="ball",
            radius=BALL_R,
            at=(*(ball_at := BALL_AT if len(flock) == 1 else FLOCK_BALL_AT), ground_at(ball_at) + BALL_R),
            mass=0.03,
            friction=1.0,
            color=(0.98, 0.78, 0.15),
        )
    )
    if with_sock:
        scene.add(sock(at=SOCK_AT, yaw=SOCK_YAW, ground=float(twin.get("DESK_Z", top))))  # see sock_lies_on_the_desk
    sim = scene.build(control_mode="joint_position")
    if "use_shell_segmentation" in twin:  # the twin's own way of putting the hollow bowl over the photo
        sim = twin["use_shell_segmentation"](sim)
    if with_sock:
        sock_lies_on_the_desk(sim, SOCK_THROUGH)
        sock_collides_soft_only(sim)
    rolling_resistance(sim, ball, ROLLING)
    bowl = sim["bowl"]
    layout = twin.get("LAYOUT", {})
    axis_local, depth = np.array([0.0, 0.0, 1.0]), 0.05
    if "bowl_mouth" in layout and "bowl_origin" in layout:
        depth = float(np.linalg.norm(np.subtract(layout["bowl_mouth"], layout["bowl_origin"])))

    def in_bowl(p) -> bool:
        """The ball's centre inside the bowl: behind the mouth and within its rim."""
        X = np.asarray(bowl.pose, dtype=np.float64)
        local = _rot(X[3:7]).T @ (np.asarray(p) - X[:3])
        along = float(local @ axis_local)
        return -0.005 < along < depth - 0.005 and float(np.linalg.norm(local - along * axis_local)) < 0.05

    def mouth() -> np.ndarray:
        """The middle of the bowl's mouth, in world [m]."""
        X = np.asarray(bowl.pose, dtype=np.float64)
        return X[:3] + _rot(X[3:7]) @ (depth * axis_local)

    return Home(
        sim, duck, ball, sim["sock"] if with_sock else None, top, in_bowl, mouth, twin.get("phone"), flock, kinds
    )


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    count = next((int(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("--ducks=")), 1)
    home = build(Path(args[0]).resolve() if args else TWIN, ducks=count)
    sim, phone = home.sim, home.camera

    def page(studio):
        from flock import Routine

        def others(me):
            return lambda: [d.trunk()[:2] for d in home.ducks if d is not me]

        plays = [
            Play(
                d,
                home.ball,
                in_bowl=home.in_bowl,
                goal=home.mouth,
                sock=home.sock if i == 0 else None,
                ground=home.ground,
                routine=Routine(d, others=others(d), seed=i, **home.kinds[i]) if home.kinds[i] else None,
            )
            for i, d in enumerate(home.ducks)
        ]
        chorus: list[list] = []  # quacks still to come: [control steps to wait, the duck's Play]

        def on_keys(held, pressed) -> None:
            # W A S D, Space and E are for the first duck; the others carry on with their own lives; Q is a chorus
            plays[0].keys(held, frozenset(k for k in pressed if k != "Q" or len(plays) == 1))
            if "Q" in pressed and len(plays) > 1:
                chorus.extend([8 * i, play] for i, play in enumerate(plays))

        legend = "the white duck" if len(plays) > 1 else "the duck"
        sock_line = "<br><b>E</b> pick up the sock, put it down" if home.sock is not None else ""
        keyboard(
            studio.gui,
            on_keys,
            legend=(
                '<div style="padding: 0 0.75em 0.75em; font-size: 0.85em; line-height: 1.6">'
                f"<b>Drag</b> a duck to pick it up<br><b>W A S D</b> walk and turn {legend}"
                f"<br><b>Space</b> kick the ball{sock_line}<br><b>Q</b> quack</div>"
            ),
        )
        resets = [sim.reset_count] if hasattr(sim, "reset_count") else [0]

        def each() -> None:
            if getattr(sim, "reset_count", 0) != resets[0]:
                resets[0] = sim.reset_count
                for play in plays:
                    play.reset()
                chorus.clear()
            for item in list(chorus):
                item[0] -= 1
                if item[0] <= 0:
                    item[1].duck.emote("quack")
                    chorus.remove(item)
            for play in plays:
                play.step()

        studio.each_step(each)
        # A click on any part of a duck takes it by the head, so the rest of it hangs below the hand: the
        # page takes hold of the body the click hit, and held by the trunk the head and neck sit on top of the
        # grip and turn it over (110 degrees off upright on a carry, against 10 by the head).
        # The handle is also levelled and squared to the desk: it holds the head the way it held it when clicked,
        # and the duck turns its head to watch the ball, so its arrows would point every which way.
        objects = getattr(studio, "objects", None)
        if objects is not None and hasattr(objects, "_attach"):
            attach = objects._attach
            head_of = {int(b): int(d.entity.head) for d in home.ducks for b in d.entity.bodies}

            def by_the_head(body) -> None:
                head = head_of.get(int(body))
                attach(head if head is not None else body)
                slot = objects.handles.find(head) if head is not None else None
                if slot is not None and slot.live:
                    R = _rot(slot.quat)
                    yaw = round(math.atan2(R[1, 0], R[0, 0]) / (math.pi / 2)) * (math.pi / 2)
                    slot.quat = np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
                    slot.sync()

            objects._attach = by_the_head
        if phone is not None:
            look_through(studio, phone)
            view = getattr(studio, "render_view", None)
            if view is not None:  # the World model view through the photo's camera, over the photo's own pixels
                view._locked_request = phone.name
                for handle in studio.gui._gui_input_handle_from_uuid.values():  # and the panel's Lock to says so
                    if getattr(handle, "label", "") == "Lock to":
                        handle.value = phone.name

    from robosimstudio_studio import StudioConfig

    rss.view(
        sim,
        page=page,
        panels=("inspector", "overlay", "object", "renderview"),
        render_every=1,
        # the page's 3D view path-traced through the photo's camera (it takes about a minute to come up)
        config=StudioConfig(world_model=True, render_view_scale=1.0),
    )


if __name__ == "__main__":
    main()
