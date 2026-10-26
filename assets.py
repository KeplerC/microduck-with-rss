# SPDX-License-Identifier: Apache-2.0
"""Fetch the Microduck: its MJCF, meshes and walking policies, pinned, into a cache.

Nothing of the robot is vendored here. The MJCF comes from Pollen Robotics'
`microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_ at a pinned
commit, and the policies from the official set
`pollen-robotics/microduck-policies <https://huggingface.co/pollen-robotics/microduck-policies>`_
at a pinned tag. The code there is Apache-2.0. The meshes are CC BY-NC-SA 4.0,
which is why they are downloaded on first use rather than shipped.

The official model has fourteen servos. The real robot has a fifteenth, the beak
(``mouth``: -5 deg shut, +30 deg open), which no policy drives and the MJCF does not
have. :func:`patched_mjcf` puts it back: the lower beak becomes a body of its own on
a hinge through the mouth servo's horn, 3 mm up and 18 mm back in the head frame,
and both halves of the beak get a flat box collider, so the duck can hold something
in its mouth.
"""

from __future__ import annotations

import math
import os
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

RL_REPO = "pollen-robotics/microduck_rl"
RL_COMMIT = "b02fdea4902c2b93b7d34a83a6f05da4ececb7d3"  # develop, 2026-10-12
MODEL_DIR = "src/mjlab_microduck/robot/microduck"
ROBOT_XML = "robot_groundcontact.xml"  # the collision set the standing policies train on

POLICY_REPO = "pollen-robotics/microduck-policies"
POLICY_TAG = "v8"
POLICIES = (
    "velstand.onnx",
    "alpha_ground_pick.onnx",
    "alpha_sitstand.onnx",
    "ball_kick_left.onnx",
    "ball_kick_right.onnx",
    "roulade.onnx",
    "manifest.json",
)

MOUTH_RANGE = (-0.0873, 0.5236)  # -5 deg shut, +30 deg open, the runtime's mouth_target range
_HINGE = (0.003, 0.0, -0.018)  # the mouth servo's horn, in the head frame [m]
_JAW_MASS = 0.015  # [kg], taken out of the head so the duck weighs what it did
_PAD = {"friction": "1.5 0.05 0.004", "condim": "6"}  # the soft TPU lips: they grip, and a ball does not roll out
_GROOVE = math.radians(12.0)  # the lower beak's two halves tilt up to their edges, so a held ball centres
PATCH = 4  # bump when the patch changes: the patched file is cached under this number


def cache_dir() -> Path:
    root = os.environ.get("MICRODUCK_CACHE") or Path.home() / ".cache" / "robosimstudio" / "microduck"
    return Path(root) / RL_COMMIT[:12]


def _fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        f.write(r.read())
    tmp.replace(dest)


def fetch() -> Path:
    """Download the model and the policies once; return the cache folder holding them."""
    root = cache_dir()
    raw = f"https://raw.githubusercontent.com/{RL_REPO}/{RL_COMMIT}/{MODEL_DIR}"
    _fetch(f"{raw}/{ROBOT_XML}", root / ROBOT_XML)
    meshes = sorted({m.get("file") for m in ET.parse(root / ROBOT_XML).getroot().iter("mesh")})
    missing = [m for m in meshes if not (root / "assets" / m).exists()]
    if missing:
        print(f"microduck: downloading {len(missing)} meshes from {RL_REPO}@{RL_COMMIT[:7]} (CC BY-NC-SA 4.0)")
    for m in missing:
        _fetch(f"{raw}/assets/{m}", root / "assets" / m)
    for p in POLICIES:
        _fetch(f"https://huggingface.co/{POLICY_REPO}/resolve/{POLICY_TAG}/{p}", root / "policies" / p)
    return root


def _vec(text: str | None, n: int = 3) -> list[float]:
    return [float(v) for v in text.split()] if text else [0.0] * n


def _fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in v)


COLOURS = {
    # scheme: (shells, accents, soles and eye ring, soft mouth), RGB in 0..1
    "classic": None,  # Pollen's own: off-white shells, orange
    "charcoal": ((0.20, 0.21, 0.23), (0.98, 0.78, 0.10), (0.55, 0.84, 0.74), (0.95, 0.88, 0.55)),
    "lavender": ((0.76, 0.70, 0.95), (0.98, 0.78, 0.10), (0.62, 0.56, 0.88), (0.95, 0.88, 0.55)),
    "mint": ((0.60, 0.87, 0.78), (0.97, 0.44, 0.40), (0.98, 0.84, 0.30), (0.98, 0.78, 0.76)),
}
"""Colour schemes for a flock. Each recolours four groups of the robot's printed parts; the servos, bearings and
boards keep theirs."""

_GROUPS = (
    ("right_shell", "left_shell", "upper_leg_left", "upper_leg_right", "top_head_shell"),
    ("foot_left", "foot_right", "ankle_left", "ankle_right", "jaw", "bottom_head_shell"),
    ("sole_left", "sole_right", "noenoeil"),
    ("jaw_soft", "soft_mouth_top"),
)


def patched_mjcf(colours: str = "classic") -> Path:
    """The robot with a beak that opens, in one of :data:`COLOURS`: written beside the original, returned as a path."""
    base = _patched()
    scheme = COLOURS[colours]
    if scheme is None:
        return base
    out = base.with_name(f"{base.stem}_{colours}.xml")
    if out.exists():
        return out
    tree = ET.parse(base)
    wanted = {f"{part}_material": rgb for parts, rgb in zip(_GROUPS, scheme, strict=True) for part in parts}
    for m in tree.getroot().iter("material"):
        if m.get("name") in wanted:
            m.set("rgba", _fmt((*wanted[m.get("name")], 1.0)))
    tree.write(out, encoding="unicode")
    return out


def _patched() -> Path:
    root = fetch()
    out = root / f"microduck_beak_v{PATCH}.xml"
    if out.exists():
        return out
    tree = ET.parse(root / ROBOT_XML)
    mj = tree.getroot()
    head = next(b for b in mj.iter("body") if b.get("name") == "jaw_soft")

    # The lower beak and its soft lining move to a body of their own on the mouth hinge.
    jaw = ET.Element("body", name="jaw", pos=_fmt(_HINGE))
    ET.SubElement(
        jaw, "joint", name="mouth", type="hinge", axis="0 1 0", range=_fmt(MOUTH_RANGE), **{"class": "chosen_actuator"}
    )
    ET.SubElement(jaw, "inertial", pos="-0.018 0 -0.027", mass=str(_JAW_MASS), diaginertia="4e-6 6e-6 3e-6")
    for g in list(head.findall("geom")):
        if g.get("mesh") in ("jaw", "jaw_soft"):
            head.remove(g)
            if g.get("class") == "visual":
                pos = [a - b for a, b in zip(_vec(g.get("pos")), _HINGE, strict=True)]
                g.set("pos", _fmt(pos))
                jaw.append(g)
        elif g.get("mesh") == "bottom_head_shell" and g.get("class") == "collision":
            head.remove(g)  # its hull would fill the mouth; the box below is the upper beak's underside
    # Flat plates, so whatever is held sits between two faces rather than two hulls.
    ET.SubElement(
        head,
        "geom",
        name="upper_beak",
        type="box",
        pos="-0.0085 0 -0.050",
        size="0.004 0.040 0.030",
        **_PAD,
        **{"class": "collision"},
    )
    for side in (1, -1):
        c, s = math.cos(_GROOVE / 2), math.sin(_GROOVE / 2)
        ET.SubElement(
            jaw, "geom", name=f"lower_beak_{'l' if side > 0 else 'r'}", type="box",
            pos=f"-0.0181 {side * 0.015:.4f} -0.035", size="0.0024 0.0155 0.025",
            quat=_fmt((c, 0.0, 0.0, -side * s)), **_PAD, **{"class": "collision"},
        )  # fmt: skip
    # A lip along the lower beak's front edge, 3 mm proud of it: what a ball is squeezed toward stops there.
    ET.SubElement(
        jaw,
        "geom",
        name="beak_lip",
        type="box",
        pos="-0.0142 0 -0.0585",
        size="0.0015 0.028 0.0015",
        **_PAD,
        **{"class": "collision"},
    )
    head.append(jaw)
    inertial = head.find("inertial")
    inertial.set("mass", f"{float(inertial.get('mass')) - _JAW_MASS:.6g}")

    # Its servo, in the daemon's joint order: after head_roll, before the right leg.
    actuators = mj.find("actuator")
    kids = list(actuators)
    at = next(i for i, a in enumerate(kids) if a.get("joint") == "head_roll") + 1
    actuators.insert(at, ET.Element("position", name="mouth", joint="mouth", **{"class": "chosen_actuator"}))

    mj.find("compiler").set("meshdir", str(root / "assets"))
    tree.write(out, encoding="unicode")
    return out


if __name__ == "__main__":
    print(patched_mjcf())
