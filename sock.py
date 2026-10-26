# SPDX-License-Identifier: Apache-2.0
"""A folded sock on the desk: a soft body the duck picks up with its beak.

A wool sock folded in half into a bundle, the way it comes out of a drawer: 95 x 30 x
25 mm, 25 g, a soft solid of tetrahedra. It is a RoboSimStudio soft body stepped by Newton's
VBD beside the duck's MuJoCo, coupled one way (``home.py``): the beak, the feet and the
desk push the sock, and nothing pushes back on them. The duck lifts it by contact and
friction alone; nothing pins it to the beak.

Why a folded bundle and not a cloth sock lying loose. At the bottom of its ground pick
the duck's beak points straight down, and the shell of its upper bill reaches 6.5 mm
past its lips, so the mouth closes no lower than 9 mm over the desk. Measured, a cloth
sock (a tube of VBD membrane, lying flat or dropped into a heap 20-30 mm tall) was caught
only at the tips of the beak in some 60 bites, over the sock's shape, where the beak came
down, the grip, and the pads' friction, and slid out as the duck stood up: the bill
flattens a hollow sock under it and the closing jaw presses the rest flat. A cloth fin
standing 25-30 mm high between the jaws did come up, which says the grip itself works.
A folded sock is a solid wad of knit at the beak's scale: the bill dents its top, and the
jaw squeezes the material behind the dent against the bill. Its flat top and upright back
matter: the same soft body as a rounded roll was caught only at the tips too.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import robosimstudio as rss
from robosimstudio.core.render.finish import Finish, register_texture
from robosimstudio.specs.asset import AssetSpec
from robosimstudio_materials.soft import EntitySoft, SoftMaterialSpec, SoftSpec

from assets import cache_dir

SIZE = (0.095, 0.030, 0.025)  # long, deep, tall [m]: a sock folded in half
"""Its length is what lets the duck's walk be coarse: bitten 25 or 35 mm off its middle, a
95 mm bundle came up every time (4 of 4); the beak is 80 mm wide."""
CELL = 0.005  # the lattice [m]: 840 nodes, 2850 tetrahedra
FOLD = 0.006  # the radius the two folded ends are rounded to [m]
VERSION = 2  # bump when the mesh changes: it is cached under this number

KNIT = SoftMaterialSpec(
    young=3.0e4,  # a soft bundle of wool knit: the bill dents it
    poisson=0.3,
    density=350.0,  # 25 g over its 71 cm^3
    stiffness_damping=5.0e-3,
    friction=1.0,  # knit against the beak's soft lips and the desk mat
    particle_radius=0.002,
)
CONTACT_KE, CONTACT_KD = 1.0e5, 1.0e-1  # its particles against the duck and the desk [N/m], [N s/m]
"""Stiff enough that the bill, pressing the sock flat, cannot crush it into what it lies on: at 1e4 and 3e4
one bite in seven pushed particles through the 3 mm desk mat and out under the desk. (It lies on the
desk itself now, under the mat: ``home.sock_lies_on_the_desk``.)"""
COLOR = (0.10, 0.63, 0.64)  # the turquoise of its knit, for views that draw no texture


def striped_knit(length_mm: float = 95.0, depth_mm: float = 30.0, px: int = 8) -> np.ndarray:
    """The sock's top as an ``(h, w, 3)`` image, its length along the width: a ribbed cream cuff, a coral band,
    turquoise with cream stripes, a coral toe; knit stitches and a little fuzz over all of it.

    Plain, the bundle read as an eraser (in red) or a sponge (in turquoise), and in navy it vanished into the
    duck's neck. The stripes run round the bundle's sides too: its UVs are its rest shape's two longest axes.
    """
    w, h = round(length_mm * px), round(depth_mm * px)
    U, V = np.meshgrid((np.arange(w) + 0.5) / px, (np.arange(h) + 0.5) / px)  # [mm]
    turquoise, cream, coral = (np.array(c) / 255.0 for c in ((26, 160, 164), (240, 233, 214), (236, 104, 84)))
    img = np.empty((h, w, 3))
    img[:] = turquoise
    for start in (30.0, 44.0, 58.0):
        img[(U >= start) & (U < start + 5.0)] = cream
    cuff = U < 16.0
    img[cuff] = cream
    img[(U >= 16.0) & (U < 20.0)] = coral
    img[U >= 80.0] = coral
    img[cuff] *= (1.0 - 0.13 * (0.5 + 0.5 * np.cos(np.pi * V[cuff])))[:, None]  # ribs along the leg, 2 mm apart
    s = (U / 1.2) % 1.0  # stockinette: V stitches pointing along the sock, 1.6 mm wales, 1.2 mm courses
    t = np.abs(((V / 1.6) % 1.0) - 0.5) * 2.0
    groove = np.clip(np.exp(-(((t - s) / 0.18) ** 2)) + 0.6 * np.exp(-((t / 0.12) ** 2)), 0.0, 1.0)
    img[~cuff] *= (1.0 - 0.10 * groove[~cuff])[:, None]
    img *= (1.0 + 0.035 * np.random.default_rng(7).standard_normal((h, w)))[..., None]
    return np.clip(img, 0.0, 1.0)


register_texture("microduck_sock_knit", striped_knit, version=1)
FINISH = Finish(roughness=0.95, texture="microduck_sock_knit", uv_scale=(1.0 / SIZE[0], 1.0 / SIZE[1]))
"""Matte wool, the knit drawn once over its top: UVs are in metres, so the scale fits one image to the sock."""


def folded_sock(size=SIZE, cell=CELL, fold=FOLD) -> tuple[np.ndarray, np.ndarray]:
    """``(vertices (n, 3), tets (m, 4))``: a box of tetrahedra on z=0 whose two ends are rounded like folds."""
    length, depth, height = size
    nx, ny, nz = (max(2, round(s / cell)) for s in size)
    xs = np.linspace(-length / 2, length / 2, nx + 1)
    ys = np.linspace(-depth / 2, depth / 2, ny + 1)
    zs = np.linspace(0.0, 1.0, nz + 1)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    # the top of each end follows a quarter circle down to the fold: the top's middle stays flat
    over = np.clip(np.abs(X) - (length / 2 - fold), 0.0, fold)
    top = (height - fold) + np.sqrt(fold * fold - over * over)
    vertices = np.stack([X, Y, Z * top], axis=-1).reshape(-1, 3)
    index = np.arange(vertices.shape[0]).reshape(nx + 1, ny + 1, nz + 1)
    # five tetrahedra per cell, the split alternating so neighbouring cells share their faces
    even = ((0, 1, 2, 4), (3, 2, 1, 7), (5, 1, 4, 7), (6, 4, 2, 7), (1, 2, 4, 7))
    odd = ((1, 0, 3, 5), (2, 3, 0, 6), (4, 0, 5, 6), (7, 3, 6, 5), (0, 3, 5, 6))
    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = [index[i + a, j + b, k + d] for d in (0, 1) for b in (0, 1) for a in (0, 1)]
                tets.extend([c[n] for n in t] for t in (even if (i + j + k) % 2 == 0 else odd))
    tets = np.asarray(tets, dtype=np.int64)
    a, b, c, d = (vertices[tets[:, n]] for n in range(4))
    flip = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) < 0.0
    tets[flip, 1], tets[flip, 2] = tets[flip, 2].copy(), tets[flip, 1].copy()
    return vertices, tets


def mesh_path() -> Path:
    """The folded sock's tetrahedra as an ``.npz``, written once into the cache."""
    path = cache_dir() / f"folded_sock_v{VERSION}.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        vertices, tets = folded_sock()
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.part.npz")  # two pages starting at once each write their own
        np.savez(tmp, vertices=vertices, tets=tets)
        tmp.replace(path)
    return path


@rss.register_entity("sock")
class Sock(EntitySoft):
    """A soft body that states its own contact with shapes, which VBD reads off the model.

    A sheet writes the model's particle-contact stiffness and friction from its material;
    a soft body built for SuperDex does not, so on VBD it would get Newton's defaults
    (1e3 N/m, friction 0.5).
    """

    def apply_model_materials(self, model) -> None:
        model.soft_contact_ke = CONTACT_KE
        model.soft_contact_kd = CONTACT_KD
        model.soft_contact_mu = float(self.spec.material.friction)


def sock(*, at=(0.0, 0.0), yaw: float = 0.0, ground: float = 0.0, name: str = "sock", color=COLOR) -> SoftSpec:
    """A folded sock lying at ``at`` (x, y in world [m]) on a surface ``ground`` high, its long side across ``yaw``.

    ``yaw`` 0 lays its 70 mm along world x. The duck bites it from either long side.
    """
    half = 0.5 * float(yaw)
    return SoftSpec(
        entity="sock",
        name=name,
        asset=AssetSpec(path=str(mesh_path())),
        pose=(float(at[0]), float(at[1]), float(ground) + 0.0005),
        quat=(0.0, 0.0, math.sin(half), math.cos(half)),
        material=KNIT,
        # Built at rest, so two frames settle it. A long settle would step the duck before its brain is wired.
        settle_frames=2,
        settle_substeps=4,
        color=tuple(color),
        finish=FINISH,
    )


def lying(ref) -> tuple[np.ndarray, np.ndarray, float]:
    """Where a sock lies: ``(centre xyz, long-side unit xy, top height)``, from its particles, in world [m]."""
    q = np.asarray(ref.particle_q, dtype=np.float64)
    centre = q.mean(axis=0)
    d = q[:, :2] - centre[:2]
    w, v = np.linalg.eigh(d.T @ d)
    along = v[:, int(np.argmax(w))]
    return centre, along / np.linalg.norm(along), float(q[:, 2].max())
