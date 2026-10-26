# SPDX-License-Identifier: Apache-2.0
"""A flock at play: each duck in its own colours, each with its own way of passing the time.

A :class:`Routine` is what one duck does when nobody is steering it, one control step at a
time, from the same calls the keys make: walk and turn, sit down and get up, a forward roll,
a peck at the desk. Each duck keeps to a spot of its own. The sitter and the pecker stay on
it and only turn about between goes; the wanderer strolls to points round it; the roller
rolls off it and walks back.

A walk steers all the way to a point, rather than turning and then setting off straight:
the gait coasts 20-40 mm each time it stops, so a duck that keeps starting and stopping
creeps off wherever it happens to face.
"""

from __future__ import annotations

import math

import numpy as np

from duck import _rot

DESK = (-0.36, 0.36, -0.42, -0.02)
"""Where a duck may stroll to, as (x0, x1, y0, y1) in world [m]: the mat, as far as the camera sees it, and the desk in
front of it."""

KINDS = ("wander", "sit", "roll", "peck")
STROLL, TURN = 0.3, 1.2  # walking speed [m/s] and turning rate [rad/s] commanded


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Routine:
    """One duck's life when nobody steers it: pause, then a move, then pause again.

    ``kind`` is one of :data:`KINDS`. ``home`` is the spot it keeps to, within ``leash`` [m]; ``others()`` gives
    the other ducks' trunk positions, so a stroll does not walk into one of them. A roller rolls along ``lane``
    (a heading [rad]), where that much of the desk is clear.
    """

    def __init__(self, duck, kind: str, others, *, home, leash: float = 0.08, lane: float = 0.0, seed: int = 0):
        self.duck, self.kind, self.others = duck, kind, others
        self.home, self.leash, self.lane = np.asarray(home, dtype=np.float64), leash, lane
        self.rng = np.random.default_rng(seed)
        self.move, self.left = "pause", int(self.rng.uniform(0.5, 1.5) * 50)
        self.target, self.heading, self.then = None, 0.0, None

    # ----------------------------------------------------------------- reading
    def _pose(self):
        T = self.duck.trunk()
        R = _rot(T[3:7])
        return T[:2].copy(), math.atan2(R[1, 0], R[0, 0])

    def _clear(self, a, b, room: float = 0.13) -> bool:
        """Whether the straight walk from ``a`` to ``b`` stays on the desk and ``room`` [m] from the other ducks."""
        x0, x1, y0, y1 = DESK
        if not (x0 <= b[0] <= x1 and y0 <= b[1] <= y1):
            return False
        for u in np.linspace(0.0, 1.0, 6)[1:]:
            q = a + u * (b - a)
            if any(np.linalg.norm(q - o) < room for o in self.others()):
                return False
        return True

    # ----------------------------------------------------------------- acting
    def _start(self, move: str, seconds: float) -> None:
        self.move, self.left = move, max(1, int(seconds * 50))

    def _goto(self, target, then=None, seconds: float = 6.0) -> None:
        self.target, self.then = np.asarray(target, dtype=np.float64), then
        self._start("walking", seconds)

    def _face(self, heading: float, then=None) -> None:
        self.heading, self.then = heading, then
        self._start("turning", 3.0)

    def _next(self) -> None:
        """Pick the next move, by kind; a duck off its spot walks back to it first."""
        duck, rng = self.duck, self.rng
        p, yaw = self._pose()
        if np.linalg.norm(self.home - p) > self.leash:
            self._goto(self.home, then="lane" if self.kind == "roll" else None)
        elif self.kind == "sit" and rng.random() < 0.7:
            duck.walk(0, 0, 0)
            duck.sit(True)
            self._start("sitting", rng.uniform(2.5, 4.0))
        elif self.kind == "peck" and rng.random() < 0.7:
            duck.peck()
            self._start("pecking", 3.2)
        elif self.kind == "roll":
            ahead = p + 0.30 * np.array([math.cos(self.lane), math.sin(self.lane)])
            if np.linalg.norm(self.home - p) > 0.04:  # each roll from its spot, so that it lands in view
                self._goto(self.home, then="lane")
            elif abs(_wrap(self.lane - yaw)) > 0.2:
                self._face(self.lane)
            elif self._clear(p, ahead, room=0.11) and rng.random() < 0.85:  # a roll passes a duck by a hair
                duck.walk(0, 0, 0)
                # stand still, the head straight (Play looks ahead while it readies itself), before it rolls: a roll set
                # off within two seconds of the head turning to watch the ball goes short, backwards or sideways
                self._start("settle", 2.6)
            else:
                self._start("pause", 1.0)
        elif self.kind == "wander":
            for _ in range(8):  # a stroll to somewhere else on its patch
                a = rng.uniform(-math.pi, math.pi)
                to = self.home + rng.uniform(0.4, 1.0) * self.leash * np.array([math.cos(a), math.sin(a)])
                if np.linalg.norm(to - p) > 0.04 and self._clear(p, to):
                    self._goto(to)
                    break
            else:
                self._start("pause", 0.8)
        else:  # the sitter and the pecker look about between goes
            self._face(self.lane + rng.uniform(-0.8, 0.8))
        if rng.random() < 0.12:
            duck.emote("quack")

    def step(self) -> None:
        """Once per control step, while nobody is steering this duck."""
        duck = self.duck
        self.left -= 1
        if self.move == "walking":  # steer all the way there, turning on the spot first when it faces away
            p, yaw = self._pose()
            to = self.target - p
            dist = float(np.linalg.norm(to))
            if dist < 0.04 or self.left <= 0:
                duck.walk(0, 0, 0)
                if self.then == "lane":
                    self._face(self.lane)
                else:
                    self._start("pause", self.rng.uniform(0.6, 1.4))
                return
            bearing = _wrap(math.atan2(to[1], to[0]) - yaw)
            if abs(bearing) > (0.35 if dist < 0.12 else 0.6):  # close by, it turns tighter than the gait can steer
                duck.walk(0, 0, math.copysign(TURN, bearing))
            else:
                duck.walk(STROLL, 0.0, float(np.clip(2.5 * bearing, -1.0, 1.0)))
            return
        if self.move == "turning":
            _, yaw = self._pose()
            err = _wrap(self.heading - yaw)
            if abs(err) < 0.15 or self.left <= 0:
                duck.walk(0, 0, 0)
                self._start("pause", self.rng.uniform(0.4, 1.0))
                return
            duck.walk(0, 0, math.copysign(TURN if abs(err) > 0.4 else 0.9, err))  # the gait barely turns below ~0.8
            return
        if self.move == "settle" and self.left <= 0:
            duck.trick("roulade")
            self._start("rolling", 2.6)
            return
        if self.move == "sitting" and self.left <= 0:
            duck.sit(False)  # stand back up, then hand back to the gait
            self._start("standing", 1.6)
            return
        if self.move == "standing" and self.left <= 0:
            duck.walk(0.01, 0, 0)
            duck.walk(0, 0, 0)
            self._start("pause", 0.8)
            return
        if self.left <= 0 and duck.mode == "walk":
            self._next()
