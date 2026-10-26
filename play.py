# SPDX-License-Identifier: Apache-2.0
"""Playing with the duck: walk it from the keyboard, kick a ball where it faces, and its little moods.

:class:`Play` runs once per control step on the page. The held keys walk and
turn it; Space lines it up on the ball and kicks with the foot on the ball's
side, toward wherever it was facing when Space went down (or into the bowl, when
the bowl is roughly that way); E walks it up to the sock and picks it up in its
beak, or puts down what it holds; Q quacks. When it stands idle it watches the
ball; when the ball lands in the bowl it cheers, and when a kick stops short it
hangs its head.

The kick is Pollen's own one-shot policy, and it does not see the ball: it
swings through one spot. So the line-up is what aims it. Measured with a 45 mm
ball, the kick sends it within a few degrees of straight ahead when the ball
sits 55-75 mm ahead of the trunk and 26-34 mm out to the kicking foot, and its
direction moves about 2 degrees per millimetre sideways. The gait cannot take a
step shorter than about 30 mm, and its turns cannot be made small to the right,
so the line-up walks in along the aim line (pure pursuit onto a line through
that spot) and stops; stopped well off the line, it backs off and walks in once
more. Measured on the home desk, 5 kicks of 12 from varied run-ups ended in the
bowl.

The bite (:meth:`duck.DuckRef.bite`) does not see the sock either: it comes down
where it comes down, and lifts the folded sock when the sock's middle lies 50-86
mm ahead of the trunk, within 30 mm of its centreline and 20 degrees of square.
So E walks in square to the sock's long side, the same way the kick lines up,
stops early (the gait coasts 20-40 mm after a stop), then turns square and steps
back or in until the sock lies 50-85 mm ahead with its near side clear of the
toes, and only then bites. A foot on the sock would pin it to the desk while the
beak pulls. When the sock did not come up it lets go and tries again, three bites
at most.
"""

from __future__ import annotations

import math

import numpy as np

from duck import BITE_REACH, _rot

KICK_AT = (0.060, 0.030)  # where the ball should sit for a straight kick: ahead of the trunk, out to the foot [m]
WALK, TURN = 0.35, 1.2  # what W and A/D command [m/s], [rad/s]


def _yaw(T) -> float:
    R = _rot(T[3:7])
    return math.atan2(R[1, 0], R[0, 0])


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Kick:
    """Line the duck up on the ball and kick it toward ``aim`` (a world heading, rad); ``step()`` per control step."""

    def __init__(self, duck, ball, aim: float, *, log=None) -> None:
        self.duck, self.ball, self.aim = duck, ball, aim
        b = self.ball.position
        T = duck.trunk()
        c, s = math.cos(aim), math.sin(aim)
        left = -s * (b[0] - T[0]) + c * (b[1] - T[1])
        self.side = "left" if left >= 0 else "right"
        self.phase, self.t, self.fixes, self.tries = "start", 0, 0, 0
        self.log = log or (lambda *_: None)
        self.done = False

    def _ball_in_trunk(self):
        T = self.duck.trunk()
        R = _rot(T[3:7])
        fwd = R[:2, 0] / np.linalg.norm(R[:2, 0])
        lft = np.array([-fwd[1], fwd[0]])
        d = self.ball.position[:2] - T[:2]
        return float(fwd @ d), float(lft @ d)

    def _target_y(self) -> float:
        return KICK_AT[1] if self.side == "left" else -KICK_AT[1]

    def step(self) -> None:
        if self.done:
            return
        duck, T = self.duck, self.duck.trunk()
        self.t += 1
        c, s = math.cos(self.aim), math.sin(self.aim)
        fwd, lft = np.array([c, s]), np.array([-s, c])
        b = self.ball.position[:2]
        stance = b - KICK_AT[0] * fwd - self._target_y() * lft  # where the trunk stands for the kick
        if self.phase == "start":  # too close to line up from here: back off first
            d = T[:2] - stance
            self.phase = "back" if float(fwd @ d) > -0.07 or abs(float(lft @ d)) > 0.5 * abs(float(fwd @ d)) else "walk"
            self.t = 0
        if self.phase == "back":
            d = T[:2] - stance
            if float(fwd @ d) < -0.16 or self.t > 150:
                duck.walk(0, 0, 0)
                self.phase, self.t = "walk", 0
            else:
                duck.walk(-0.3, 0, 0)
            return
        if self.phase == "walk":
            d = T[:2] - stance
            along = float(fwd @ d)  # along < 0: behind the stance, as it should be
            if along > -0.002 or self.t > 600:
                duck.walk(0, 0, 0)
                self.phase, self.t = "settle", 0
                return
            # pure pursuit onto the aim line through the stance, 60 mm ahead of where the trunk is on it
            look = stance + max(along, -0.25) * fwd + 0.06 * fwd
            if along < -0.25:  # far behind: head for the line's start first
                look = stance - 0.2 * fwd
            to = look - T[:2]
            bearing = _wrap(math.atan2(to[1], to[0]) - _yaw(T))
            if abs(bearing) > 0.7:
                duck.walk(0, 0, math.copysign(TURN, bearing))
            else:
                duck.walk(WALK if along < -0.06 else 0.3, 0, float(np.clip(2.5 * bearing, -1.0, 1.0)))
        elif self.phase == "settle":
            if self.t >= 15:
                self.phase, self.t = "check", 0
        elif self.phase == "check":
            bx, by = self._ball_in_trunk()
            self.side = "left" if by >= 0 else "right"  # the foot on the ball's side, as it stands now
            err_y = by - self._target_y()
            off = math.degrees(_wrap(_yaw(T) - self.aim))
            self.log(
                f"check {self.fixes}: ball {1000 * bx:.0f} ahead, {1000 * by:+.0f} left"
                f" (want {1000 * self._target_y():+.0f}), heading {off:+.0f} deg off aim"
            )
            if self.tries < 1 and (abs(err_y) > 0.015 or bx < 0.045):  # stopped off the line: back off, walk in again
                self.tries += 1
                self.phase, self.t = "back", 0
            elif self.fixes < 1 and bx > 0.078:  # out of the foot's reach: one short step in
                self.fixes += 1
                duck.walk(0.3, 0, 0)
                self.phase, self.t, self.pulse = "pulse", 0, 12
            elif self.fixes < 2 and 0.006 < abs(err_y) < 0.025 and 0.045 < bx < 0.085:
                # off across the foot: one short turn moves the ball across it by about the trunk-to-ball distance
                # per radian. Turning right moves it out to the left; the gait's shortest right turn is about
                # 11 degrees (13 mm here), its shortest left about 5 (6 mm).
                self.fixes += 1
                if err_y < 0:
                    duck.walk(0, 0, -0.6)
                    self.phase, self.t, self.pulse = "pulse", 0, 9
                else:
                    duck.walk(0, 0, 1.0)
                    self.phase, self.t, self.pulse = "pulse", 0, 10
            else:
                duck.trick(f"kick_{self.side}")
                self.phase, self.t = "kick", 0
        elif self.phase == "pulse":
            if self.t >= self.pulse:
                duck.walk(0, 0, 0)
                self.phase, self.t = "settle", 0
        elif self.phase == "kick":
            if duck.mode == "walk" and self.t > 10:
                self.done = True


class Fetch:
    """Walk up to the sock square to its long side and bite it; ``step()`` per control step.

    ``sock`` is the sock's ref and ``ground`` the height it lies on [m]. ``done`` when the duck
    stands holding it (``held``), or after three bites that did not lift it, or when it cannot
    get into a place to bite from. It never bites from outside that place: a bite that misses
    crushes the sock under the bill, and it is then harder to bite.
    """

    WINDOW = (0.050, 0.085)  # where the sock's middle may lie ahead of the trunk for a bite [m]
    CLEAR = 0.038  # and its near side at least this far ahead: the toes reach 35 mm, and a foot on it pins it [m]
    SIDE = 0.030  # and how far off the trunk's centreline [m]
    SQUARE = math.radians(15.0)  # and how far off square the duck may face it
    TOES = 0.072  # the walk is stopped before the sock's near side comes this close to the trunk [m]: it coasts
    # 20-40 mm after a stop, and its toes reach 35 mm ahead; a foot on the sock pins it to the desk

    def __init__(self, duck, sock, *, ground: float, log=None) -> None:
        self.duck, self.sock, self.ground = duck, sock, ground
        self.log = log or (lambda *_: None)
        self.phase, self.t, self.age = "start", 0, 0
        self.tries = self.turns = self.bites = self.nudges = 0
        self.stop_at = -0.020  # how far short of the stance the walk is told to stop
        self.done = self.held = False
        self._aim()

    def _aim(self) -> None:
        from sock import lying

        centre, along, _ = lying(self.sock)
        normal = np.array([-along[1], along[0]])
        T = self.duck.trunk()
        if normal @ (centre[:2] - T[:2]) < 0.0:  # from whichever long side the duck is on
            normal = -normal
        self.u = normal  # the walk in: toward the sock, square to its long side

    def _frame(self):
        T = self.duck.trunk()
        R = _rot(T[3:7])
        fwd = R[:2, 0] / np.linalg.norm(R[:2, 0])
        return T, fwd, np.array([-fwd[1], fwd[0]])

    def _sock_in_trunk(self):
        """The sock's middle ahead of the trunk and to its left [m], and how far the duck is off square [rad]."""
        from sock import lying

        T, fwd, lft = self._frame()
        d = lying(self.sock)[0][:2] - T[:2]
        return float(fwd @ d), float(lft @ d), _wrap(_yaw(T) - math.atan2(self.u[1], self.u[0]))

    def _near(self) -> float:
        """How far ahead of the trunk the nearest part of the sock in front of the feet is [m]."""
        T, fwd, lft = self._frame()
        d = np.asarray(self.sock.particle_q)[:, :2] - T[:2]
        ahead, side = d @ fwd, d @ lft
        front = ahead[(np.abs(side) < 0.05) & (ahead > 0.0)]
        return float(front.min()) if front.size else 1.0

    def lifted(self) -> bool:
        """The whole sock off the desk: its lowest point 20 mm up."""
        return float(np.asarray(self.sock.particle_q)[:, 2].min()) > self.ground + 0.02

    def _again(self, why: str) -> None:
        self.log(why)
        self.tries += 1
        self.phase, self.t = "back", 0

    def step(self) -> None:
        if self.done:
            return
        from sock import lying

        duck, T = self.duck, self.duck.trunk()
        self.t += 1
        self.age += 1
        if self.age > 30 * 50 and self.phase not in ("bite", "verify"):  # rather than shuffle about on it
            self.log("gave up: too long")
            duck.walk(0, 0, 0)
            self.done = True
            return
        u, side = self.u, np.array([-self.u[1], self.u[0]])
        stance = lying(self.sock)[0][:2] - BITE_REACH * u  # where the trunk stands for a bite
        d = T[:2] - stance
        along = float(u @ d)  # < 0: short of the stance, as it should be
        if self.phase == "start":  # too close to walk in from here: back off first
            self.phase = "back" if along > -0.06 or abs(float(side @ d)) > max(0.04, -along) else "walk"
            self.t = 0
        if self.phase == "back":
            if along < -0.15 or self.t > 75:
                duck.walk(0, 0, 0)
                self.phase, self.t = "walk", 0
            else:
                duck.walk(-0.4, 0, 0)  # -0.3 is inside the gait's dead band
            return
        if self.phase == "walk":
            if along > self.stop_at or self._near() < self.TOES or self.t > 600:
                duck.walk(0, 0, 0)
                self.phase, self.t = "settle", 0
                return
            # pure pursuit onto the line through the stance, square to the sock, 80 mm ahead of the trunk
            look = stance + max(along, -0.20) * u + 0.08 * u
            if along < -0.20:
                look = stance - 0.15 * u
            to = look - T[:2]
            bearing = _wrap(math.atan2(to[1], to[0]) - _yaw(T))
            if abs(bearing) > 0.7:
                duck.walk(0, 0, math.copysign(TURN, bearing))
            else:
                duck.walk(WALK if along < -0.08 else 0.3, 0, float(np.clip(2.5 * bearing, -1.0, 1.0)))
        elif self.phase == "settle":  # a bite from a duck still swaying from its walk comes down off where it was aimed
            if self.t >= 40:
                self.phase, self.t = "check", 0
        elif self.phase in ("turn", "nudge"):
            if self.t >= self.turn_steps:
                duck.walk(0, 0, 0)
                self.phase, self.t = "settle", 0
        elif self.phase == "check":
            ahead, left, off = self._sock_in_trunk()
            lo, hi = self.WINDOW
            self.log(
                f"check: sock {1000 * ahead:.0f} ahead, {1000 * left:+.0f} left,"
                f" {math.degrees(off):+.0f} deg off square"
            )
            if abs(off) > self.SQUARE and self.turns < 2:  # askew: turn square on the spot
                self.turns += 1
                self.turn_steps = max(3, round(abs(off) / TURN * 50))
                duck.walk(0, 0, -math.copysign(TURN, off))
                self.phase, self.t = "turn", 0
                return
            # Nudges the gait makes from a standstill, measured three times each: back 0.4 for 15 steps,
            # 16-20 mm; forward 0.3 for 10 steps, 6-9 mm. Its sideways steps were not repeatable.
            too_close = ahead < lo or self._near() < self.CLEAR
            if too_close and self.nudges < 4:
                self.nudges += 1
                duck.walk(-0.4, 0, 0)
                self.phase, self.t, self.turn_steps = "nudge", 0, 15
                return
            if ahead > hi and self.nudges < 6:
                self.nudges += 1
                duck.walk(0.3, 0, 0)
                self.phase, self.t, self.turn_steps = "nudge", 0, 10
                return
            if too_close or ahead > hi or abs(left) > self.SIDE or abs(off) > self.SQUARE:
                if self.tries < 2:  # still not in place: back off and walk in again
                    self.nudges = self.turns = 0
                    self._again("not in place: back off and walk in again")
                else:
                    self.log("gave up: cannot get in place")
                    self.done = True
                return
            duck.bite()
            self.bites += 1
            self.phase, self.t = "bite", 0
        elif self.phase == "bite":
            if self.t > 10 and duck.mode == "walk" and not duck.biting:
                self.phase, self.t = "verify", 0
        elif self.phase == "verify":
            if self.t < 25:  # let it hang still
                return
            if self.lifted():
                self.log(f"holding the sock after {self.bites} bite(s)")
                self.done = self.held = True
            elif self.bites < 3:
                self.log("missed it: let go and try again")
                duck.mouth(0.0)
                self.tries = self.turns = self.nudges = 0
                self._aim()
                self.phase, self.t = "start", 0
            else:
                self.log("gave up")
                duck.mouth(0.0)
                self.done = True


class Play:
    """The duck at play, run once per control step from the page.

    ``keys(held, pressed)`` takes the keyboard (:func:`keys.keyboard`). ``in_bowl(ball_xyz)
    -> bool`` says when the ball has landed in the bowl, and ``goal() -> xy`` where its mouth
    is: a kick toward the duck's facing that would pass within 45 degrees of it aims at it.
    ``sock`` is the sock's ref, lying on a surface ``ground`` high [m], for E to fetch.
    """

    def __init__(self, duck, ball, *, in_bowl=None, goal=None, sock=None, ground: float = 0.0, routine=None) -> None:
        self.duck, self.ball, self.in_bowl, self.goal = duck, ball, in_bowl, goal
        self.sock, self.ground = sock, ground
        self.routine = routine  # what it does on its own while no key steers it (flock.Routine), or None
        self.held: frozenset[str] = frozenset()
        self.kick: Kick | None = None
        self.fetch: Fetch | None = None
        self.look = np.zeros(2)  # head yaw, head pitch, eased
        self.scored = False
        self.stand_z = None
        self.lifted = 0
        self.watching = 0  # control steps since a kick ended, while the ball still rolls
        self.last_ball = None
        self.cheer = False  # the ball went in while it was busy kicking: cheer once it stands again

    def keys(self, held: frozenset[str], pressed: frozenset[str]) -> None:
        self.held = held
        if "_" in pressed and self.kick is None and self.fetch is None and self.duck.mode == "walk":
            aim = _yaw(self.duck.trunk())
            if self.goal is not None:
                to = np.asarray(self.goal())[:2] - self.ball.position[:2]
                at = math.atan2(to[1], to[0])
                if abs(_wrap(at - aim)) < math.radians(45):
                    aim = at
            self.kick = Kick(self.duck, self.ball, aim)
        if "E" in pressed and self.kick is None:
            if self.duck.holding:  # put it down
                self.fetch = None
                self.duck.put_down()
            elif self.sock is not None and self.fetch is None and self.duck.mode == "walk":
                self.fetch = Fetch(self.duck, self.sock, ground=self.ground)
        if "Q" in pressed:
            self.duck.emote("quack")
        if held & set("WASD"):  # walking takes the duck back
            self.kick = self.fetch = None

    def reset(self) -> None:
        self.held, self.kick, self.scored, self.lifted, self.watching = frozenset(), None, False, 0, 0
        self.fetch = None
        self.cheer = False
        self.look[:] = 0.0

    def step(self) -> None:
        duck = self.duck
        steered = bool(self.held & set("WASD")) or self.kick is not None or self.fetch is not None
        if duck.mode != "walk" and self.kick is None and self.fetch is None:
            # sitting, rolling, pecking: the routine times its own moves, with the head straight -- the head's
            # offsets go into every policy, and a peck with the head turned to the ball runs out 25 cm when it ends
            self.look[:] = 0.0
            if not duck.emoting:
                duck.look()
            if self.routine is not None and not steered:
                self.routine.step()
            return
        T = duck.trunk()
        if self.stand_z is None:
            self.stand_z = T[2]
        # picked up by the mouse: feet off the ground, and it complains about it
        up = T[2] - self.stand_z
        self.lifted = self.lifted + 1 if up > 0.04 else 0
        if self.lifted and self.lifted % 40 == 1:
            duck.emote("squawk")
        if self.fetch is not None:
            self.fetch.step()
            if self.fetch.done:
                self.fetch = None
        elif self.kick is not None:
            self.kick.step()
            if self.kick.done:
                self.kick, self.watching = None, 1
        elif self.routine is not None and not steered:
            self.routine.step()
        elif duck.mode == "walk":
            fwd = (0.3 if "W" in self.held else 0.0) - (0.25 if "S" in self.held else 0.0)
            turn = (TURN if "A" in self.held else 0.0) - (TURN if "D" in self.held else 0.0)
            duck.walk(WALK if fwd > 0 else fwd, 0.0, turn)
        # it watches the ball while it stands, and looks straight ahead when it goes for the sock or holds it: a bite
        # comes down where the head points. It also looks ahead while it readies itself for a roll: a roll set off
        # within two seconds of the head turning goes short, backwards or sideways, where one from a head held
        # straight goes 0.31-0.38 m forward
        readying = self.routine is not None and self.routine.move == "settle"
        busy = self.fetch is not None or duck.holding or duck.biting or duck.releasing or readying
        if duck.biting or duck.releasing:
            self.look[:] = 0.0
        idle = not (self.held & set("WASD")) and self.kick is None and not busy and not duck.emoting
        if self.ball is not None and idle:
            R = _rot(T[3:7])
            d = self.ball.position - T[:3]
            ahead, left = R[:2, 0] @ d[:2], R[:2, 1] @ d[:2]
            dist = math.hypot(ahead, left)
            want = np.array(
                [np.clip(math.atan2(left, ahead), -1.0, 1.0), np.clip(1.2 * (math.atan2(0.1, dist) - 0.25), -0.3, 0.5)]
            )
        else:
            want = np.zeros(2)
        self.look += 0.08 * (want - self.look)
        if not duck.emoting:
            duck.look(head_yaw=float(self.look[0]), head_pitch=float(self.look[1]))
        # and is pleased when the ball ends up in the bowl, and sorry when a kick of its stops short of it
        if self.in_bowl is not None and self.ball is not None:
            inside = self.in_bowl(self.ball.position)
            if inside and not self.scored:
                self.cheer, self.watching = True, 0
            self.scored = inside
            if self.cheer and duck.mode == "walk" and self.kick is None:
                duck.emote("happy")
                self.cheer = False
            here = self.ball.position.copy()
            rolling = self.last_ball is not None and np.linalg.norm(here - self.last_ball) * 50.0 > 0.02
            self.last_ball = here
            if self.watching:
                self.watching += 1
                if not inside and duck.mode == "walk" and (self.watching > 300 or (self.watching > 40 and not rolling)):
                    duck.emote("sad")
                    self.watching = 0
