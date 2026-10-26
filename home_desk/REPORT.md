# Home desk simulation set

Delivered in `examples/live/twins/home_desk/`. The live page is
**http://localhost:8231**, PID 1356629 at delivery, with
`CUDA_VISIBLE_DEVICES=6` verified in its process environment. Newton sees that
single card as `cuda:0`. No other GPU was used. No robot, ball or routine is
in `scene.py`. Nothing was committed; no library or test files were changed.
Other work was present in `examples/10_microduck/` and was left alone.

The set is usable for the showcase, with a contact-tested bowl cavity. It is
**not a metrically verified replica**: the photograph supplies no ruler,
measured mass, material test or surveyed camera landmarks. The assumptions and
visible mismatches below are part of the delivery.

## Calibration and size

The original is 2856 × 2142, with no usable EXIF. The supplied 72° horizontal
field of view is fixed, equivalent to 57.17° vertical. The preview is
1824 × 1368. `assets/calibration.json` stores the fitted camera; the camera was
held fixed throughout appearance work.

Scale assumes a **216 mm writing-pad width**. Four visible flat-sheet corners
constrain the pose and exposed sheet length, fitted to 254 mm. Two edge
midpoints are derived consistency checks, not independent measurements.
`./ctl camera fit` then gives RMS **3.31 pixels** at original resolution:
`run/camera_fit.txt` and `run/camfit_phone.png`. This small residual validates
internal consistency with the assumed rectangle, **not the true metric scale
or camera accuracy**. There are no independently measured landmarks at depth.
The inferred camera is about 298 mm above the desk.

| Object | Authored dimensions | State / assumed mass |
|---|---|---|
| Desk top | 1300 × 800 × 25 mm; top z = 0.750 m | Fixed |
| Cat mat | Traced outline, about 790 × 275 × 3 mm | Fixed |
| Bowl | 115 mm outer mouth diameter, 47 mm depth, roughly 3 mm wall/floor | Free, 220 g |
| Can | 48.1 mm diameter × 101.3 mm height | Free, 200 g including assumed contents |
| Flat pad | 216 × 254 × 4 mm visible stack | Fixed |
| Rolled pages | 216 mm wide × 40 mm diameter envelope | Fixed |
| Pen | 127 mm main barrel; about 138 mm including nib and ornament; 9.6 mm barrel diameter | Free, 12 g |
| Tablet/folio | 217 × 131 × 8 mm, rectangular completion of the partial view | Fixed |

The desk extends beyond the photo and supports the completed tablet. Its total
size, height and thickness are assumptions. The unseen mat outline, tablet
extent, bowl profile and paper curl are inferred. The can's brand, remaining
contents and actual dimensions were not independently established.

## Physics and construction

The bowl's rendered mesh has both inner and outer surfaces. **288 convex
wedges**, six meridian bands × 48 sectors, follow that shell. No convex hull,
plane, attachment or invisible plug seals its mouth. A continuous cavity
about 109 mm across at the inner rim admits the 45 mm test ball. The hidden
shoulder/profile was adjusted to support the photographed side-rest pose;
its exact real profile is unknown. The initial bowl pose is the result of an
unforced settling probe, stored in `assets/settled_poses.json`. It remains an
ordinary free body; nothing writes its pose while the scene runs.

The pen has a tapered barrel, raised metal clip, nib and small head/ear
ornament. Its collision barrel is an inscribed octagonal hull, deviating by
at most about 0.37 mm from the round visual barrel, plus clip and head hulls.
This avoids the persistent rolling/creep seen with a finely faceted round
mesh. The can uses a convex collider with a flat bottom. Their inertia is
computed from geometry and scaled to the stated mass.

Every desk item has collision geometry. Mat contact wedges preserve its
outline; the pad rests on the mat. The pad and rolled pages are fixed rigid
set props, not deformable paper; the roll uses a solid outer envelope. The
mat is fixed, not deformable rubber. The tablet is a fixed closed folio.
The window and blind exist only in the phone background, not as obstacles.
No springs, welds or motors hold the bowl, can or pen.

Physics: MuJoCo Warp, 50 Hz control, 10 substeps (500 Hz), 0.2 mm default
contact gap, contact stiffness 100,000 and damping 1000, enlarged contact
buffers for the compound shell. Pen gap is 0.15 mm. Friction assumptions:
desk 0.70, mat 1.00, bowl/paper 0.65, can/pen 0.55, tablet 0.70.
These are tested for simulation behavior, **not measurements of the real
materials**. Real masses cannot be recovered from the photo.

## Readings and evidence

`run/check.json` and `run/log.jsonl` record the complete final-geometry gate:
all required objects present; build, finite state, camera aspect, penetration,
rest and finite-after all **PASS**. Over 150 steps / three simulated seconds,
maximum drift is **0.93 mm** (limit 2 mm). Worst measured overlap is **0.34 mm**,
between pen and pad (limit 1 mm).

`run/rest_probe.json` records six seconds, sampled every 0.2 seconds. At three
seconds the displacement was bowl 0.00056 mm, can 0.259 mm, pen 0.777 mm;
additional drift over the next three seconds was 0.00012, 0.00085 and 0.198 mm.
These measurements are on the same movable-prop geometry and initial poses;
subsequent changes completed the static tablet, enlarged the desk and refined
UVs. The final gate above rechecks those static changes.

The ball tests run disposable scenes **inside the page process on GPU 6**.
The delivered set is reset and contains no test ball. Test ball: 45 mm diameter,
25 g, friction 0.45. A prescribed initial rolling velocity represents the kick;
there is no imposed motion after time zero and no robot involved.

- `run/ball_probe.json`: 0.35 and 0.60 m/s rolls reach about 19.4–19.6 mm behind
  the mouth plane, then come back out. A 0.90 m/s roll enters and remains.
  The zero-speed control leaves the bowl effectively unmoved.
- `run/ball_contact_proof.mp4`: three seconds of the 0.90 m/s contact test.
- `run/ball_proof.json`, `run/ball_t10.0.png`: after **ten simulated seconds**,
  the ball remains in the cavity, its center 19.28 mm behind the mouth plane.
  The free bowl has moved **7.05 mm** through contact. The ball partly protrudes
  beyond the mouth, as expected for its diameter and the bowl depth. Small
  residual contact jitter remains; this is not a promise that every kick stays in.
- `run/impact_probes.json`: separate 0.8 m/s ball impacts leave finite states;
  can displacement 0.344 mm, pen displacement 11.17 mm over two seconds. Mass
  readback is 0.200 kg and 0.012 kg. The heavier can is much harder to displace.

Pictures inspected: `run/camfit_phone.png`, `ref/plate_home.png`,
`run/cmp_005_beside.png`, **`run/cmp_005_look.png`**, **`run/snap_008.png`**
(free-view geometry), and `run/ball_t10.0.png`. Intermediate failed-rest
readings are preserved as `run/rest_probe_initial.json` and
`run/check_initial.json`; they are not the delivered state.

## Visual match and remaining differences

Final comparison against manually traced reference masks (`run/cmp_005.json`):

| Object | Width / height ratio | Center error | Mask IoU |
|---|---|---|---|
| Bowl | 1.06 / 1.00 | 10 px left, 10 px low | 0.87 |
| Can | 0.91 / 0.96 | 11 px left, 15 px high | 0.71 |
| Pen | 1.02 / 0.93 | 3 px right, 2 px low | 0.65 |

Pixels are at the 1824 × 1368 comparison resolution. Masks are hand
annotations, not exact segmentation ground truth. Whole-frame mean CIE76
color difference is **4.9**, and **4.4** behind the keyed things; PSNR is
23.2 / 24.6 dB. The background therefore does **not** meet a near-zero
pixel-difference claim.

The bowl is too broad at its rear shoulder, has simpler glaze detail and a
different highlight. The can is too narrow and looks brighter/redder than
the backlit original. The pen ornament, clip and finish remain approximations.
The page's free view has no reconstructed room/window; only the phone view
has the photographed background. Off-frame textures are extended/clamped,
not measured. The mat's hidden artwork and the can's unseen label are unknown;
can UVs extend visible ink instead of projecting the occluding bowl onto it.
The pad/roll carry baked photographic shading, which is not relit perfectly
when viewed from another angle. There is no motion reference to match.

The renderer uses an overhead area light and neutral dome, not the real
window's radiance. Contact shadows are synthesized. The clean plate is an
AI-assisted reconstruction and changes some static detail and exposure;
there is no independent empty-desk photograph. Bowl/can/pen and their old
shadows were removed, so moving the simulated bodies does not uncover ghosts.

`compositing.py` is a **twin-local** adaptation: it uses the live collider
segmentation for the plate, bypassing a depth-neighborhood guard that erased
valid inside-wall pixels beside the near rim. It does not change the library,
freeze any object pixels, or alter physics. The shell colliders supply live
occlusion; all movable objects are still drawn by the renderer.

## Generated assets and prompts

Both edits used the built-in **imagegen** tool, not the CLI/API fallback.
The project assets are self-contained:

- `ref/plate_home.png` (1448 × 1086). Prompt: remove only the drink can,
  side-lying ceramic bowl and pink pen, including their shadows; fill with
  continuing cream cat mat and pad stripes; preserve exact camera, 4:3 framing,
  desk, tablet, rolled pad, blind, wall and exposure; no replacement objects,
  robot, ball, relighting or crop. The actual result has slight static changes.
- `assets/can_albedo_source.png`. Prompt: change only the can; brighten its
  printed sidewall and silver top approximately two photographic stops and
  neutralize backlit shading for a diffuse texture; preserve artwork, lettering,
  silhouette, position and 4:3 composition; no redesign or invented brand.
  This compensates partly for applying render lighting to an already-lit photo.

## Items requiring a person

Measure pad width, bowl mouth/depth/wall profile, can dimensions and remaining
contents, pen length, mat extent and tablet model; weigh bowl, can and pen.
Supply a clean empty-desk photo and several measured landmarks at different
depths for independent calibration. These measurements may change the scale,
lighting fit and kick outcome. No answers were requested during this run.

## Changed after delivery, for the duck

Two edits to Codex's twin, made when the Microduck was put in it (`../home.py`):

- **The mat collides as one flat piece.** Codex's 23 wedges (`mat_contact_00..22.obj`) all meet at
  one point near the middle of the mat, (-0.006, -0.141). A 45 mm ball resting on that seam
  rolled off by itself, 10-35 mm in three seconds, and one placed at (-0.12, -0.05) fell
  through. The mat is flat and 3 mm thick, so its collider is now the convex hull of the
  wedges (`assets/mat_contact_hull.obj`): the ball rests still everywhere on it, and the
  only difference is 3 mm of collider over the notch between the cat's ears.
- **`scene.py` does not build when `home.py` loads it** (run as `home_twin`), so the scene is
  built once, with the duck and the ball in it. Opened by the live page it builds as before.
