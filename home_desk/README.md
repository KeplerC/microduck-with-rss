# Home desk set

Start from this folder:

```bash
CUDA_VISIBLE_DEVICES=0 ./ctl up
```

Open the address it prints. In **World model**, select **Lock to → phone**
for the photo-matched view. Select **free** or **overview** to inspect geometry.
The server is left paused/reset at delivery. Resume for physics or use:

```bash
./ctl run 150
./ctl reset
./ctl snapshot --camera phone --beside ref/IMG_9067.jpg
```

`scene.py` contains the desk, mat, hollow bowl, can, pad, rolled pages, pen,
and tablet. No robot, ball or routine is included. Bowl/can/pen are free bodies.
The bowl has been tested with a rolling 45 mm ball; see
[the video](run/ball_contact_proof.mp4), [ten-second result](run/ball_t10.0.png),
[comparison](run/cmp_005_look.png), and [REPORT.md](REPORT.md).

All runtime assets are in this folder. Keep `scene.py`, `compositing.py`,
`assets/`, and `ref/` together. `run/` contains evidence and optional authoring
scripts; it is not required to build the scene. No model download is needed.

For later Microduck integration: world coordinates are metres, z-up; the desk
is at z=0.750 and the mat top at z=0.753. Approximate clear mat center is
(-0.05, -0.15). This scene uses `cell='void'`: an `on='surface'` helper alone
would refer to the floor, not this added desk. Place the duck on the desk with
the mouse, or explicitly adapt its initial world pose. Keep control at 50 Hz.
Dimensions and masses remain photo-based assumptions, detailed in the report.

To reproduce asset authoring, the optional scripts in `run/` are executed in
this order: `build_assets.py`, `detail_pen.py`, `complete_tablet.py`,
`refine_static_textures.py`, `refine_can_uv.py`, then `./ctl reload`.
The generated images, calibration and settled pose are already supplied.
Do not run the repository's library test tiers for this twin.
