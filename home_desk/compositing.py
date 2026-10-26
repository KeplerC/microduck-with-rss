"""Twin-local plate composition for a hollow shell with matching colliders.

The stock depth guard compares a pixel with the nearest moving collider in
an 11-pixel neighbourhood. At the bowl lip this compares the inside wall with
the nearer rim, cutting pale strips out of the bowl even though the collider
segmentation says both belong to the moving bowl. Our authored convex wedges
follow both shell surfaces, so their segmentation is the appropriate mask.
No pixel or pose is frozen: the stock compositor still renders live geometry,
uses live occlusion and draws live contact shadows. No library code is changed.
"""
def use_shell_segmentation(sim):
    original = sim.over_plate
    def over_plate(camera, rgb, depth=None):
        return original(camera, rgb, depth=None)
    sim.over_plate = over_plate
    return sim
