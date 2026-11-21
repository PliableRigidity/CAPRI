# Arena geometry: one design, one place, and the 1130 vs 1150 mm question

Written 21 Nov 2026. Desk investigation of files only; nothing was measured on the real arena.

## Where the numbers live

| What | Where |
| --- | --- |
| The competition arena (canonical, the only place to read it) | `config/arena_geometry.yaml`, loaded by `control_center/tools/arena_design.py` |
| Its source | `Arena Floor Technical Drawing.pdf`, Fusion Drawings, **Rev. 1**, Alex Chung, 11/11/2025, millimetres, scale 1/1 (sha256 `c6f59bdf…aeda0`, recorded in the YAML) |
| The simulator's copy | `config/sim.yaml` (`arena:`); `tests/test_arena_design.py` fails if it and the design ever differ |
| What a calibration saved | the profile's `arena.json` (LIVE: `config/arena_overlay_arena.json`), including a `design` block: the design numbers used, any overrides, and the drawing's hash |

The simulator's training config is not rewired to read the YAML (its frozen evaluation settings must not change under a
trained policy); it is compared with it instead.

## What the drawing says

- Floor panel **1150 x 1150 mm**.
- Two **R250** quarter-circle pit cut-outs on opposite corners. The two **900** dimensions are the straight edges left on
  each side of a pit corner (1150 - 250), not a second arena size.
- One **135 degree** chamfer with **335 mm** legs on a third corner: the flipper.
- Nothing about walls. The simulator adds 20 mm walls outside the floor; that is a simulator choice, not part of the drawing.

Frame (the simulator's, and the one the policy's observation uses): origin at the plain corner, pits centred on the corners
(0, H) and (W, 0), flipper corner (W, H). The drawing as printed already has this layout.

## 1130 vs 1150

- **1150 mm is the design.** It is the drawing, `sim.yaml`, `config/control_center.yaml` and the competition rules' number.
- **1130 mm appears only in the operator's own typed draft** in the Command Centre's remembered state (the old ARENA tab
  asked for the width and the draft held `1.13`). It is not in the drawing, the simulator or any config file.
- A plausible reason would be a **10 mm inset on each side** (walls or tape standing 10 mm in from the floor edge, so the
  visible playing area is 1130). That was never measured, so it is a hypothesis, not a finding.

Decision: the default is the drawing (1150). Nothing was changed silently. If the real arena measures differently
the operator enters it once, deliberately, under **ARENA > Advanced: measured differences**: an inset per side (0.010 m
gives 1130 mm) or a width, height, pit radius or flipper leg, with a reason. The override is stored in the profile's
arena file next to the design numbers it replaced, and the tab shows "ADVANCED OVERRIDE in use" every time.

Open check for the real arena: measure the floor edge to floor edge and the visible playing area with a tape, once.
If both are 1150, there is nothing to enter. If the playing area is 1130, enter an inset of 10 mm and note it.

## What is no longer typed

The operator no longer enters the arena size, pit radii or flipper vertices for the LIVE profile. They click the four
floor corners, the centre of pit 1, the centre of pit 2, and one point on the flipper; the software knows the sizes. The
flipper's corner and direction come from which corner the click is nearest (the legs run along the walls), and the
two pits must be on opposite corners. HOME TEST (an A4 patch, no drawing) still asks for its own dimensions and is
never given the competition numbers.
