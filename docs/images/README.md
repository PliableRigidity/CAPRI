# Planning diagrams, October 2026

These two diagrams were drawn in October 2026, when the project was still called AutoBot, and show the architecture
as it was **planned** at that time. They are kept as a design record.

| File | What it shows |
|---|---|
| `planned-architecture-command-control-2026-10.png` | Planned command and control architecture: camera, control centre, base-station ESP32, onboard arbiter |
| `planned-architecture-detailed-2026-10.png` | Planned signal and control flow in detail: perception, estimation and prediction, planning, actuation |

They do not describe Capri V1 as built. In particular the ESP-NOW base station, the onboard arbiter, the state
estimator, the tactical planner and the trajectory predictor's place in the control loop were never built, and the
markers are ArUco 4x4, not AprilTags. In the diagrams' own key, dashed boxes are planned integrations.

For the system as it actually stands, see the diagrams in the [README](../../README.md#hardware-architecture) and
[Figure 3 of the case study](../case-study/README.md#1-the-three-parts-and-how-they-relate).
