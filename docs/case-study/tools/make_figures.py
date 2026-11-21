"""Build the case-study figures from data already in the repository.

Every number drawn here is read from a file under artifacts/, runs/ or config/ (named next to each
figure function). Nothing under those folders is written. Figures go to docs/case-study/figures/.

Run from the repository root:
    python docs/case-study/tools/render_cad.py      # CAD views (needs MuJoCo's offscreen renderer)
    python docs/case-study/tools/replay_traces.py   # replays recorded rounds into tools/.cache/
    python docs/case-study/tools/make_figures.py    # everything else
"""
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
CACHE = HERE / ".cache"

BLUE, RED, INK, MUTED, GRID = "#1f6fd1", "#d6453d", "#1d2733", "#5f6b7a", "#d9dee5"
GREEN, AMBER, GREY = "#2e9e5b", "#e0a321", "#aab3bf"
PIT_FILL, PIT_EDGE, FLIP_FILL, FLIP_EDGE, FLOOR = "#f6d2ce", "#c0392b", "#f3d58a", "#b8860b", "#f4f6f8"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 150, "savefig.facecolor": "white"})

SIM = yaml.safe_load((ROOT / "runs/ppo-curriculum-overnight/effective_sim.yaml").read_text(encoding="utf-8"))
ARENA = SIM["arena"]
W, H = ARENA["width_m"], ARENA["height_m"]
OUTLINE = {"blue": np.array(SIM["robots"]["autobot"]["outline_m"]), "red": np.array(SIM["robots"]["last_year"]["outline_m"])}


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------------------------------------
# Arena helpers
def draw_arena(ax, labels=False, lw=1.2):
    ax.add_patch(Rectangle((0, 0), W, H, facecolor=FLOOR, edgecolor=INK, lw=lw, zorder=0))
    r = ARENA["pit_radius_m"]
    for (cx, cy), (a0, a1) in zip(ARENA["pit_centres_m"], ((270, 360), (90, 180))):
        ax.add_patch(Wedge((cx, cy), r, a0, a1, facecolor=PIT_FILL, edgecolor=PIT_EDGE, lw=lw, zorder=1))
    ax.add_patch(Polygon(ARENA["flipper_vertices_m"], closed=True, facecolor=FLIP_FILL, edgecolor=FLIP_EDGE, lw=lw, zorder=1))
    if labels:
        ax.text(0.085, H - 0.085, "pit", ha="center", va="center", fontsize=8, color=PIT_EDGE)
        ax.text(W - 0.085, 0.085, "pit", ha="center", va="center", fontsize=8, color=PIT_EDGE)
        ax.text(W - 0.1, H - 0.1, "fixed\nflipper", ha="center", va="center", fontsize=7, color="#7a5a00")
    ax.set_xlim(-0.03, W + 0.03); ax.set_ylim(-0.03, H + 0.03); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def draw_bot(ax, x, y, yaw, which, alpha=1.0, z=5):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = OUTLINE[which] @ np.array([[c, s], [-s, c]]) + (x, y)
    colour = BLUE if which == "blue" else RED
    ax.add_patch(Polygon(pts, closed=True, facecolor=colour, edgecolor="white", lw=0.6, alpha=alpha, zorder=z))
    ax.plot([x, x + 0.06 * c], [y, y + 0.06 * s], color="white", lw=1.1, alpha=alpha, zorder=z + 1, solid_capstyle="round")


# ---------------------------------------------------------------------------------------------------------
# 1. System diagram (hand-drawn from the verified architecture; see the evidence index in the README)
def box(ax, x, y, w, h, title, sub="", face="#ffffff", edge=INK, dashed=False, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.12", facecolor=face, edgecolor=edge,
                                lw=1.2, linestyle="--" if dashed else "-", zorder=3))
    ax.text(x + w / 2, y + h * (0.63 if sub else 0.5), title, ha="center", va="center", fontsize=fs, fontweight="bold", zorder=4)
    if sub:
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", fontsize=fs - 2, color=MUTED, zorder=4, linespacing=1.15)


def arrow(ax, x0, y0, x1, y1, colour=INK, label="", dashed=False, rad=0.0, lo=0.14):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, lw=1.4, color=colour,
                                 linestyle="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}", zorder=2))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + lo, label, ha="center", va="bottom", fontsize=7, color=colour, zorder=4)


def lane(ax, y, h, title, status, colour, tint):
    ax.add_patch(FancyBboxPatch((0.1, y), 15.8, h, boxstyle="round,pad=0.0,rounding_size=0.18", facecolor=tint, edgecolor=colour, lw=1.3, zorder=1))
    ax.text(0.35, y + h - 0.3, title, ha="left", va="center", fontsize=11, fontweight="bold", color=colour, zorder=4)
    ax.text(15.65, y + h - 0.3, status, ha="right", va="center", fontsize=8.5, color=colour, zorder=4, style="italic")


def fig_system():
    fig, ax = plt.subplots(figsize=(13.5, 10.4))
    ax.set_xlim(0, 16); ax.set_ylim(0, 12.4); ax.axis("off")
    ax.text(0.1, 12.1, "AutoBots: three separate paths", fontsize=15, fontweight="bold", va="center")
    ax.text(0.1, 11.7, "Each box exists and has been exercised only as its lane label says. No path in lanes B or C carries a motor command.",
            fontsize=9, color=MUTED, va="center")

    # Lane A: manual drive
    lane(ax, 8.85, 2.5, "A  Physical robot: manual drive", "hardware: driven by hand over the radio (no bench log or video in the repo yet)", "#1b6e3c", "#eef8f1")
    ya, hb = 9.2, 1.25
    box(ax, 0.5, ya, 2.5, hb, "RadioMaster", "transmitter, sticks\n(driver: me)")
    box(ax, 3.7, ya, 2.4, hb, "ELRS receiver", "CRSF over UART\n→ GPIO 20")
    box(ax, 6.8, ya, 2.9, hb, "ESP32-C3", "RadioMaster-only sketch\nmixer + link timeout")
    box(ax, 10.4, ya, 2.3, hb, "DRV8833", "dual H-bridge\nGPIO 2, 3, 4, 5")
    box(ax, 13.4, ya, 2.2, hb, "2 × N20 motors", "differential drive\nwedge chassis")
    arrow(ax, 3.0, ya + hb / 2, 3.7, ya + hb / 2, label="2.4 GHz")
    arrow(ax, 6.1, ya + hb / 2, 6.8, ya + hb / 2)
    arrow(ax, 9.7, ya + hb / 2, 10.4, ya + hb / 2, label="PWM")
    arrow(ax, 12.7, ya + hb / 2, 13.4, ya + hb / 2)

    # Lane B: read-only telemetry
    lane(ax, 4.75, 3.8, "B  Read-only telemetry: listens, never commands", "IMU: separate bench test    ·    camera: historical footage and synthetic frames only", "#8a5a00", "#fdf6e6")
    yb1 = 6.55
    box(ax, 0.5, yb1, 2.5, hb, "LSM6DS3 IMU", "on the robot, I2C\naddress 0x6B")
    box(ax, 3.7, yb1, 2.9, hb, "ESP32-C3", "IMU-only test sketch\nprints ax, ay, az in g")
    box(ax, 7.3, yb1, 2.9, hb, "USB serial", "text lines to the PC\n(no commands back)")
    box(ax, 10.9, yb1, 4.7, hb, "Orientation label", "az_g > 0 upright  ·  az_g < 0 inverted\ntelemetry only; never fed to the drive code")
    arrow(ax, 3.0, yb1 + hb / 2, 3.7, yb1 + hb / 2, colour="#8a5a00")
    arrow(ax, 6.6, yb1 + hb / 2, 7.3, yb1 + hb / 2, colour="#8a5a00")
    arrow(ax, 10.2, yb1 + hb / 2, 10.9, yb1 + hb / 2, colour="#8a5a00")
    yb2 = 5.0
    box(ax, 0.5, yb2, 2.5, hb, "Angled camera", "March footage (not mine)\nlive camera: planned", dashed=False)
    box(ax, 3.7, yb2, 2.9, hb, "Floor mapping", "clicked floor landmarks\n→ image-to-arena transform")
    box(ax, 7.3, yb2, 2.9, hb, "Tracking", "footage: unmarked blobs\nlive plan: ArUco markers")
    box(ax, 10.9, yb2, 4.7, hb, "Top-down arena map", "positions on the arena drawing, pits and flipper overlaid\nwritten to screen and JSON report only")
    arrow(ax, 3.0, yb2 + hb / 2, 3.7, yb2 + hb / 2, colour="#8a5a00", label="frames")
    arrow(ax, 6.6, yb2 + hb / 2, 7.3, yb2 + hb / 2, colour="#8a5a00")
    arrow(ax, 10.2, yb2 + hb / 2, 10.9, yb2 + hb / 2, colour="#8a5a00")

    # Lane C: simulator
    lane(ax, 1.55, 2.9, "C  Simulator: training, live play and replay (MuJoCo only)", "simulation only: the blue policy reads simulator truth", "#1a4f9c", "#edf3fc")
    yc = 1.9
    box(ax, 0.5, yc, 2.5, hb, "Keyboard", "I drive red (WASD)\nor a scripted red")
    box(ax, 3.7, yc, 2.9, hb, "MuJoCo arena", "pits, fixed flipper, walls\n20 ms control step")
    box(ax, 7.3, yc, 2.9, hb, "Blue policy", "64-64 network picks one of\nfive tactics each step")
    box(ax, 10.9, yc, 2.2, hb, "Input log", "red + blue commands\none JSON line per step")
    box(ax, 13.6, yc, 2.0, hb, "Replay +\nbenchmarks", "")
    arrow(ax, 3.0, yc + hb / 2, 3.7, yc + hb / 2, colour="#1a4f9c")
    ax.text(3.35, yc + hb / 2 + 0.2, "red\nwheels", ha="center", va="bottom", fontsize=7, color="#1a4f9c", zorder=4, linespacing=1.0)
    arrow(ax, 6.6, yc + hb * 0.72, 7.3, yc + hb * 0.72, colour="#1a4f9c")
    ax.text(6.95, yc + hb + 0.08, "poses +\nsim IMU", ha="center", va="bottom", fontsize=7, color="#1a4f9c", zorder=4, linespacing=1.0)
    arrow(ax, 7.3, yc + hb * 0.28, 6.6, yc + hb * 0.28, colour="#1a4f9c")
    ax.text(6.95, yc - 0.06, "tactic", ha="center", va="top", fontsize=7, color="#1a4f9c", zorder=4)
    arrow(ax, 10.2, yc + hb / 2, 10.9, yc + hb / 2, colour="#1a4f9c")
    arrow(ax, 13.1, yc + hb / 2, 13.6, yc + hb / 2, colour="#1a4f9c")
    arrow(ax, 14.6, yc + hb, 8.75, yc + hb, colour="#1a4f9c", rad=0.2, label="")
    ax.text(11.7, yc + hb + 0.12, "demonstrations and regression cases\n→ next cloned policy", ha="center", va="bottom", fontsize=7.5, color="#1a4f9c", zorder=4, linespacing=1.1)

    # Barrier between sim and hardware
    ax.plot([0.1, 15.9], [4.6, 4.6], color="white", lw=0)
    ax.text(8.0, 8.67, "no command path from B or C into A: the camera tool, the IMU logger and the simulator only read",
            ha="center", va="center", fontsize=8.5, color=INK, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=INK, lw=1.0), zorder=6)
    ax.text(8.0, 4.57, "the learned policy has never controlled the physical robot", ha="center", va="center", fontsize=8.5, color=INK,
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=INK, lw=1.0), zorder=6)

    # Planned strip
    ax.add_patch(FancyBboxPatch((0.1, 0.1), 15.8, 1.2, boxstyle="round,pad=0.0,rounding_size=0.18", facecolor="#f5f5f6", edgecolor=GREY, lw=1.2, linestyle="--", zorder=1))
    ax.text(0.35, 1.02, "Planned or unverified (not drawn as connections on purpose)", fontsize=10, fontweight="bold", color=MUTED, va="center")
    ax.text(0.35, 0.5, "• merged firmware with RADIO/PC handoff + IMU telemetry: compiles and passes mocked host tests, never flashed"
                       "      • ArUco pose from my own camera: synthetic frames only\n"
                       "• camera + IMU used together: not built      • any PC-to-robot drive link: not built, not tested",
            fontsize=8, color=MUTED, va="center", linespacing=1.5)
    save(fig, "system-diagram.png")


# ---------------------------------------------------------------------------------------------------------
# 2. CAD composite (views rendered by render_cad.py)
def autocrop(image, pad=30):
    a = np.asarray(image.convert("RGB")); mask = (a < 245).any(axis=2); ys, xs = np.where(mask)
    return image.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, a.shape[1]), min(ys.max() + pad, a.shape[0])))


def fig_cad():
    names = [("cad_autobot_view0.png", "Current chassis, three-quarter view\n(assets/BattleBot v3.2.stl)"),
             ("cad_autobot_view1.png", "Current chassis, top view\n162 × 135 × 33.5 mm (STL bounds)"),
             ("cad_last_year_view0.png", "Last year's chassis, used as the red opponent model\n(assets/Opp_Chassis.stl, 110 × 125 × 34 mm)")]
    if not all((CACHE / n).exists() for n, _ in names):
        print("skip cad (run render_cad.py first)"); return
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7), gridspec_kw={"width_ratios": [1.25, 1.0, 1.15]})
    for ax, (name, title) in zip(axes, names):
        ax.imshow(autocrop(Image.open(CACHE / name))); ax.axis("off"); ax.set_title(title, fontsize=9.5, color=INK, pad=8)
    fig.text(0.5, 0.02, "CAD geometry rendered from the STL exports. Colours are arbitrary; these are not photographs of the built robot.",
             ha="center", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.08, wspace=0.05)
    save(fig, "cad-chassis.png")


# ---------------------------------------------------------------------------------------------------------
# 3. Arena drawing from the frozen simulator config (equal to config/arena_overlay.yaml)
def fig_arena():
    scenarios = load("runs/ppo-curriculum-overnight/evaluation_scenarios.json")
    spawn = next(s for s in scenarios if s["name"] == "default_spawn")
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    draw_arena(ax, lw=1.6)
    ax.set_xlim(-0.2, W + 0.2); ax.set_ylim(-0.2, H + 0.2)
    draw_bot(ax, *spawn["player"], "blue"); draw_bot(ax, *spawn["opponent"], "red")
    ax.text(spawn["player"][0], spawn["player"][1] + 0.09, "blue: learned policy\n(current chassis outline)", fontsize=8, va="bottom", ha="center", color=BLUE)
    ax.text(spawn["opponent"][0], spawn["opponent"][1] - 0.09, "red: human or scripted\n(last year's chassis outline)", fontsize=8, va="top", ha="center", color=RED)
    dim = dict(arrowstyle="<->", color=MUTED, lw=0.9)
    ax.annotate("", (0, -0.08), (W, -0.08), arrowprops=dim); ax.text(W / 2, -0.115, "1150 mm", ha="center", va="top", fontsize=8.5, color=MUTED)
    ax.annotate("", (-0.08, 0), (-0.08, H), arrowprops=dim); ax.text(-0.115, H / 2, "1150 mm", ha="right", va="center", fontsize=8.5, color=MUTED, rotation=90)
    ax.annotate("R250 pit", (0.25 * math.cos(math.radians(-45)), H + 0.25 * math.sin(math.radians(-45))), (0.3, 0.8), fontsize=8.5, color=PIT_EDGE,
                arrowprops=dict(arrowstyle="-", color=PIT_EDGE, lw=0.8))
    ax.annotate("R250 pit", (W - 0.25 * math.cos(math.radians(45)), 0.25 * math.sin(math.radians(45))), (0.72, 0.33), fontsize=8.5, color=PIT_EDGE,
                arrowprops=dict(arrowstyle="-", color=PIT_EDGE, lw=0.8))
    ax.annotate("fixed flipper boundary\n335 × 335 mm corner cut", (0.98, 0.98), (0.62, 0.74), fontsize=8.5, color="#7a5a00",
                arrowprops=dict(arrowstyle="-", color="#7a5a00", lw=0.8))
    ax.plot(0, 0, marker="o", color=INK, ms=4, zorder=6); ax.text(0.03, 0.045, "origin (plain corner)", fontsize=8, color=INK)
    ax.annotate("", (0.2, 0.0), (0.0, 0.0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0)); ax.text(0.215, 0.022, "+x", fontsize=8, va="center")
    ax.annotate("", (0.0, 0.2), (0.0, 0.0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0)); ax.text(0.025, 0.2, "+y", fontsize=8, ha="left", va="center")
    ax.set_title("Arena model shared by the simulator and the vision overlay", fontsize=11, pad=6)
    fig.text(0.5, 0.035, "Values from the arena floor drawing as held in the frozen simulator config. Not a tape measurement of the arena as built.\n"
                         "Robots are drawn at the simulator's normal spawn using their collision outlines.", ha="center", fontsize=8, color=MUTED)
    save(fig, "arena-model.png")


# ---------------------------------------------------------------------------------------------------------
# 4. What worked: turn-needed drill and ordinary starts by method
def fig_methods():
    ladder = load("artifacts/heading-ladder-results.json")["summary"]
    pilot = load("artifacts/clone-finetune-comparison.json")["final"]
    combined = load("artifacts/combined-clone-comparison.json")["final"]
    recovery = load("artifacts/reactive-recovery-comparison.json")["tables"]

    def cc(entry):  # (drill wins, ordinary wins, ordinary own falls) for combined-clone / pilot style entries
        o = entry["ordinary"]
        return (entry["turn_needed"]["all"]["wins"], o["earlier_ordinary"]["wins"] + o["new_confirmation"]["wins"] if "earlier_ordinary" in o else o["all"]["wins"] + o["new_confirmation"]["wins"],
                o["earlier_ordinary"]["own_falls"] + o["new_confirmation"]["own_falls"] if "earlier_ordinary" in o else o["all"]["own_falls"] + o["new_confirmation"]["own_falls"])

    def rec(table):
        d = table["drill_turn_needed"]
        return d["top_left"]["wins"] + d["bottom_right"]["wins"], table["ordinary"]["all"]["wins"], table["ordinary"]["all"]["own_falls"]

    rows = [
        ("Always-Advance\n(baseline)", [cc(combined["always_advance"])], GREY),
        ("PPO from scratch\n(heading ladder, 3 seeds)", [(ladder[f"heading-ladder-seed{s}/final_model"]["holdout_needing_a_turn"]["wins"], None, None) for s in (17, 23, 31)], "#8d6cc4"),
        ("PPO fine-tuned from\nthe clone (3 seeds)", [cc(pilot[f"finetune/seed{s}"]) for s in (17, 23, 31)], "#8d6cc4"),
        ("Drill-only clone\n(behaviour cloning, 3 seeds)", [cc(combined[f"drill_only_clone/seed{s}"]) for s in (0, 1, 2)], BLUE),
        ("Combined clone\n(balanced + DAgger, 3 seeds)", [cc(combined[f"balanced_dagger/seed{s}"]) for s in (0, 1, 2)], BLUE),
        ("Recovery challenger\n(selected seed)", [rec(recovery["challenger_v2"])], GREEN),
        ("Scripted teacher\n(reads simulator truth)", [cc(combined["turn_then_advance_controller"])], GREY),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True)
    panels = [(0, 324, "Turning drill: held-out starts that need a turn first\nwins out of 324"),
              (1, 204, "Ordinary starts against the original scripted red\nwins out of 204 (red text: own falls into a pit)")]
    y = 0; ticks = []; names = []
    for name, seeds, colour in rows:
        centre = y + (len(seeds) - 1) / 2
        ticks.append(centre); names.append(name)
        for values in seeds:
            for index, total, _ in panels:
                ax = axes[index]; v = values[index]
                if v is None:
                    if index == 1 and y == centre or len(seeds) == 1:
                        ax.text(2, centre, "not run on this 204-start set (0 to 3 wins of 60 on the earlier ordinary set)", va="center", fontsize=8, color=MUTED, style="italic")
                    continue
                ax.barh(y, v, height=0.72, color=colour, zorder=3)
                label = f"{v}"
                if index == 1 and values[2]:
                    ax.text(v + total * 0.012, y, label, va="center", fontsize=8.5, color=INK)
                    ax.text(v + total * 0.012 + total * 0.035 * len(label), y, f"({values[2]} own falls)", va="center", fontsize=8, color=RED)
                else:
                    ax.text(v + total * 0.012, y, label, va="center", fontsize=8.5, color=INK)
            y += 1
        y += 0.7
    for index, total, title in panels:
        ax = axes[index]; ax.set_xlim(0, total * 1.12); ax.set_title(title, fontsize=10, loc="left")
        ax.axvline(total, color=GRID, lw=1, zorder=1); ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    axes[0].set_yticks(ticks); axes[0].set_yticklabels(names, fontsize=9); axes[0].invert_yaxis()
    fig.suptitle("Reinforcement learning never learned to turn before pushing. Copying a scripted controller did.", fontsize=12.5, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.01, "Simulation only, greedy evaluation, same held-out starts and seeds for every policy (3 evaluation seeds per start). All policies read simulator truth; none is a robot controller.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    save(fig, "methods-comparison.png")


# ---------------------------------------------------------------------------------------------------------
# 5. PPO fine-tuning erases the cloned skill
def fig_finetune():
    monitor = load("artifacts/clone-finetune-comparison.json")["monitor"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), sharey=True)
    for ax, seed in zip(axes, (17, 23, 31)):
        rows = [r for r in monitor[f"finetune/seed{seed}"]]
        seen = {}
        for r in rows:
            seen[r["actor_updates"]] = r      # the warm-up row and update 0 share actor_updates = 0; keep the later one
        updates = sorted(seen)
        x = [u if u > 0 else 0.5 for u in updates]
        for pit, style, name in (("top_left", "-", "top-left pit"), ("bottom_right", "--", "bottom-right pit")):
            ax.plot(x, [seen[u]["turn_needed"][pit]["wins"] for u in updates], style, marker="o", ms=4, color=BLUE if pit == "top_left" else AMBER, label=name, lw=1.8)
        ax.set_xscale("log"); ax.set_xticks([0.5, 1, 10, 100, 1000]); ax.set_xticklabels(["0", "1", "10", "100", "1000"])
        ax.set_ylim(-2, 57); ax.set_title(f"PPO seed {seed}", fontsize=10); ax.set_xlabel("PPO actor updates after starting from the clone")
        ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("held-out turn-needed wins (of 54 per pit)"); axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.suptitle("Fine-tuning the working clone with PPO removed the turning skill, one pit at a time", fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, -0.02, "Source: artifacts/clone-finetune-comparison.json (monitor). Update 0 is the unchanged clone after a critic-only warm-up.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "finetune-collapse.png")


# ---------------------------------------------------------------------------------------------------------
# 6. Fixed-input comparison filmstrip (traces from replay_traces.py)
def fig_filmstrip():
    path = CACHE / "traces.npz"
    if not path.exists():
        print("skip filmstrip (run replay_traces.py first)"); return
    z = np.load(path)
    red_cmd = z["spin_red_commands"]
    times = [0.0, 1.0, 1.8, 5.0, 8.1, 14.0, 20.0]
    tactics = ("stop", "advance", "reverse", "turn left", "turn right")
    fig, axes = plt.subplots(2, len(times), figsize=(15.5, 6.1))
    for row, (name, title) in enumerate((("reference", "Reference clone"), ("challenger", "Recovery challenger"))):
        poses, actions, contact = z[f"spin__{name}__poses"], z[f"spin__{name}__actions"], z[f"spin__{name}__contact"]
        previous = 0
        for col, t in enumerate(times):
            ax = axes[row, col]; i = min(int(round(t / 0.02)), len(poses) - 1)
            draw_arena(ax, lw=0.8)
            ax.plot(poses[previous:i + 1, 0], poses[previous:i + 1, 1], color=BLUE, lw=1.0, alpha=0.75, zorder=3)
            ax.plot(poses[previous:i + 1, 3], poses[previous:i + 1, 4], color=RED, lw=1.0, alpha=0.75, zorder=3, linestyle=(0, (3, 2)))
            draw_bot(ax, *poses[i, 3:6], "red"); draw_bot(ax, *poses[i, 0:3], "blue")
            j = min(i, len(actions) - 1)
            gap = float(np.linalg.norm(poses[i, :2] - poses[i, 3:5]))
            idle = bool(np.all(red_cmd[min(j, len(red_cmd) - 1)] == 0))
            touching = bool(contact[j])
            state = f"blue: {tactics[int(actions[j])]}\ngap {gap:.2f} m" + (" · touching" if touching else " · not touching")
            if row == 0:
                ax.set_title(f"t = {t:.1f} s · red input: {'idle' if idle else 'driving'}\n{state}", fontsize=8, color=INK)
            else:
                ax.set_xlabel(state, fontsize=8, color=INK, labelpad=4)
            if col == 0:
                ax.set_ylabel(title, fontsize=10, fontweight="bold", color=INK, labelpad=6)
            previous = i
    fig.suptitle("Fixed-input comparison: the same recorded red commands replayed against two blue policies (not two live matches)",
                 fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.012, "Drawn from simulator state when the logged round-2 red commands (18 Nov, 00:00 session) are replayed step for step. Lines show each robot's path since the previous panel.\n"
                          "Red is an open-loop recording: once the challenger pins it at about 1 s, the later inputs no longer mean what they meant in the original round.",
             fontsize=8, color=MUTED, linespacing=1.3)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94), w_pad=0.2, h_pad=0.6)
    save(fig, "fixed-input-filmstrip.png")


# ---------------------------------------------------------------------------------------------------------
# 7. Matched benchmark: reference clone vs recovery challenger (scripted test cases)
def fig_benchmark():
    tables = load("artifacts/reactive-recovery-comparison.json")["tables"]
    ref, ch = tables["reference"], tables["challenger_v2"]

    def drill(t):
        a, b = t["drill_turn_needed"]["top_left"], t["drill_turn_needed"]["bottom_right"]
        return {k: a[k] + b[k] for k in ("episodes", "wins", "own_falls", "timeouts", "red_self_falls", "draws", "simulation_bugs")}

    sets = [("Fresh normal starts\n(stationary, moving-away,\nside-changing reds)", lambda t: t["fresh"]["all"]),
            ("Any geometry\n(both robots placed anywhere)", lambda t: t["anywhere"]["all"]),
            ("Harder scripted reds\n(five driving styles)", lambda t: t["pool"]["harder"]),
            ("Ordinary starts\n(original scripted red)", lambda t: t["ordinary"]["all"]),
            ("Turning drill\n(both pits)", drill)]
    fig = plt.figure(figsize=(13.5, 8.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.75], hspace=0.55, wspace=0.42)

    # (a) reaches red
    ax = fig.add_subplot(grid[0, 0])
    labels = ["Fresh normal starts", "Any geometry"]; keys = ["fresh", "anywhere"]
    for k, key in enumerate(keys):
        for offset, table, colour, name in ((-0.2, ref, GREY, "Reference clone"), (0.2, ch, GREEN, "Recovery challenger")):
            c = table[key]["all"]; share = c["reaches_red"] / c["episodes"]
            ax.barh(k + offset, share, height=0.36, color=colour, zorder=3, label=name if k == 0 else None)
            ax.text(share + 0.015, k + offset, f"{c['reaches_red']} / {c['episodes']}", va="center", fontsize=9)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("Episodes in which blue touches red", fontsize=10.5, loc="left"); ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)

    # (b) seconds spinning / stalled
    ax = fig.add_subplot(grid[0, 1])
    for k, key in enumerate(keys):
        for offset, table, colour in ((-0.2, ref, GREY), (0.2, ch, GREEN)):
            c = table[key]["all"]; spin, stall = c["mean_spinning_s"], c["mean_stalled_s"]
            ax.barh(k + offset, spin, height=0.36, color=RED, zorder=3, label="spinning" if (k == 0 and offset < 0) else None)
            ax.barh(k + offset, stall, left=spin, height=0.36, color=colour, zorder=3, hatch="///", edgecolor="white", lw=0,
                    label=None)
            who = "reference" if offset < 0 else "challenger"
            ax.text(spin + stall + 0.3, k + offset, f"{who}: {spin:.2f} s spinning\n+ {stall:.1f} s stalled", va="center", fontsize=8.2, linespacing=1.05)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(["Fresh normal\nstarts", "Any\ngeometry"]); ax.invert_yaxis(); ax.set_xlim(0, 20)
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_xlabel("mean seconds per episode (episodes last up to 20 s)")
    ax.set_title("Time spent spinning on the spot or stalled", fontsize=10.5, loc="left")

    # (c) outcomes
    ax = fig.add_subplot(grid[1, :])
    parts = [("wins", GREEN, "win: red pushed into a pit"), ("timeouts", "#c9d1db", "timeout (20 s, nobody falls)"), ("own_falls", RED, "own fall: blue ends in a pit"),
             ("other", "#7d5ba6", "other: wall breach reported by the simulator, or red falling by itself")]
    y = 0; ticks = []; names = []
    for label, getter in sets:
        for table, who in ((ref, "reference"), (ch, "challenger")):
            c = getter(table); total = c["episodes"]
            counts = {"wins": c["wins"], "timeouts": c["timeouts"], "own_falls": c["own_falls"]}
            counts["other"] = total - sum(counts.values())
            left = 0.0
            for key, colour, _ in parts:
                share = counts[key] / total
                ax.barh(y, share, left=left, height=0.74, color=colour, zorder=3, edgecolor="white", lw=0.6)
                if counts[key] and share > 0.035:
                    ax.text(left + share / 2, y, str(counts[key]), ha="center", va="center", fontsize=8.5, color="white" if key != "timeouts" else INK, zorder=4)
                left += share
            ax.text(-0.012, y, who, ha="right", va="center", fontsize=8.5, color=MUTED)
            ax.text(1.012, y, f"{counts['wins']} wins · {counts['own_falls']} own falls · {counts['timeouts']} timeouts"
                              + (f" · {counts['other']} other" if counts["other"] else "") + f"  (of {total})", va="center", fontsize=8.3)
            ticks.append(y); y += 1
        names.append((y - 1.5, label)); y += 0.75
    for centre, label in names:
        ax.text(-0.115, centre, label, ha="right", va="center", fontsize=9, color=INK, linespacing=1.1)
    ax.set_yticks([]); ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.spines["left"].set_visible(False)
    handles = [Rectangle((0, 0), 1, 1, color=colour) for _, colour, _ in parts]
    ax.legend(handles, [name for _, _, name in parts], frameon=False, fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.07))
    ax.set_title("Outcomes on matched test cases (counts inside the bars)", fontsize=10.5, loc="left")

    fig.suptitle("Reference clone vs recovery challenger on the same scripted test cases", fontsize=13, fontweight="bold", x=0.02, ha="left", y=0.975)
    fig.text(0.02, 0.925, "Simulation only. Every red here is scripted. No human round is in this chart. The challenger closes the gap far more reliably; most rounds still end as timeouts.",
             fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.2, right=0.76, top=0.87, bottom=0.1)
    save(fig, "benchmark-reference-vs-challenger.png")


# ---------------------------------------------------------------------------------------------------------
# 8. Vision overlay composite: raw reference frame, and a tracked frame beside its top-down map
def fig_vision():
    base = ROOT / "artifacts/vision-footage/historical_bristol_2026_03_22"
    raw = Image.open(base / "reference_frame.png").convert("RGB")
    tracked = Image.open(base / "clip_433s_last_frame.png").convert("RGB")
    w, h = tracked.size
    # the tool's own layout: camera view (1280 wide) | top-down panel, with a status text strip underneath
    bar_top = int(np.argmax(np.asarray(tracked.convert("L")).mean(axis=1)[h // 2:] < 12)) + h // 2
    overlay = tracked.crop((0, 0, 1280, bar_top)); topdown = tracked.crop((1280, 0, w, bar_top))
    ratios = [raw.width / raw.height, overlay.width / overlay.height, topdown.width / topdown.height]
    width = 13.5; height = width / (sum(ratios) + 0.06) + 0.75
    fig = plt.figure(figsize=(width, height))
    grid = fig.add_gridspec(1, 3, width_ratios=ratios, wspace=0.03, left=0.005, right=0.995, top=1 - 0.32 / height, bottom=0.42 / height)
    for index, (image, title) in enumerate(((raw, "1  Raw angled frame (reference, 7:18)"), (overlay, "2  Arena outline and blob tracks drawn on"),
                                            (topdown, "3  Mapped onto the arena drawing"))):
        ax = fig.add_subplot(grid[0, index]); ax.imshow(image, aspect="auto"); ax.axis("off"); ax.set_title(title, fontsize=10, loc="left", pad=5)
    fig.text(0.005, 0.1 / height, "HISTORICAL footage (Bristol University Antweight Robot Wars, 22 March 2026, 360p, not my camera). Approximate calibration; read-only tracking of unmarked robots. "
                                  "Panels 2 and 3: last frame of the 24 s test clip.", fontsize=8, color=MUTED)
    save(fig, "vision-overlay-historical.png")


# ---------------------------------------------------------------------------------------------------------
# 9. Proposed ArUco marker layout (placeholder values in config/arena_overlay.yaml; nothing installed)
def fig_markers():
    cfg = yaml.safe_load((ROOT / "config/arena_overlay.yaml").read_text(encoding="utf-8"))["bot"]
    size = cfg["marker_size_m"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6))
    for ax, face in zip(axes, ("top", "bottom")):
        outline = OUTLINE["blue"] @ np.array([[0, 1], [-1, 0]])      # nose (+x) drawn pointing up
        ax.add_patch(Polygon(outline, closed=True, facecolor="#dfeaf8", edgecolor=BLUE, lw=1.5))
        for m in (m for m in cfg["markers"] if m["face"] == face):
            x, y = -m["left_m"], m["forward_m"]
            ax.add_patch(Rectangle((x - size / 2, y - size / 2), size, size, facecolor="white", edgecolor=INK, lw=1.2, linestyle="--"))
            ax.text(x, y, f"ID {m['id']}", ha="center", va="center", fontsize=9)
        ax.annotate("nose", (0, 0.068), (0, 0.092), ha="center", fontsize=8.5, color=BLUE, arrowprops=dict(arrowstyle="<|-", color=BLUE, lw=1.0))
        ax.set_xlim(-0.1, 0.1); ax.set_ylim(-0.09, 0.11); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{face} face" + (" (robot upside down)" if face == "bottom" else ""), fontsize=10)
    fig.suptitle("PROPOSED marker layout: not printed, not installed, not tested", fontsize=11.5, fontweight="bold", color="#8a5a00")
    fig.text(0.5, 0.03, f"Placeholder values from config/arena_overlay.yaml: {cfg['dictionary']}, {size * 1000:.0f} mm markers, 2 × 2 grid at ±35 mm / ±30 mm from the bot centre.\n"
                        "Different IDs on each face would let the camera tell upright from inverted. Chassis shape is the simulator's collision outline, not the real chassis.",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    save(fig, "marker-layout-proposed.png")


# ---------------------------------------------------------------------------------------------------------
# 10. Timeline
def fig_timeline():
    from datetime import date
    lanes = {"Robot + firmware": 3.6, "Simulator + learning": 1.8, "Vision": 0}
    colours = {"Robot + firmware": "#1b6e3c", "Simulator + learning": "#1a4f9c", "Vision": "#8a5a00"}
    events = [
        ("Vision", date(2026, 9, 20), "Control centre shows\nC525 feed at 30 fps", 1),
        ("Robot + firmware", date(2026, 9, 23), "Architecture + parts list;\nultrasonic positioning dropped", 1),
        ("Vision", date(2026, 10, 11), "Opponent trajectory\npredictor prototype", -1),
        ("Robot + firmware", date(2026, 11, 10), "RadioMaster\ndrive sketch", 1),
        ("Simulator + learning", date(2026, 11, 14), "MuJoCo arena;\nfirst PPO backs\ninto a wall", 1),
        ("Simulator + learning", date(2026, 11, 15), "8 h PPO run jitters;\nturning drill; cloning\nworks where PPO fails", -1),
        ("Simulator + learning", date(2026, 11, 17), "Combined clone\n+ DAgger; live play", 1),
        ("Robot + firmware", date(2026, 11, 17), "IMU found on I2C;\nleft-wheel incident", -1),
        ("Simulator + learning", date(2026, 11, 18), "Live spin diagnosed;\nrecovery challenger", -1),
        ("Vision", date(2026, 11, 18), "Arena overlay on\nMarch footage", 1),
        ("Robot + firmware", date(2026, 11, 21), "Competition day\n(not yet happened)", 1),
    ]
    start = date(2026, 9, 14)

    def xpos(d):
        days = (d - start).days
        # compress the quiet weeks so November is readable: piecewise linear
        knots = [(0, 0.0), (10, 1.6), (27, 2.6), (55, 3.6), (61, 5.0), (65, 9.4), (68, 11.4)]
        for (d0, x0), (d1, x1) in zip(knots, knots[1:]):
            if d0 <= days <= d1:
                return x0 + (x1 - x0) * (days - d0) / (d1 - d0)
        return knots[-1][1]

    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    for name, level in lanes.items():
        ax.plot([-0.05, 11.9], [level, level], color=colours[name], lw=2, alpha=0.35, zorder=1)
        ax.text(-0.25, level, name, ha="right", va="center", fontsize=10, fontweight="bold", color=colours[name])
    for name, d, text, side in events:
        x = xpos(d); level = lanes[name]; planned = d == date(2026, 11, 21)
        ax.plot(x, level, "o", ms=9, markerfacecolor="white" if planned else colours[name], markeredgecolor=colours[name], mew=1.8, zorder=3)
        ax.text(x, level + 0.17 * side, f"{d.day} {d.strftime('%b')}\n{text}" if side > 0 else f"{text}\n{d.day} {d.strftime('%b')}", ha="center",
                va="bottom" if side > 0 else "top", fontsize=7.8, color=INK, linespacing=1.15)
    for label, d in (("Sep", date(2026, 9, 14)), ("Oct", date(2026, 10, 1)), ("Nov", date(2026, 11, 1)), ("10 Nov", date(2026, 11, 10)), ("14 Nov", date(2026, 11, 14)), ("18 Nov", date(2026, 11, 18)), ("21 Nov", date(2026, 11, 21))):
        ax.text(xpos(d), -0.92, label, ha="center", va="top", fontsize=8, color=MUTED); ax.plot([xpos(d)] * 2, [-0.86, -0.8], color=MUTED, lw=0.8)
    ax.text(xpos(date(2026, 10, 20)), -1.12, "time axis is compressed before 10 November", ha="center", va="top", fontsize=7.5, color=MUTED, style="italic")
    ax.set_xlim(-2.6, 12.2); ax.set_ylim(-1.3, 4.6); ax.axis("off")
    ax.set_title("Project timeline (dates from git history and file timestamps in the repository)", fontsize=11.5, fontweight="bold", loc="left")
    save(fig, "timeline.png")


# ---------------------------------------------------------------------------------------------------------
# 11. Existing evidence images, copied (and cropped where needed) rather than redrawn
def copy_existing():
    FIG.mkdir(exist_ok=True)
    for source, name in (("runs/live-round-diagnostic-2026-11-18/round-spin.png", "spin-diagnosis-round2.png"),
                         ("runs/live-round-diagnostic-2026-11-18/red-position-probe.png", "spin-red-position-probe.png"),
                         ("runs/reactive-recovery-2026-11-18/before-after-recorded-round.png", "fixed-input-paths-and-gap.png")):
        shutil.copyfile(ROOT / source, FIG / name); print("copied", name)
    # the 14 Nov screenshot of the first PPO policy parked against the wall: crop the black margins
    early = Image.open(ROOT / "artifacts/ppo-debug-10k-pre-timeout.png").convert("RGB")
    early.crop((170, 20, 830, 680)).save(FIG / "early-policy-against-wall.png", optimize=True); print("cropped early-policy-against-wall.png")
    hud = Image.open(ROOT / "runs/reactive-recovery-2026-11-18/play-check/challenger-frame-synthetic.png").convert("RGB")
    hud.save(FIG / "sim-play-window.png", optimize=True); print("copied sim-play-window.png")
    scene = Image.open(ROOT / "artifacts/final-overhead-3d.png").convert("RGB")
    scene.crop((170, 20, 830, 680)).save(FIG / "sim-arena-3d.png", optimize=True); print("cropped sim-arena-3d.png")


if __name__ == "__main__":
    copy_existing()
    fig_system(); fig_cad(); fig_arena(); fig_methods(); fig_finetune(); fig_filmstrip(); fig_benchmark(); fig_vision(); fig_markers(); fig_timeline()
