"""Matplotlib plots and animation of a parking run."""
from __future__ import annotations

import math
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon
from matplotlib.ticker import MaxNLocator

from .scenario import Scenario
from .simulator import RunResult

C_LINE, C_OBS, C_PATH, C_CAR, C_GOAL = "#9aa0a6", "#5f6368", "#1a73e8", "#e8710a", "#188038"
C_INK_MUTED = "#5f6368"
SEEN_RAMP = LinearSegmentedColormap.from_list("first_seen", ["#86b6ef", "#184f95"])   # earlier -> later


def _draw_lot(ax, sc: Scenario):
    for (a, b) in sc.lines:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=C_LINE, lw=1.2, zorder=1)
    for ob in sc.obstacles:
        ax.add_patch(Polygon(ob, closed=True, fc=C_OBS, ec="none", alpha=0.85, zorder=2))
    if sc.goal is not None:     # an aisle lot has no target until the driver chooses one
        ax.add_patch(Polygon(sc.target_stall, closed=True, fc=C_GOAL, ec="none", alpha=0.12, zorder=0))
        g = sc.goal
        ax.add_patch(Polygon(sc.vehicle.footprint(g.x, g.y, g.yaw), closed=True, fill=False,
                             ec=C_GOAL, ls="--", lw=1.2, zorder=3))
    xmin, xmax, ymin, ymax = sc.bounds
    ax.set_xlim(xmin + 3, xmax - 3)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def _draw_path(ax, res: RunResult):
    p = res.planned
    for a, b, d in p.segments():
        ax.plot(p.x[a:b + 1], p.y[a:b + 1], color=C_PATH, lw=1.5,
                ls="-" if d > 0 else "--", zorder=4)


def plot_run(sc: Scenario, res: RunResult, path: str, title: Optional[str] = None) -> None:
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], hspace=0.35)
    ax = fig.add_subplot(gs[:, 0])
    _draw_lot(ax, sc)
    _draw_path(ax, res)
    log = res.log
    ax.plot(log["x"], log["y"], color=C_CAR, lw=1.0, zorder=5, label="rear axle (driven)")
    for i in range(0, len(log["x"]), 75):
        ax.add_patch(Polygon(sc.vehicle.footprint(log["x"][i], log["y"][i], log["yaw"][i]),
                             closed=True, fill=False, ec=C_CAR, lw=0.6, alpha=0.5, zorder=5))
    ax.add_patch(Polygon(sc.vehicle.footprint(log["x"][-1], log["y"][-1], log["yaw"][-1]),
                         closed=True, fc=C_CAR, ec=C_CAR, alpha=0.6, zorder=6))
    ax.plot([], [], color=C_PATH, lw=1.5, label="plan: forward")
    ax.plot([], [], color=C_PATH, lw=1.5, ls="--", label="plan: reverse")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title(title or res.summary().split("  [")[0], fontsize=9)

    a1 = fig.add_subplot(gs[0, 1])
    a1.plot(log["t"], [math.degrees(v) for v in log["steer_cmd"]], color=C_PATH, lw=1, label="command")
    a1.plot(log["t"], [math.degrees(v) for v in log["delta"]], color=C_CAR, lw=1, label="actual")
    a1.set_ylabel("road-wheel angle [deg]")
    a1.legend(fontsize=8, frameon=False)
    a1.grid(alpha=0.3)
    a2 = fig.add_subplot(gs[1, 1])
    a2.plot(log["t"], log["speed_cmd"] * 3.6, color=C_PATH, lw=1, label="command")
    a2.plot(log["t"], log["v"] * 3.6, color=C_CAR, lw=1, label="actual")
    a2.set_ylabel("speed [km/h]")
    a2.set_xlabel("time [s]")
    a2.legend(fontsize=8, frameon=False)
    a2.grid(alpha=0.3)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_search(res: RunResult, path: str, title: Optional[str] = None) -> None:
    """Top-down search run with detected slots shaded by when they were first seen, and a
    timeline of the detected-slot count with the run's milestones."""
    sc, log = res.search.scenario, res.log
    fig, (ax, tl) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]},
                                 constrained_layout=True)
    _draw_lot(ax, sc)
    ax.set_xlim(*sc.bounds[:2])

    t_end = float(log["t"][-1])
    seen = [slot.first_seen_t for slot in res.search.slots]
    shade = ScalarMappable(Normalize(min(seen, default=0.0), max(seen, default=1.0)), SEEN_RAMP)
    for slot in res.search.slots:
        ax.add_patch(Polygon(slot.corners, closed=True, fc=shade.to_rgba(slot.first_seen_t), ec="white",
                             lw=1.5, zorder=1.5))
        # Label in the aisle, just off the opening, where parked or driving cars don't cover it
        ox, oy = (slot.corners[0][0] + slot.corners[1][0]) / 2, slot.corners[0][1]
        below = oy > sc.aisle_y
        ax.text(ox, oy + (-0.25 if below else 0.25), f"{slot.slot_id}\n{slot.first_seen_t:.1f} s",
                ha="center", va="top" if below else "bottom", fontsize=7, color=C_INK_MUTED, zorder=7)
    fig.colorbar(shade, ax=ax, label="slot first seen [s]", location="bottom", shrink=0.4, aspect=45)

    if res.planned is not None:
        _draw_path(ax, res)
    ax.plot(log["x"], log["y"], color=C_CAR, lw=1.2, zorder=5)
    ax.add_patch(Polygon(sc.vehicle.footprint(log["x"][-1], log["y"][-1], log["yaw"][-1]),
                         closed=True, fc=C_CAR, ec=C_CAR, alpha=0.6, zorder=6))
    ax.plot([], [], color=C_CAR, lw=1.2, label="rear axle (driven)")
    if res.planned is not None:
        ax.plot([], [], color=C_PATH, lw=1.5, label="plan: forward")
        ax.plot([], [], color=C_PATH, lw=1.5, ls="--", label="plan: reverse")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, edgecolor="none")
    ax.set_title(title or res.summary().split("  [")[0], fontsize=9)

    tl.step([0.0, *seen, t_end], [0, *range(1, len(seen) + 1), len(seen)], where="post", color=C_PATH, lw=2)
    tl.set_ylabel("slots detected")
    tl.set_xlabel("time [s]")
    tl.set_xlim(0.0, t_end)
    tl.set_ylim(0, max(len(seen), 1) * 2.0)
    tl.yaxis.set_major_locator(MaxNLocator(integer=True))
    tl.grid(axis="y", alpha=0.3)
    for name, state in (("selected", "SLOT_SELECTED"), ("stopped", "WAIT_CONFIRM"),
                        ("confirmed", "PLANNING"), ("parked", "COMPLETED")):
        times = [tr.t for tr in res.transitions if tr.to_state.value == state]
        if times:
            tl.axvline(times[-1], color=C_INK_MUTED, lw=0.8, ls=":")
            tl.annotate(name, (times[-1], 1.0), xycoords=("data", "axes fraction"), xytext=(0, -2),
                        textcoords="offset points", rotation=90, ha="right", va="top",
                        fontsize=8, color=C_INK_MUTED)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def animate_run(sc: Scenario, res: RunResult, path: str, fps: int = 20, speedup: float = 2.0) -> None:
    log = res.log
    dt = log["t"][1] - log["t"][0]
    stride = max(1, int(round(speedup / (fps * dt))))
    frames = list(range(0, len(log["t"]), stride)) + [len(log["t"]) - 1] * fps
    fig, ax = plt.subplots(figsize=(9, 7))
    _draw_lot(ax, sc)
    _draw_path(ax, res)
    car = Polygon(sc.vehicle.footprint(log["x"][0], log["y"][0], log["yaw"][0]),
                  closed=True, fc=C_CAR, ec=C_CAR, alpha=0.7, zorder=6)
    ax.add_patch(car)
    trail, = ax.plot([], [], color=C_CAR, lw=1, zorder=5)
    txt = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", fontsize=9, family="monospace")

    def upd(i):
        car.set_xy(sc.vehicle.footprint(log["x"][i], log["y"][i], log["yaw"][i]))
        trail.set_data(log["x"][:i + 1], log["y"][:i + 1])
        gear = "R" if log["gear"][i] < 0 else "D"
        txt.set_text(f"t {log['t'][i]:5.1f} s  gear {gear}  v {log['v'][i]*3.6:+4.1f} km/h  "
                     f"steer {math.degrees(log['delta'][i]):+5.1f} deg")
        return car, trail, txt

    anim = animation.FuncAnimation(fig, upd, frames=frames, blit=True)
    writer = animation.FFMpegWriter(fps=fps) if path.endswith(".mp4") else animation.PillowWriter(fps=fps)
    anim.save(path, writer=writer, dpi=90)
    plt.close(fig)
