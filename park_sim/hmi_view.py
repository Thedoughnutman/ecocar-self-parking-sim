"""The interactive window, in pygame-ce: a RunObserver that draws the run and feeds keys to hmi.KeyState. It only reads the run: nothing here
feeds back but the person's inputs. An optional dev dependency, imported only by this module. Tests use SDL's dummy video driver.

The lot gets the full width, since it's long and thin, with a row of status cards under it. Everything that doesn't move sits on a saved surface, so a frame is
one blit plus what moved. pygame.SCALED presents each frame through SDL's renderer, on the GPU where the machine has one. Shapes are drawn into the frame by the CPU, in SDL's C code."""
from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pygame
from pygame import gfxdraw

from .hmi import (OFFERED, REJECTED, SELECTED, UNCONFIRMED, KeyState, StallNumbers, Units, explain_result, plan_failure_text, selected_now, slot_at,
                  slot_statuses, slots_in_map, status_from_tick, summary_text)
from .localization import MapToOdom
from .sensors.line_camera import LineCameraMeasurement
from .sensors.ultrasonic import UltrasonicMeasurement, UltrasonicSensor
from .simulator import RunObserver, RunTick


def _rgb(text: str) -> Tuple[int, int, int]:
    return tuple(int(text[i:i + 2], 16) for i in (1, 3, 5))


BACKGROUND, SURFACE, ASPHALT, BORDER = _rgb("#0d1117"), _rgb("#161b22"), _rgb("#12171e"), _rgb("#2d333b")
TEXT, MUTED = _rgb("#e6edf3"), _rgb("#8b949e")
BLUE, GREEN, AMBER, RED, PURPLE, TEAL, CYAN, PINK = (_rgb(c) for c in ("#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff", "#2dd4bf", "#39c5cf", "#f778ba"))
PARKED, PARKED_EDGE, PAINT, CURB = _rgb("#30363d"), _rgb("#484f58"), _rgb("#454c55"), _rgb("#4a3f35")
SLOT_COLORS = {UNCONFIRMED: MUTED, OFFERED: BLUE, SELECTED: GREEN, REJECTED: RED}
STATE_COLORS = {"OFF": _rgb("#484f58"), "SEARCHING": BLUE, "SLOTS_AVAILABLE": BLUE, "SLOT_SELECTED": AMBER, "WAIT_CONFIRM": AMBER, "PLANNING": PURPLE,
                "MANEUVERING": GREEN, "PAUSED": AMBER, "COMPLETED": GREEN, "ABORTED": RED, "FAULT": RED}
KEYS = (("W S", "speed"), ("A D", "steer"), ("B", "brake"), ("P", "feature"), ("Click", "pick a stall"), ("Enter", "confirm / pause"), ("T", "take over"),
        ("Z", "zoom"), ("U", "units"), ("R", "again"), ("N", "next seed"), ("Esc", "quit"))
ENTER_STAYS_ON_IN = ("WAIT_CONFIRM", "PLANNING", "MANEUVERING", "PAUSED")
SHOW_PLAN_FAILURE_IN = ("SEARCHING", "SLOTS_AVAILABLE")     # after this, a new selection is under way
LEGEND = (("offered", BLUE), ("selected", GREEN), ("no path", RED), ("plan", PURPLE), ("driven", TEAL), ("ultrasonic", AMBER), ("camera", CYAN))
PANEL_WIDTH = 46            # characters in the status card
CLEARANCE_SCALE_M = 2.0     # the bar's full length; for display only
TRAIL_STEP_M = 0.1          # m between points of the path driven
SIZE = (1400, 800)
MAP = pygame.Rect(16, 48, 1368, 496)
CARDS = {"state": (16, 416), "speed": (444, 180), "steer": (636, 180), "togo": (828, 160), "clear": (1000, 180), "slots": (1192, 192)}
CARD_Y, CARD_H = 556, 188


class HmiWindow(RunObserver):
    def __init__(self, cfg: dict, keys: KeyState, rejected_of=lambda: set(), realtime: bool = True, min_clearance: float = 0.30):
        self.cfg, self.keys, self.realtime, self.min_clearance = cfg, keys, realtime, min_clearance
        self.rejected_of = rejected_of
        self.subtitle = ""
        self.units = Units.named(cfg["units"])
        self._result = None      # the finished run and its criteria, so its banner can change units
        self._needs_redraw = False
        self.closed = False
        self.next_action: Optional[str] = None      # after a run: "again", "next" or "quit"
        self.zoomed = False
        self.status = None
        self.sc = None
        self.safety = None
        self.frames_drawn = 0
        self.max_speed_kph, self.max_steer_deg = 15.0, 36.4
        self._banner: Optional[Tuple[str, str, tuple]] = None
        self._suite_mounts = {}
        self._measurements, self._seen = (), 0
        self._ranges: Dict[str, Optional[float]] = {}
        self._paint: Dict[str, np.ndarray] = {}
        self._corners: Dict[int, list] = {}
        self.stall_numbers = StallNumbers()
        self._shown_numbers: Dict[int, int] = {}
        self._wall0: Optional[float] = None
        self._last_draw = 0.0
        self._draw_took = 0.0
        self._trail: list = []
        self._last_tick: Optional[RunTick] = None
        self._view = (0.0, 0.0, 1.0)      # center x, center y, pixels per meter
        self._showing_zoom = None
        self._backdrop: Optional[pygame.Surface] = None
        self._fonts: dict = {}
        self._texts: dict = {}
        pygame.init()
        pygame.key.set_repeat()     # a held key is one press and one release
        flags = pygame.SCALED | pygame.RESIZABLE
        try:
            self.screen = pygame.display.set_mode(SIZE, flags)
        except pygame.error:        # no renderer: still works, without the GPU
            self.screen = pygame.display.set_mode(SIZE, pygame.RESIZABLE)
        pygame.display.set_caption("Self-parking test")

    # ---------- input ----------
    def pump(self) -> None:
        """Take in what the person did since the last call."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._on_close()
            elif event.type == pygame.KEYDOWN:
                self._on_key_press(pygame.key.name(event.key))
            elif event.type == pygame.KEYUP:
                self.keys.release(pygame.key.name(event.key))
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._on_click(event.pos)
            elif event.type == pygame.WINDOWFOCUSLOST:      # a key let go in another window never reports it
                self.keys.focus_lost()

    def _on_key_press(self, key: str) -> None:
        key = (key or "").lower()
        if key == "escape":
            self.next_action = "quit"
            self.keys.request_stop()
        elif key == "z":
            self.zoomed = not self.zoomed
            self._needs_redraw = True
        elif key == "u":
            self.units = self.units.other()
            self._backdrop = None       # the unit labels are on it
            self._needs_redraw = True
            self._write_banner()
        elif key == "r":
            self.next_action = self.next_action or "again"
            self.keys.request_stop()
        elif key == "n":
            self.next_action = self.next_action or "next"
            self.keys.request_stop()
        else:
            self.keys.press(key)

    def _on_click(self, pos) -> None:
        if MAP.collidepoint(pos):
            slot = slot_at(*self._to_world(pos), self._corners)
            if slot is not None:
                self.keys.click_slot(slot)

    def _on_close(self) -> None:
        self.closed = True
        self.next_action = "quit"
        self.keys.request_stop()

    # ---------- meters and pixels ----------
    def _to_screen(self, x, y):
        cx, cy, scale = self._view
        return MAP.centerx + (np.asarray(x) - cx) * scale, MAP.centery - (np.asarray(y) - cy) * scale

    def _to_world(self, pos) -> Tuple[float, float]:
        cx, cy, scale = self._view
        return cx + (pos[0] - MAP.centerx) / scale, cy - (pos[1] - MAP.centery) / scale

    def _points(self, points) -> list:
        points = np.asarray(points, dtype=float)
        x, y = self._to_screen(points[:, 0], points[:, 1])
        return list(zip(x.tolist(), y.tolist()))

    # ---------- text ----------
    def _font(self, size: int, bold: bool) -> pygame.font.Font:
        key = (size, bold)
        if key not in self._fonts:
            self._fonts[key] = pygame.font.SysFont("segoeui,inter,helveticaneue,dejavusans", size, bold=bold)
        return self._fonts[key]

    def _text(self, target, text: str, pos, size: int = 13, color=TEXT, bold: bool = False, anchor: str = "topleft") -> pygame.Rect:
        key = (text, size, color, bold)
        rendered = self._texts.get(key)
        if rendered is None:
            if len(self._texts) > 2000:
                self._texts.clear()
            rendered = self._texts[key] = self._font(size, bold).render(text, True, color)
        return target.blit(rendered, rendered.get_rect(**{anchor: pos}))

    # ---------- what doesn't change during a view ----------
    def _build_backdrop(self) -> None:
        surface = pygame.Surface(SIZE)
        surface.fill(BACKGROUND)
        self._text(surface, "Self-parking test", (16, 24), 20, TEXT, True, "midleft")
        self._text(surface, self.subtitle, (196, 24), 13, MUTED, False, "midleft")
        x = MAP.right
        for label, color in reversed(LEGEND):
            width = self._font(12, False).size(label)[0]
            x -= width
            self._text(surface, label, (x, 24), 12, MUTED, False, "midleft")
            x -= 15
            pygame.draw.rect(surface, color, (x, 19, 10, 10), border_radius=2)
            x -= 16
        pygame.draw.rect(surface, ASPHALT, MAP, border_radius=8)
        if not self.zoomed:
            surface.set_clip(MAP.inflate(-2, -2))
            self._draw_lot(surface)
            surface.set_clip(None)
        pygame.draw.rect(surface, BORDER, MAP, width=1, border_radius=8)
        self._draw_cards(surface)
        self._draw_keys(surface)
        self._backdrop = surface.convert()

    def _draw_lot(self, surface) -> None:
        sc = self.sc
        for a, b in sc.lines:
            pygame.draw.aaline(surface, PAINT, *self._points([a, b]))
        for ob in sc.curbs:
            pygame.draw.polygon(surface, CURB, self._points(ob))
        for ob in sc.parked_cars:
            points = self._points(ob)
            pygame.draw.polygon(surface, PARKED, points)
            pygame.draw.aalines(surface, PARKED_EDGE, True, points)
        for lot_object in sc.lot_objects:
            points = self._points(lot_object.polygon)
            pygame.draw.polygon(surface, AMBER, points)
            pygame.draw.aalines(surface, TEXT, True, points)

    def _draw_cards(self, surface) -> None:
        for x, w in CARDS.values():
            pygame.draw.rect(surface, SURFACE, (x, CARD_Y, w, CARD_H), border_radius=10)
            pygame.draw.rect(surface, BORDER, (x, CARD_Y, w, CARD_H), width=1, border_radius=10)
        u = self.units
        for key, label, unit in (("speed", "SPEED", u.speed_unit), ("steer", "STEERING", "deg"), ("togo", "TO GO", u.long_unit), ("clear", "CLEARANCE", u.long_unit)):
            x, w = CARDS[key]
            self._text(surface, label, (x + 16, CARD_Y + 14), 11, MUTED, True)
            self._text(surface, unit, (x + w - 16, CARD_Y + 92), 12, MUTED, False, "bottomright")
            if key != "togo":
                self._bar_track(surface, x, w)
        x, w = CARDS["slots"]
        self._text(surface, "STALLS", (x + 16, CARD_Y + 14), 11, MUTED, True)
        for k, (label, color) in enumerate((("Selected", GREEN), ("Offered", BLUE), ("No path", RED))):
            self._text(surface, label, (x + 16, CARD_Y + 58 + k * 28), 13, color, False, "midleft")
        x, w = CARDS["steer"]
        center = x + w // 2
        pygame.draw.line(surface, MUTED, (center, CARD_Y + CARD_H - 34), (center, CARD_Y + CARD_H - 18))

    @staticmethod
    def _bar_track(surface, x: int, w: int) -> None:
        pygame.draw.rect(surface, BORDER, (x + 16, CARD_Y + CARD_H - 30, w - 32, 8), border_radius=4)

    def _draw_keys(self, surface) -> None:
        x, y = 16, 756
        for key, label in KEYS:
            w = max(self._font(11, False).size(key)[0] + 16, 28)
            pygame.draw.rect(surface, SURFACE, (x, y, w, 24), border_radius=5)
            pygame.draw.rect(surface, BORDER, (x, y, w, 24), width=1, border_radius=5)
            self._text(surface, key, (x + w / 2, y + 12), 11, TEXT, False, "center")
            x += w + 7
            x = self._text(surface, label, (x, y + 12), 12, MUTED, False, "midleft").right + 18

    # ---------- a run ----------
    def on_run_start(self, sc, suite, measurements, safety, map_frames, search_flow) -> None:
        self.sc = sc
        self.safety = safety
        self.next_action = None
        self._banner, self._result = None, None
        self._measurements, self._seen = measurements, 0
        self._ranges, self._paint, self._corners = {}, {}, {}
        self.stall_numbers, self._shown_numbers = StallNumbers(), {}
        self._trail = []
        self._suite_mounts = {s.name: s for s in suite.sensors} if suite is not None else {}
        self._wall0 = None
        self.max_speed_kph = self.cfg["max_speed_kph"]
        self.max_steer_deg = math.degrees(sc.vehicle.max_steer)
        self._showing_zoom = None
        self._backdrop = None

    def on_tick(self, tick: RunTick) -> None:
        if self.closed:
            return
        if not self.realtime:
            self._take_in(tick)
            self._redraw(tick)
            self.pump()
            return
        now = time.perf_counter()
        if self._wall0 is None or now - (self._wall0 + tick.t) > self.cfg["resync_after_s"]:
            self._wall0 = now - tick.t      # a plan took a while, say: carry on from here rather than race
        wait = self._wall0 + tick.t - now
        if wait > 0.0:
            time.sleep(wait)
        self._take_in(tick)
        started = time.perf_counter()
        if started - self._last_draw >= max(1.0 / self.cfg["redraw_hz"], self.cfg["draw_gap_factor"] * self._draw_took) or tick.final:
            self._redraw(tick)      # with a gap after each frame, so the drawing can't hold the sim back
            self._last_draw = started
            self._draw_took = time.perf_counter() - started
        self.pump()

    def _take_in(self, tick: RunTick) -> None:
        for m in self._measurements[self._seen:]:
            if isinstance(m, UltrasonicMeasurement):
                self._ranges[m.sensor] = m.distance
            elif isinstance(m, LineCameraMeasurement):
                self._paint[m.sensor] = m.segments
        self._seen = len(self._measurements)
        self._corners = slots_in_map(tick)
        self._shown_numbers = self.stall_numbers.update(tick.slots, slot_statuses(tick.slots, selected_now(tick), self.rejected_of()))
        if str(tick.row["sup_state"][0]) not in ENTER_STAYS_ON_IN:
            self.keys.clear_confirm()
        self._last_tick = tick

    def _frame_view(self, s) -> None:
        """The whole lot, or a zoomed view that follows the car. Zoomed, the lot is drawn each frame instead of being part of the backdrop."""
        half = self.cfg["zoom_half_width_m"]
        if self.zoomed:
            self._view = (s.x, s.y, MAP.width / (2.0 * half))
            if self._showing_zoom is not True:
                self._backdrop = None
        elif self._showing_zoom is not False:
            b = self.sc.bounds
            scale = min(MAP.width / (b[1] - b[0]), MAP.height / (b[3] - b[2]))
            self._view = ((b[0] + b[1]) / 2.0, (b[2] + b[3]) / 2.0, scale)
            self._backdrop = None
        self._showing_zoom = self.zoomed

    def _redraw(self, tick: RunTick) -> None:
        s = tick.true_pose
        self._frame_view(s)
        if self._backdrop is None:
            self._build_backdrop()
        if not self._trail or math.hypot(s.x - self._trail[-1][0], s.y - self._trail[-1][1]) >= TRAIL_STEP_M:
            self._trail.append((s.x, s.y))
        self.status = status_from_tick(tick, self.rejected_of())
        clearance = self.sc.clearance_exact(s.x, s.y, s.yaw)
        screen = self.screen
        screen.blit(self._backdrop, (0, 0))
        screen.set_clip(MAP.inflate(-2, -2))
        if self.zoomed:
            self._draw_lot(screen)
        self._draw_slots()
        if len(self._trail) >= 2:
            pygame.draw.aalines(screen, TEAL, False, self._points(self._trail))
        self._draw_plan(tick)
        self._draw_sensors(s)
        self._draw_actors()
        self._draw_car(s)
        self._draw_plan_failure(tick)
        self._draw_banner()
        screen.set_clip(None)
        self._draw_status(self.status, clearance)
        pygame.display.flip()
        self.frames_drawn += 1

    def _draw_slots(self) -> None:
        statuses = self.status.slots
        scale = self._view[2]
        for slot_id, corners in self._corners.items():
            color = SLOT_COLORS[statuses[slot_id]]
            points = self._points(corners)
            gfxdraw.filled_polygon(self.screen, points, (*color, 70))
            gfxdraw.aapolygon(self.screen, points, (*color, 255))
            number = self._shown_numbers.get(slot_id)
            if number is not None:      # only stalls available right now carry a number
                cx, cy = self._to_screen(*np.mean(corners, axis=0))
                self._text(self.screen, str(number), (float(cx), float(cy)), 13 if scale < 25 else 16, TEXT, True, "center")

    def _draw_plan(self, tick: RunTick) -> None:
        if tick.path is None:
            return
        row = tick.row
        x, y = MapToOdom(tick.true_pose, _Pose(row["est_x"][0], row["est_y"][0], row["est_yaw"][0])).to_map(tick.path.x, tick.path.y)
        pygame.draw.lines(self.screen, PURPLE, False, self._points(np.column_stack([x, y])), 2)

    def _draw_sensors(self, s) -> None:
        c, si = math.cos(s.yaw), math.sin(s.yaw)
        for name, distance in self._ranges.items():
            sensor = self._suite_mounts.get(name)
            if distance is None or not isinstance(sensor, UltrasonicSensor):
                continue
            ox, oy = s.x + c * sensor.mount.x - si * sensor.mount.y, s.y + si * sensor.mount.x + c * sensor.mount.y
            half = math.radians(sensor.cone_deg) / 2.0
            angles = s.yaw + sensor.mount.yaw + np.linspace(-half, half, 9)
            points = self._points(np.column_stack([ox + distance * np.cos(angles), oy + distance * np.sin(angles)]))
            pygame.draw.aalines(self.screen, _dim(AMBER, 0.6), False, points)
        for segments in self._paint.values():
            for (ax, ay), (bx, by) in np.asarray(segments).reshape(-1, 2, 2):
                a, b = self._points([(s.x + c * ax - si * ay, s.y + si * ax + c * ay), (s.x + c * bx - si * by, s.y + si * bx + c * by)])
                pygame.draw.line(self.screen, CYAN, a, b, 2)

    def _draw_actors(self) -> None:
        frames = self.safety.actor_frames if self.safety is not None else ()
        if frames:
            for _, _, polygon in frames[-1][1]:
                pygame.draw.polygon(self.screen, PINK, self._points(polygon), 2)

    def _draw_car(self, s) -> None:
        v = self.sc.vehicle
        points = self._points(v.footprint(s.x, s.y, s.yaw))
        gfxdraw.filled_polygon(self.screen, points, BLUE)
        pygame.draw.aalines(self.screen, TEXT, True, points)
        along, half = v.wheelbase * 0.72, v.width / 2.0 - 0.18    # the windshield, to show which way it faces
        c, si = math.cos(s.yaw), math.sin(s.yaw)
        a, b = self._points([(s.x + c * along + si * half, s.y + si * along - c * half), (s.x + c * along - si * half, s.y + si * along + c * half)])
        pygame.draw.line(self.screen, BACKGROUND, a, b, 3)

    def _draw_plan_failure(self, tick: RunTick) -> None:
        """Why the last stall tried has no path, at the top of the lot until a stall is picked or a plan works."""
        failed = tick.failed_plan
        if failed is None or self._banner is not None or str(tick.row["sup_state"][0]) not in SHOW_PLAN_FAILURE_IN or failed.slot_id not in self.rejected_of():
            return      # the last line: the car has moved on, so the stall can be tried again and the message no longer holds
        number = self.stall_numbers.number_of(failed.slot_id)
        headline = f"Can't park in stall {number}" if number is not None else "Can't park in that stall"
        if not math.isnan(failed.map_center[0]) and getattr(tick, "true_pose", None) is not None:
            away = math.hypot(failed.map_center[0] - tick.true_pose.x, failed.map_center[1] - tick.true_pose.y)
            headline += f", {self.units.long(away):.1f} {self.units.long_unit} away"
        lines = _wrap(plan_failure_text(failed.why), 88)
        box = pygame.Rect(0, 0, 720, 44 + 19 * len(lines))
        box.midtop = (MAP.centerx, MAP.y + 12)
        pygame.draw.rect(self.screen, SURFACE, box, border_radius=10)
        pygame.draw.rect(self.screen, RED, box, width=2, border_radius=10)
        self._text(self.screen, headline, (box.centerx, box.y + 18), 15, RED, True, "center")
        for k, line in enumerate(lines):
            self._text(self.screen, line, (box.centerx, box.y + 37 + k * 19), 13, TEXT, False, "midtop")

    def _draw_banner(self) -> None:
        if self._banner is None:
            return
        headline, detail, color = self._banner
        lines = _wrap(detail, 72)
        height = 72 + 22 * len(lines)
        box = pygame.Rect(0, 0, 620, height)
        box.midbottom = (MAP.centerx, MAP.bottom - 16)
        pygame.draw.rect(self.screen, SURFACE, box, border_radius=12)
        pygame.draw.rect(self.screen, color, box, width=2, border_radius=12)
        self._text(self.screen, headline, (box.centerx, box.y + 22), 18, TEXT, True, "center")
        for k, line in enumerate(lines):
            self._text(self.screen, line, (box.centerx, box.y + 48 + k * 20), 13, TEXT, False, "center")
        self._text(self.screen, "R again   ·   N next seed   ·   Esc quit", (box.centerx, box.bottom - 16), 12, MUTED, False, "center")

    def _draw_status(self, st, clearance: float) -> None:
        screen = self.screen
        x, w = CARDS["state"]
        color = STATE_COLORS.get(st.state, MUTED)
        pygame.draw.rect(screen, color, (x + 16, CARD_Y + 14, 210, 32), border_radius=16)
        self._text(screen, st.state.replace("_", " "), (x + 121, CARD_Y + 30), 14, BACKGROUND, True, "center")
        self._text(screen, f"{st.t:.1f} s", (x + w - 16, CARD_Y + 30), 13, MUTED, False, "midright")
        u = self.units
        message = u.message(st.message or "—").replace("Hold confirm", "Press Enter").replace("Press confirm", "Press Enter")     # the supervisor doesn't know the keyboard
        for k, line in enumerate(_wrap(message, PANEL_WIDTH)[:2]):
            self._text(screen, line, (x + 16, CARD_Y + 60 + k * 21), 15)
        reason = u.message(self.stall_numbers.in_words(st.last_reason)) or "—"
        if st.abort_reason and st.abort_reason != "none":
            reason = f"{st.abort_reason}: {reason}"
        lines = _wrap(f"Last change: {reason}", PANEL_WIDTH + 12)[:2]
        for k, line in enumerate(lines):
            self._text(screen, line, (x + 16, CARD_Y + CARD_H - 14 - (len(lines) - 1 - k) * 17), 12, MUTED, False, "bottomleft")

        reverse = st.gear == "R"
        self._number("speed", f"{u.speed(st.speed_kph):.1f}" if u.imperial else f"{u.speed(st.speed_kph):.2f}")
        x, w = CARDS["speed"]
        self._text(screen, st.gear, (x + w - 16, CARD_Y + 14), 13, AMBER if reverse else TEXT, True, "topright")
        self._bar("speed", 0.0, min(st.speed_kph / self.max_speed_kph, 1.0), AMBER if reverse else BLUE)
        self._number("steer", f"{st.steering_deg:+.0f}")
        share = max(-1.0, min(st.steering_deg / self.max_steer_deg, 1.0))
        self._bar("steer", 0.5 + min(share, 0.0) / 2.0, abs(share) / 2.0, AMBER if abs(share) > 0.9 else BLUE)
        self._number("togo", "—" if st.distance_to_go_m is None else f"{u.long(st.distance_to_go_m):.1f}")
        low = clearance < self.min_clearance
        self._number("clear", f"{u.long(clearance):.1f}" if u.imperial else f"{clearance:.2f}", RED if low else TEXT)
        self._bar("clear", 0.0, min(clearance / CLEARANCE_SCALE_M, 1.0), RED if low else GREEN)
        x, w = CARDS["clear"]
        tick = x + 16 + int((w - 32) * self.min_clearance / CLEARANCE_SCALE_M)
        pygame.draw.line(screen, TEXT, (tick, CARD_Y + CARD_H - 34), (tick, CARD_Y + CARD_H - 18))

        x, w = CARDS["slots"]
        selected = self._shown_numbers.get(st.selected_slot)
        counts = ("—" if selected is None else str(selected), str(sum(v == OFFERED for v in st.slots.values())), str(sum(v == REJECTED for v in st.slots.values())))
        for k, value in enumerate(counts):
            self._text(screen, value, (x + w - 16, CARD_Y + 58 + k * 28), 15, TEXT, True, "midright")
        self._text(screen, f"Estimate off by {u.long(st.estimate_off_m):.2f} {u.long_unit}", (x + 16, CARD_Y + CARD_H - 14), 11, MUTED, False, "bottomleft")

    def _number(self, card: str, text: str, color=TEXT) -> None:
        x, _ = CARDS[card]
        self._text(self.screen, text, (x + 16, CARD_Y + 96), 40, color, True, "bottomleft")

    def _bar(self, card: str, start: float, share: float, color) -> None:
        x, w = CARDS[card]
        length = w - 32
        if share > 0.0:
            pygame.draw.rect(self.screen, color, (x + 16 + start * length, CARD_Y + CARD_H - 30, max(share * length, 2.0), 8), border_radius=4)

    # ---------- after a run ----------
    def show_result(self, res, criteria: dict) -> None:
        self._result = (res, criteria)
        self._write_banner()
        if self._last_tick is not None and not self.closed:
            self._redraw(self._last_tick)
        self.pump()

    def _write_banner(self) -> None:
        if self._result is None:
            return
        res, criteria = self._result
        summary, explanation = summary_text(res, self.units), explain_result(res, criteria, self.units)
        self._banner = ("Parked" if res.success else "Not parked", f"{explanation}  {summary}".strip(), GREEN if res.success else RED)

    def wait_for_next(self, poll_s: float = 0.03) -> str:
        """Keep the result on screen until the person says what's next."""
        while self.next_action is None and not self.closed:
            self.pump()
            if self._last_tick is not None and self._needs_redraw and not self.closed:
                self._needs_redraw = False
                self._redraw(self._last_tick)
            time.sleep(poll_s)
        return self.next_action or "quit"

    def close(self) -> None:
        pygame.quit()


class _Pose:
    def __init__(self, x, y, yaw):
        self.x, self.y, self.yaw = float(x), float(y), float(yaw)


def _dim(color, share: float):
    return tuple(int(c * share + b * (1.0 - share)) for c, b in zip(color, ASPHALT))


def _wrap(text: str, width: int) -> list:
    out, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return out + ([line] if line else [])
