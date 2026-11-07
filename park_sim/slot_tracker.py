"""Slot tracks in the odometry frame: corners with a covariance, updated by a simple Kalman filter.

A track is confirmed after enough consistent detections and then keeps its slot id for good, so a slot seen again gets its old number.
Unconfirmed tracks that stop getting detections are dropped. Whether a confirmed one is still offered is up to the detector.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np


def corner_covariance(along: np.ndarray, sigma_along: float, sigma_across: float) -> np.ndarray:
    """2 x 2 covariance with sigma_along in the unit direction `along` and sigma_across square to it."""
    across = np.array([-along[1], along[0]])
    return sigma_along ** 2 * np.outer(along, along) + sigma_across ** 2 * np.outer(across, across)


def slot_heading(corners: np.ndarray) -> float:
    """Direction from the opening (first two corners) to the rear (last two), rad."""
    d = corners[2:].mean(axis=0) - corners[:2].mean(axis=0)
    return math.atan2(d[1], d[0])


@dataclass
class SlotDetection:
    corners: np.ndarray     # (4, 2): the opening first, in order along the direction of travel
    cov: np.ndarray         # (4, 2, 2)
    source: str             # "lines" or "space"


@dataclass
class SlotTrack:
    track_id: int
    corners: np.ndarray
    cov: np.ndarray
    source: str
    first_seen_t: float
    last_update_t: float
    hits: int = 1
    slot_id: Optional[int] = None       # set when confirmed, then never changed
    confirmed_t: Optional[float] = None
    passed: bool = False
    readings: int = 0                   # occupancy evidence
    returns_inside: int = 0
    free: Optional[bool] = None         # what that evidence last said; None until there is enough of it

    @property
    def confirmed(self) -> bool:
        return self.slot_id is not None

    @property
    def center(self) -> np.ndarray:
        return self.corners.mean(axis=0)


@dataclass
class SlotTracker:
    cfg: dict
    tracks: List[SlotTrack] = field(default_factory=list)
    _next_track: int = 0
    _next_slot: int = 0

    def step(self, t: float, detections: Sequence[SlotDetection], allow_new: bool = True) -> None:
        """Associate this fit's detections, update the tracks they match, start new ones, drop stale ones."""
        for det in detections:
            track = self._match(det)
            if track is not None:
                self._apply(track, det, t)
            elif allow_new:
                self.tracks.append(SlotTrack(self._next_track, det.corners.copy(), det.cov.copy(), det.source, t, t))
                self._next_track += 1
        self.tracks = [tr for tr in self.tracks
                       if tr.confirmed or t - tr.last_update_t <= self.cfg["unconfirmed_timeout_s"]]

    def _match(self, det: SlotDetection) -> Optional[SlotTrack]:
        center, heading = det.corners.mean(axis=0), slot_heading(det.corners)
        best, best_distance = None, self.cfg["gate_distance"]
        for tr in self.tracks:
            distance = float(np.hypot(*(tr.center - center)))
            turn = abs((slot_heading(tr.corners) - heading + math.pi) % (2.0 * math.pi) - math.pi)
            if distance <= best_distance and turn <= math.radians(self.cfg["gate_heading_deg"]):
                best, best_distance = tr, distance
        return best

    def _apply(self, track: SlotTrack, det: SlotDetection, t: float) -> None:
        if det.source == "space" and track.source == "lines":
            return      # lines first: the coarser gap estimate adds nothing to a slot the lines found
        if det.source == "lines" and track.source == "space":
            track.corners, track.cov, track.source = det.corners.copy(), det.cov.copy(), "lines"
        else:
            self.predict(track, t)
            for i in range(4):
                self.update_corner(track, i, det.corners[i], det.cov[i])
        track.last_update_t = t
        track.hits += 1
        if not track.confirmed and track.hits >= self.cfg["confirm_detections"]:
            track.slot_id, track.confirmed_t = self._next_slot, t
            self._next_slot += 1

    def predict(self, track: SlotTrack, t: float) -> None:
        dt = max(t - track.last_update_t, 0.0)
        track.cov = track.cov + np.eye(2) * self.cfg["process_noise"] ** 2 * dt
        track.last_update_t = t

    @staticmethod
    def update_along(track: SlotTrack, i: int, axis: np.ndarray, value: float, sigma: float) -> None:
        """Update corner i from a measurement of its position along one unit axis only."""
        p_axis = track.cov[i] @ axis
        gain = p_axis / (axis @ p_axis + sigma ** 2)
        track.corners[i] = track.corners[i] + gain * (value - track.corners[i] @ axis)
        track.cov[i] = track.cov[i] - np.outer(gain, p_axis)

    @staticmethod
    def update_corner(track: SlotTrack, i: int, z: np.ndarray, r: np.ndarray) -> None:
        p = track.cov[i]
        gain = p @ np.linalg.inv(p + r)
        track.corners[i] = track.corners[i] + gain @ (z - track.corners[i])
        track.cov[i] = (np.eye(2) - gain) @ p
