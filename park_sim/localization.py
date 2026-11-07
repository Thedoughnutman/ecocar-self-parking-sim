"""Where the car thinks it is: one interface, several estimators. See docs/localization.md.

Every estimator has update(measurements, dt) -> PoseEstimate (pose + 3x3 covariance over x, y, yaw).

  truth     ground truth, no error
  white     another estimator's pose plus independent noise whenever the controller reads it
  odometry  dead reckoning from wheel speeds and yaw rate; frame "odom", drifts from the start pose
  gnss_ins  approximates OxTS output: map frame, slowly wandering error, occasional degraded periods

`frame` says what an estimate's coordinates mean. "map": the true lot's frame, up to the estimate's
error. "odom": a frame that starts on the true start pose and drifts from it. Measurements carry the
true pose because truth, white and gnss_ins are error models around it; odometry never reads it.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

ESTIMATORS = ("auto", "truth", "white", "odometry", "gnss_ins")
ESTIMATION_METRICS = ("est_err_final_m", "est_err_max_m", "est_yaw_err_final_deg", "est_yaw_err_max_deg", "est_drift_final_m")
_CLOCK_EPS = 1e-9
_STOPPED_SPEED = 1e-3   # m/s, wheel speed that counts as stopped when quantization is off
_NO_COVARIANCE = np.zeros((3, 3))
_NO_COVARIANCE.setflags(write=False)


@dataclass(frozen=True, eq=False)
class PoseEstimate:
    x: float
    y: float
    yaw: float
    cov: np.ndarray = field(default_factory=lambda: _NO_COVARIANCE)     # 3x3 over x, y, yaw (m^2, rad^2)

    @property
    def sigma_xy(self) -> float:
        """1-sigma position error, m (root of the mean of the x and y variances)."""
        return math.sqrt((self.cov[0, 0] + self.cov[1, 1]) / 2.0)

    @property
    def sigma_yaw(self) -> float:
        return math.sqrt(self.cov[2, 2])


@dataclass(frozen=True)
class Measurements:
    """One tick of inputs. x, y, yaw are the true pose, for the error models only."""
    x: float
    y: float
    yaw: float
    speed: float                                        # m/s, rear-axle center, signed
    yaw_rate: float                                     # rad/s
    wheels: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)   # m/s, signed: FL, FR, RL, RR

    @classmethod
    def from_state(cls, state, wheelbase: float, track: float) -> "Measurements":
        yaw_rate = state.v * math.tan(state.delta) / wheelbase
        return cls(state.x, state.y, state.yaw, state.v, yaw_rate, wheel_speeds(state.v, yaw_rate, wheelbase, track))


def wheel_speeds(v: float, yaw_rate: float, wheelbase: float, track: float) -> Tuple[float, float, float, float]:
    """Each wheel's signed speed for a rigid car with no slip: FL, FR, RL, RR. Rear wheels move along
    the heading; front wheels also carry the sideways speed at the front axle."""
    left, right = v - yaw_rate * track / 2.0, v + yaw_rate * track / 2.0
    side = yaw_rate * wheelbase
    return (math.copysign(math.hypot(left, side), v), math.copysign(math.hypot(right, side), v), left, right)


class MapToOdom:
    """The rigid transform from the map frame to the odometry frame, at one instant."""

    def __init__(self, true_pose, estimated_pose):
        self.true, self.est = true_pose, estimated_pose
        self.dyaw = estimated_pose.yaw - true_pose.yaw
        self._c, self._s = math.cos(self.dyaw), math.sin(self.dyaw)

    def to_odom(self, x, y):
        dx, dy = x - self.true.x, y - self.true.y
        return self.est.x + self._c * dx - self._s * dy, self.est.y + self._s * dx + self._c * dy

    def to_map(self, x, y):
        dx, dy = x - self.est.x, y - self.est.y
        return self.true.x + self._c * dx + self._s * dy, self.true.y - self._s * dx + self._c * dy


class Localizer:
    """The interface. `estimate` is the latest output, starting on the true start pose."""
    name = ""
    frame = "map"

    def __init__(self, start) -> None:
        self.estimate = PoseEstimate(start.x, start.y, start.yaw)
        self._clock = 0.0

    def update(self, measurements: Measurements, dt: float) -> PoseEstimate:
        raise NotImplementedError

    def for_control(self) -> PoseEstimate:
        """The pose the controller reads this tick: the latest estimate, unless noise is added on reading."""
        return self.estimate

    @property
    def sample_time(self) -> float:
        """s on this estimator's own clock (0 at the start pose): when `estimate` was measured."""
        return self._clock

    def set_latency(self, latency_s: float) -> None:
        """Change the output delay from now on, for injecting a delayed pose."""
        raise ValueError(f"the {self.name} estimator has no latency; use odometry or gnss_ins")


class TruthLocalizer(Localizer):
    name = "truth"

    def update(self, measurements: Measurements, dt: float) -> PoseEstimate:
        self._clock += dt
        self.estimate = PoseEstimate(measurements.x, measurements.y, measurements.yaw)
        return self.estimate


class WhiteNoiseLocalizer(Localizer):
    """Noise on the pose the controller reads, around another estimator's pose. Detectors and frame
    conversions see that estimator's pose without the noise. Noise is drawn per controller reading,
    so the old --noise-xy / --noise-yaw results reproduce number for number."""
    name = "white"

    def __init__(self, inner: Localizer, rng: np.random.Generator, noise_xy: float, noise_yaw_deg: float):
        self.inner, self._rng, self._xy, self._yaw_deg = inner, rng, noise_xy, noise_yaw_deg
        self.frame = inner.frame
        self._extra = np.diag([noise_xy ** 2, noise_xy ** 2, math.radians(noise_yaw_deg) ** 2])
        self.estimate = self._with_cov(inner.estimate)

    def _with_cov(self, e: PoseEstimate) -> PoseEstimate:
        return PoseEstimate(e.x, e.y, e.yaw, e.cov + self._extra)

    def update(self, measurements: Measurements, dt: float) -> PoseEstimate:
        self.estimate = self._with_cov(self.inner.update(measurements, dt))
        return self.estimate

    def for_control(self) -> PoseEstimate:
        e = self.estimate
        self.estimate = PoseEstimate(
            e.x + self._rng.normal(0, self._xy) if self._xy else e.x,
            e.y + self._rng.normal(0, self._xy) if self._xy else e.y,
            e.yaw + math.radians(self._rng.normal(0, self._yaw_deg)) if self._yaw_deg else e.yaw,
            e.cov)
        return self.estimate

    @property
    def sample_time(self) -> float:
        return self.inner.sample_time

    def set_latency(self, latency_s: float) -> None:
        self.inner.set_latency(latency_s)


class _SampledLocalizer(Localizer):
    """A sensor with an update rate and a delay: samples at rate_hz, and outputs the newest sample
    at least latency_s old. Between samples the output holds."""

    def __init__(self, start, rate_hz: float, latency_s: float):
        super().__init__(start)
        self._period = 1.0 / rate_hz
        self._latency = latency_s
        self._clock = self._last_sample = 0.0
        self._next_sample = self._period
        self._samples = deque([(0.0, self.estimate)])

    def _sample(self, m: Measurements, period: float) -> PoseEstimate:
        raise NotImplementedError

    @property
    def sample_time(self) -> float:
        return self._samples[0][0]

    def set_latency(self, latency_s: float) -> None:
        self._latency = latency_s

    def update(self, measurements: Measurements, dt: float) -> PoseEstimate:
        self._clock += dt
        if self._clock >= self._next_sample - _CLOCK_EPS:
            self._samples.append((self._clock, self._sample(measurements, self._clock - self._last_sample)))
            self._last_sample = self._clock
            while self._next_sample <= self._clock + _CLOCK_EPS:
                self._next_sample += self._period
        due = self._clock - self._latency + _CLOCK_EPS
        while len(self._samples) > 1 and self._samples[1][0] <= due:
            self._samples.popleft()
        self.estimate = self._samples[0][1]
        return self.estimate


def _quantize(value: float, step: float) -> float:
    return round(value / step) * step if step > 0.0 else value


class WheelOdometry(_SampledLocalizer):
    """Dead reckoning. Scale error and gyro bias have a fixed size and a sign drawn once per run;
    noise is new every sample. The covariance is a conservative sum of the configured errors."""
    name = "odometry"
    frame = "odom"

    def __init__(self, start, cfg: dict, rng: np.random.Generator, wheelbase: float = 0.0, track: float = 0.0,
                 latency_s: float = 0.0):
        super().__init__(start, cfg["rate_hz"], latency_s)
        self._x, self._y, self._yaw = start.x, start.y, start.yaw
        self._rng = rng
        self._cfg = cfg
        self._wheelbase = wheelbase
        self._four_wheels = cfg["wheel_source"] == "four_wheel"
        if cfg["wheel_source"] not in ("rear_average", "four_wheel"):
            raise ValueError(f"unknown wheel_source {cfg['wheel_source']!r}")
        self._scale_size = cfg["wheel_scale_error"]
        self._bias_size = math.radians(cfg["yaw_rate_bias_deg_s"])
        self._scale = 1.0 + self._scale_size * float(rng.choice((-1.0, 1.0)))
        self._bias = self._bias_size * float(rng.choice((-1.0, 1.0)))
        self._wheel_noise = cfg["wheel_noise_std"]
        self._wheel_step = cfg["wheel_quantization"]
        self._yaw_noise = math.radians(cfg["yaw_rate_noise_std_deg_s"])
        self._yaw_step = math.radians(cfg["yaw_rate_quantization_deg_s"])
        self._pos_sigma = self._yaw_sigma = self._yaw_noise_var = 0.0
        self._calibrate = cfg["standstill_calibration"]
        self._rest_needed = cfg["standstill_min_time_s"]
        self._rest_time = self._rest_integral = 0.0

    def _wheel(self, speed: float) -> float:
        """A stopped wheel gives no pulses, so no noise either."""
        v = speed * self._scale
        if self._wheel_noise > 0.0 and abs(speed) > self._stopped_below:
            v += self._rng.normal(0.0, self._wheel_noise)
        return _quantize(v, self._wheel_step)

    @property
    def _stopped_below(self) -> float:
        return max(self._wheel_step, _STOPPED_SPEED)

    def _gyro(self, yaw_rate: float, stopped: bool, period: float) -> float:
        """The yaw rate to integrate. With standstill_calibration, a car whose wheels are all stopped
        isn't turning: its gyro reading is the bias, and later readings have that average removed."""
        w = yaw_rate + self._bias
        if self._yaw_noise > 0.0:
            w += self._rng.normal(0.0, self._yaw_noise)
        w = _quantize(w, self._yaw_step)
        if not self._calibrate:
            return w
        if stopped:
            self._rest_time += period
            self._rest_integral += w * period
            return 0.0
        return w - self._rest_integral / self._rest_time if self._rest_time >= self._rest_needed else w

    def _bias_uncertainty(self) -> float:
        """rad/s: the configured size, or what's left of it after averaging at standstill."""
        if self._calibrate and self._rest_time >= self._rest_needed:
            samples = self._rest_time * self._cfg["rate_hz"]
            return self._yaw_noise / math.sqrt(samples) + self._yaw_step
        return self._bias_size

    def _sample(self, m: Measurements, period: float) -> PoseEstimate:
        fl, fr, rl, rr = (self._wheel(s) for s in m.wheels)
        stopped = all(abs(s) < self._stopped_below for s in (fl, fr, rl, rr))
        w = self._gyro(m.yaw_rate, stopped, period)
        v = (rl + rr) / 2.0
        if self._four_wheels:
            # each front wheel also moves sideways at the front axle; keep the part along the heading
            side2 = (w * self._wheelbase) ** 2
            front = sum(math.copysign(math.sqrt(max(s * s - side2, 0.0)), s) for s in (fl, fr)) / 2.0
            v = (v + front) / 2.0

        half = w * period / 2.0
        chord = v * period * (math.sin(half) / half if abs(half) > 1e-9 else 1.0)   # exact for a constant arc
        self._x += chord * math.cos(self._yaw + half)
        self._y += chord * math.sin(self._yaw + half)
        self._yaw = (self._yaw + w * period + math.pi) % (2.0 * math.pi) - math.pi

        if not (self._calibrate and stopped):
            self._yaw_sigma += self._bias_uncertainty() * period
        self._yaw_noise_var += (self._yaw_noise * period) ** 2
        yaw_sigma = self._yaw_sigma + math.sqrt(self._yaw_noise_var)
        self._pos_sigma += (self._scale_size + yaw_sigma) * abs(v * period)
        cov = np.diag([self._pos_sigma ** 2, self._pos_sigma ** 2, yaw_sigma ** 2])
        return PoseEstimate(self._x, self._y, self._yaw, cov)


class GnssInsLocalizer(_SampledLocalizer):
    """Approximates OxTS output in the map frame: true pose plus a first-order Gauss-Markov error.
    In a degraded period an extra error grows in a direction drawn at its start, then decays."""
    name = "gnss_ins"

    def __init__(self, start, cfg: dict, rng: np.random.Generator, latency_s: float = 0.0):
        super().__init__(start, cfg["rate_hz"], latency_s)
        self._rng, self._cfg = rng, cfg
        self._pos_sigma, self._pos_tau = cfg["position_sigma_m"], cfg["position_tau_s"]
        self._yaw_sigma, self._yaw_tau = math.radians(cfg["heading_sigma_deg"]), cfg["heading_tau_s"]
        d = cfg["degraded"]
        self._pos_growth, self._yaw_growth = d["position_growth_m_s"], math.radians(d["heading_growth_deg_s"])
        self._error = np.array([rng.normal(0.0, self._pos_sigma), rng.normal(0.0, self._pos_sigma),
                                rng.normal(0.0, self._yaw_sigma)])
        self._drift = np.zeros(3)
        self._drift_rate = np.zeros(3)
        self._drift_sigma = np.zeros(2)     # how large the drift could be: position, yaw
        self._random_until: Optional[float] = None
        self.degraded = False
        self.estimate = self._apply(start.x, start.y, start.yaw)
        self._samples[0] = (0.0, self.estimate)

    def _apply(self, x: float, y: float, yaw: float) -> PoseEstimate:
        ex, ey, eyaw = self._error + self._drift
        pos_var = self._pos_sigma ** 2 + self._drift_sigma[0] ** 2
        yaw_var = self._yaw_sigma ** 2 + self._drift_sigma[1] ** 2
        return PoseEstimate(x + float(ex), y + float(ey), yaw + float(eyaw), np.diag([pos_var, pos_var, yaw_var]))

    def _is_degraded(self, period: float) -> bool:
        d = self._cfg["degraded"]
        if any(a <= self._clock < b for a, b in d["windows"]):
            return True
        if self._random_until is not None and self._clock < self._random_until:
            return True
        self._random_until = None
        if d["rate_per_s"] > 0.0 and self._rng.random() < d["rate_per_s"] * period:
            self._random_until = self._clock + d["duration_s"]
            return True
        return False

    def _sample(self, m: Measurements, period: float) -> PoseEstimate:
        phi = np.exp(-period / np.array([self._pos_tau, self._pos_tau, self._yaw_tau]))
        sigma = np.array([self._pos_sigma, self._pos_sigma, self._yaw_sigma])
        self._error = phi * self._error + sigma * np.sqrt(1.0 - phi ** 2) * self._rng.normal(size=3)

        degraded = self._is_degraded(period)
        if degraded and not self.degraded:
            a = self._rng.uniform(-math.pi, math.pi)
            self._drift_rate = np.array([self._pos_growth * math.cos(a), self._pos_growth * math.sin(a),
                                         self._yaw_growth * self._rng.choice((-1.0, 1.0))])
        self.degraded = degraded
        if degraded:
            self._drift = self._drift + self._drift_rate * period
            self._drift_sigma = self._drift_sigma + np.array([self._pos_growth, self._yaw_growth]) * period
        else:
            decay = math.exp(-period / self._cfg["degraded"]["recovery_tau_s"])
            self._drift = self._drift * decay
            self._drift_sigma = self._drift_sigma * decay
        return self._apply(m.x, m.y, m.yaw)


def make_localizer(params: dict, start, vehicle, sensor_based: bool, rng: Optional[np.random.Generator] = None,
                   noise_rng: Optional[np.random.Generator] = None, noise_xy: float = 0.0,
                   noise_yaw_deg: float = 0.0) -> Localizer:
    """The estimator localization.estimator asks for. `auto` means truth, or odometry if a sensor-based
    detector is active. `white` is that same choice plus noise. noise_xy / noise_yaw_deg (--noise-xy,
    --noise-yaw) add white noise on top of any estimator; `white` with neither uses the configured sizes."""
    cfg = params["localization"]
    name = cfg["estimator"]
    if name not in ESTIMATORS:
        raise ValueError(f"unknown localization.estimator {name!r}; expected one of {ESTIMATORS}")
    rng = rng if rng is not None else np.random.default_rng(0)
    if name == "white" and not noise_xy and not noise_yaw_deg:
        noise_xy, noise_yaw_deg = cfg["white"]["noise_xy"], cfg["white"]["noise_yaw_deg"]
    base = ("odometry" if sensor_based else "truth") if name in ("auto", "white") else name
    if base == "truth":
        loc: Localizer = TruthLocalizer(start)
    elif base == "odometry":
        loc = WheelOdometry(start, cfg["odometry"], rng, vehicle.wheelbase, vehicle.track, cfg["latency_s"])
    else:
        loc = GnssInsLocalizer(start, cfg["gnss_ins"], rng, cfg["latency_s"])
    if noise_xy or noise_yaw_deg:
        loc = WhiteNoiseLocalizer(loc, noise_rng if noise_rng is not None else np.random.default_rng(0), noise_xy, noise_yaw_deg)
    return loc
