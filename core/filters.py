import math
import time
from typing import Optional, Tuple


class EMAFilter2D:
    """Exponential Moving Average (EMA) 2D coordinate filter."""

    def __init__(self, alpha: float = 0.25) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0.0, 1.0]")
        self.alpha = alpha
        self._x: Optional[float] = None
        self._y: Optional[float] = None

    def reset(self) -> None:
        self._x = None
        self._y = None

    def update(
        self,
        x: float,
        y: float,
        timestamp: Optional[float] = None,
        damping_scale: float = 1.0,
    ) -> Tuple[float, float]:
        if self._x is None or self._y is None:
            self._x = x
            self._y = y
        else:
            eff_alpha = max(0.02, min(1.0, self.alpha * damping_scale))
            self._x = eff_alpha * x + (1.0 - eff_alpha) * self._x
            self._y = eff_alpha * y + (1.0 - eff_alpha) * self._y
        return self._x, self._y


class _OneEuroFilter1D:
    """1D One Euro Filter for adaptive low-latency smoothing."""

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.015,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._t_prev: Optional[float] = None

    @staticmethod
    def _smoothing_factor(t_e: float, cutoff: float) -> float:
        r = 2.0 * math.pi * cutoff * t_e
        return r / (r + 1.0)

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def update(
        self,
        x: float,
        t: Optional[float] = None,
        damping_scale: float = 1.0,
    ) -> float:
        now = time.monotonic() if t is None else t
        if self._x_prev is None or self._t_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = now
            return x

        t_e = max(1e-5, now - self._t_prev)

        # Filtered derivative
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self._x_prev) / t_e
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Adaptive cutoff frequency (scaled down during pinch-anchor stabilization)
        scale = max(0.05, min(2.0, damping_scale))
        cutoff = (self.min_cutoff * scale) + (self.beta * scale * scale) * abs(dx_hat)
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = now
        return x_hat


class OneEuroFilter2D:
    """2D One Euro Filter combining low jitter at rest with low lag during fast movement."""

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.015,
        d_cutoff: float = 1.0,
    ) -> None:
        self._fx = _OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self._fy = _OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

    def set_params(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> None:
        self._fx.min_cutoff = min_cutoff
        self._fx.beta = beta
        self._fx.d_cutoff = d_cutoff
        self._fy.min_cutoff = min_cutoff
        self._fy.beta = beta
        self._fy.d_cutoff = d_cutoff

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()

    def update(
        self,
        x: float,
        y: float,
        timestamp: Optional[float] = None,
        damping_scale: float = 1.0,
    ) -> Tuple[float, float]:
        t = time.monotonic() if timestamp is None else timestamp
        return (
            self._fx.update(x, t, damping_scale=damping_scale),
            self._fy.update(y, t, damping_scale=damping_scale),
        )
