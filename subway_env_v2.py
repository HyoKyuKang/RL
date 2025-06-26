"""
subway_env_v2_votes.py – SubwayCoolingEnv with 1‑minute time and 5‑minute vote window
──────────────────────────────────────────────────────────────────────────────
• Time dimension is **minute of day** (0 … 1439).
• Votes arriving each minute are stored in a 5‑minute sliding window.
• Comfort statistics come from the aggregated votes; fallback to PMV if none.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

# ╔══════════════════════════════════════════════════════╗
# ║   pythermalcomfort   (v2 / v3 / v4) compatibility   ║
# ╚══════════════════════════════════════════════════════╝

def _import_pmv():
    """Return fn(**kw) → (pmv, ppd) regardless of pythermalcomfort version."""

    try:
        from pythermalcomfort.models import pmv_ppd as pmv_ppd_v2  # v2.x

        def f(**kw):
            res = pmv_ppd_v2(**kw)
            return (res["pmv"], res["ppd"]) if isinstance(res, dict) else res

        return f
    except (ImportError, AttributeError):
        pass

    try:
        from pythermalcomfort.models import pmv as _pmv, ppd as _ppd  # v3.x
        return lambda **kw: (_pmv(**kw), _ppd(pmv=_pmv(**kw)))
    except (ImportError, AttributeError):
        pass

    try:
        from pythermalcomfort.models.steady_state import pmv_ppd as pmv_ppd_ss  # v3.2+
        return lambda **kw: (
            lambda d=pmv_ppd_ss(**kw): (d["pmv"], d["ppd"])
        )()
    except (ImportError, AttributeError, KeyError):
        pass

    print("⚠️  pythermalcomfort not found – using fallback comfort model.")

    def _fallback(tdb, vr, **_):  # crude linear approximation
        pmv_val = 0.25 * (tdb - 24) - 0.35 * vr
        ppd_val = float(np.clip(abs(pmv_val) * 33 - 5, 5, 100))
        return pmv_val, ppd_val

    return _fallback


_calc_pmv_ppd = _import_pmv()

MIN_PER_DAY = 24 * 60
VOTE_WINDOW = 5  # minutes kept in sliding window


class SubwayCoolingEnv(gym.Env):
    """Subway HVAC control with real passenger feedback (minute resolution)."""

    # Reward weights
    K_DIR, K_COMFORT, K_ENERGY, K_OVER = -0.25, -0.5, -0.1, -0.2
    K_RATE, K_WRONG, K_DISLIKE = -0.3, -1.0, -0.6

    MAX_STAY = 60.0  # minutes a passenger can have stayed
    ALPHA = 0.20     # inertia for AC device lag

    # fan‑level → air‑speed (m/s)
    FAN_V = [0.10, 0.20, 0.30, 0.40, 0.50]

    def __init__(self, max_steps: int = 720):  # default: half‑day rollout
        super().__init__()

        low = np.array(
            [0, 0,   # time(min), passengers
             -1.0, 0.0, 0.0,   # comfort_avg, comfort_std, ratio_comfy
             0.0, 0.0,         # x, y
             0.0,              # stay(min)
             -10.0, 1.0,       # ext_temp, ext_hum
             0.0, 0.0],        # cur_tlv, cur_flv
            dtype=np.float32,
        )
        high = np.array(
            [MIN_PER_DAY - 1, 100,
             1.0, 1.0, 1.0,
             1.0, 1.0,
             self.MAX_STAY,
             40.0, 100.0,
             12.0, 4.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low, high)
        self.action_space = spaces.MultiDiscrete([13, 5])  # 0‑12 temp_lv, 0‑4 fan_lv
        self.max_steps = max_steps

        self._votes_buffer: list[list[int]] = []
        self.np_random, _ = seeding.np_random(None)
        self.reset()

    # ---------------------------------------------------
    def update_votes(self, votes: list[int] | np.ndarray | None):
        """Call **before** step; pushes vote list for current minute."""
        self._votes_buffer.append(list(votes) if votes is not None else [])
        if len(self._votes_buffer) > VOTE_WINDOW:
            self._votes_buffer.pop(0)

    # ---------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)

        self.current_step = 0
        self.num_comfort = 0
        self.total_energy = 0.0
        self.total_overreaction = 0
        self._votes_buffer.clear()

        self.state = np.array(
            [0, 50, 0.0, 0.0, 1.0, 0.5, 0.5, 0.0, 25.0, 50.0, 6.0, 2.0],
            dtype=np.float32,
        )
        return self.state, {}

    # ---------------------------------------------------
    def step(self, action):
        # unpack state
        (
            t_min, pax, comfort_avg, comfort_std, ratio_comfy,
            x, y, stay, ext_t, ext_h, cur_tlv, cur_flv,
        ) = self.state

        set_tlv, set_flv = map(int, action)

        # inertia (device lag)
        cur_tlv += self.ALPHA * (set_tlv - cur_tlv)
        cur_flv += self.ALPHA * (set_flv - cur_flv)
        temp_c = 18.0 + cur_tlv
        fan_lv = cur_flv

        # ----- vote aggregation -----
        all_votes = [v for minute in self._votes_buffer for v in minute]
        if all_votes:
            v_arr = np.asarray(all_votes, dtype=np.float32)
            comfort_avg = float(np.clip(v_arr.mean() / 2, -1.0, 1.0))
            comfort_std = float(np.clip(v_arr.std() / 2, 0.0, 1.0))
            ratio_comfy = float(np.mean(np.abs(v_arr) <= 1))
            dislike_ratio = float(np.mean(np.abs(v_arr) >= 2))
        else:
            # fallback to PMV
            v_air = self.FAN_V[int(np.clip(round(fan_lv), 0, 4))]
            safe_rh = float(np.clip(ext_h, 1.0, 100.0))
            pmv_val, _ = _calc_pmv_ppd(tdb=temp_c, tr=temp_c, vr=v_air,
                                       rh=safe_rh, met=1.2, clo=0.5)
            pmv_val = 0.0 if not np.isfinite(pmv_val) else pmv_val
            target_comf = float(np.clip(pmv_val / 3.0, -1.0, 1.0))
            comfort_avg += 0.3 * (target_comf - comfort_avg)
            comfort_avg = float(np.clip(comfort_avg, -1.0, 1.0))
            comfort_std = 0.0
            ratio_comfy = 1.0 if abs(comfort_avg) < 0.3 else 0.0
            dislike_ratio = 1.0 - ratio_comfy

        if abs(comfort_avg) < 0.3:
            self.num_comfort += 1

        # ----- energy & penalties -----
        no_ac_t = ext_t + 0.01 * pax
        d_cool = max(0.0, no_ac_t - temp_c)
        d_heat = max(0.0, temp_c - no_ac_t)
        energy = 0.8 * d_cool + 0.5 * d_heat + 0.3 * fan_lv
        self.total_energy += energy
        over = int(d_cool >= 6) + int(d_heat >= 6) + int(fan_lv >= 5)
        self.total_overreaction += over

        rate_pen = self.K_RATE * (abs(set_tlv - cur_tlv) + 0.5 * abs(set_flv - cur_flv))
        wrong = (
            comfort_avg < -0.05 and (set_tlv < cur_tlv or set_flv > cur_flv)
        ) or (
            comfort_avg > 0.05 and (set_tlv > cur_tlv or set_flv < cur_flv)
        )
        wrong_pen = self.K_WRONG * int(wrong)
        ideal_tlv, ideal_flv = self._comfort_to_target(comfort_avg)
        mismatch = abs(ideal_tlv - set_tlv) + 0.5 * abs(ideal_flv - set_flv)
        dir_rwd = self.K_DIR * mismatch * (1 + abs(comfort_avg))

        reward = (
            dir_rwd
            + self.K_COMFORT * comfort_avg**2
            + self.K_ENERGY * energy
            + self.K_OVER * over
            + rate_pen
            + wrong_pen
            + self.K_DISLIKE * dislike_ratio
        )

        # ----- environment propagation -----
        self.current_step += 1
        t_min = (t_min + 1) % MIN_PER_DAY
        pax = self._simulate_passengers(t_min // 60)
        x = np.clip(x + self.np_random.uniform(-0.02, 0.02), 0.0, 1.0)
        y = np.clip(y + self.np_random.uniform(-0.02, 0.02), 0.0, 1.0)
        stay = min(stay + 1.0, self.MAX_STAY)
        ext_t = float(np.clip(ext_t + self.np_random.normal(0, 0.1), -10.0, 40.0))
        ext_h = float(np.clip(ext_h + self.np_random.normal(0, 0.4), 1.0, 100.0))

        self.state = np.array([
            t_min, pax, comfort_avg, comfort_std, ratio_comfy,
            x, y, stay, ext_t, ext_h, cur_tlv, cur_flv
        ], dtype=np.float32)

        terminated = False
        truncated = self.current_step >= self.max_steps
        if truncated:
            comfort_ratio = self.num_comfort / self.max_steps
            reward += (
                +1.0 * comfort_ratio
                + self.K_ENERGY * (self.total_energy / self.max_steps)
                + self.K_OVER * (self.total_overreaction / self.max_steps)
            )

        return self.state, float(reward), terminated, truncated, {}

    # ---------------------------------------------------
    def _simulate_passengers(self, hour: int) -> int:
        return self.np_random.integers(60, 100) if 7 <= hour <= 9 or 17 <= hour <= 19 \
            else self.np_random.integers(10, 50)

    # ---------------------------------------------------
    @staticmethod
    def _comfort_to_target(c: float) -> tuple[int, int]:
        """Map comfort (‑1 … +1) → (temp_level 0‑12, fan_level 0‑4)."""
        return (
            int(round(np.clip(24 - 4 * c, 18, 30) - 18)),
            int(round(np.clip(3 + 2 * c, 1, 5) - 1)),
        )
