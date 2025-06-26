"""
subway_env_v2.py  –  PMV 기반 체감 · 에너지 · 패널티 통합 모델
────────────────────────────────────────────────────────────
state = [time, passengers, comfort, x_pos, y_pos,
         enter_time, ext_temp, ext_hum,
         cur_temp_lv, cur_fan_lv]

comfort  = clip(PMV / 3, -1 … +1)
Δcomfort = β · (PMV_new – comfort_prev)   with β = 0.3
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

# ╔══════════════════════════════════════════════════════╗
# ║   pythermalcomfort   (2.x / 3.x / 4.x)  호환 래퍼    ║
# ╚══════════════════════════════════════════════════════╝
def _import_pmv():
    """
    return: fn(**kw) -> (pmv_float, ppd_float)
    * v2.x   : models.pmv_ppd                       (tuple or dict)
    * v3.x   : models.pmv  + models.ppd             (separate)
    * v3.2+  : models.steady_state.pmv_ppd          (dict)
    * absent : very-simple fallback model
    """
    # 1️⃣ v2.x
    try:
        from pythermalcomfort.models import pmv_ppd as pmv_ppd_v2

        def f(**kw):
            res = pmv_ppd_v2(**kw)
            return (res["pmv"], res["ppd"]) if isinstance(res, dict) else res
        return f
    except (ImportError, AttributeError):
        pass

    # 2️⃣ v3.x
    try:
        from pythermalcomfort.models import pmv as _pmv, ppd as _ppd
        return lambda **kw: (_pmv(**kw), _ppd(pmv=_pmv(**kw)))
    except (ImportError, AttributeError):
        pass

    # 3️⃣ v3.2+ / 4.x
    try:
        from pythermalcomfort.models.steady_state import pmv_ppd as pmv_ppd_ss
        return lambda **kw: (lambda d=pmv_ppd_ss(**kw): (d["pmv"], d["ppd"]))()
    except (ImportError, AttributeError, KeyError):
        pass

    # 4️⃣ fallback
    print("⚠️  pythermalcomfort 미설치 → 간이 comfort 모델로 대체합니다.")
    def _fallback(tdb, vr, **_):
        pmv_val = 0.25 * (tdb - 24) - 0.35 * vr    # 매우 단순 추정
        ppd_val = float(np.clip(abs(pmv_val) * 33 - 5, 5, 100))
        return pmv_val, ppd_val
    return _fallback


_calc_pmv_ppd = _import_pmv()

# ════════════════════════════════════════════════════════
#                       ENVIRONMENT
# ════════════════════════════════════════════════════════
class SubwayCoolingEnv(gym.Env):
    # -------- Reward / Penalty coefficients -------------
    K_DIR, K_COMFORT, K_ENERGY, K_OVER = -0.25, -0.5, -0.1, -0.2
    K_RATE, K_WRONG                    = -0.3 , -1.0
    MAX_STAY, ALPHA                    = 60.0 , 0.20      # inertia

    # fan-level → air speed (m/s)
    FAN_V = [0.10, 0.20, 0.30, 0.40, 0.50]

    def __init__(self, max_steps: int = 60):
        super().__init__()
        low  = np.array([0, 0, -1.0, 0, 0,   0, -10,   1, 0, 0], dtype=np.float32)
        high = np.array([23,100, 1.0, 1, 1,  60,  40, 100,12, 4], dtype=np.float32)
        self.observation_space = spaces.Box(low, high)
        self.action_space      = spaces.MultiDiscrete([13, 5])
        self.max_steps = max_steps

        self.np_random, _ = seeding.np_random(None)
        self.reset()

    # ----------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)

        self.current_step        = 0
        self.num_comfort         = 0
        self.total_energy        = 0.0
        self.total_overreaction  = 0

        # [time, pax, comfort, x, y, stay, extT, extH, cur_tlv, cur_flv]
        self.state = np.array(
            [8, 50, 0.0,   0.5, 0.5, 0.0, 25.0, 50.0, 6.0, 2.0], dtype=np.float32
        )
        return self.state, {}

    # ----------------------------------------------------
    def step(self, action):
        (t, pax, comfort, x, y, stay,
         ext_t, ext_h, cur_tlv, cur_flv) = self.state

        set_tlv, set_flv = map(int, action)

        # ---- inertia (device lag) -----------------------
        cur_tlv += self.ALPHA * (set_tlv - cur_tlv)
        cur_flv += self.ALPHA * (set_flv - cur_flv)
        temp_c   = 18 + cur_tlv
        fan_lv   = cur_flv

        # ---- comfort update by PMV ----------------------
        v        = self.FAN_V[int(np.clip(round(fan_lv), 0, 4))]
        safe_rh  = float(np.clip(ext_h, 1.0, 100.0))       # ⬅️ 0 % 방지
        pmv_val, _ = _calc_pmv_ppd(tdb=temp_c, tr=temp_c, vr=v,
                                   rh=safe_rh, met=1.2, clo=0.5)

        if not np.isfinite(pmv_val):                       # NaN → 0
            pmv_val = 0.0

        target_comf = float(np.clip(pmv_val / 3.0, -1.0, 1.0))
        comfort    += 0.3 * (target_comf - comfort)        # β
        comfort     = float(np.clip(comfort, -1.0, 1.0))
        if abs(comfort) < 0.3:
            self.num_comfort += 1

        # ---- energy model --------------------------------
        no_ac_t = ext_t + 0.01 * pax
        d_cool  = max(0.0, no_ac_t - temp_c)
        d_heat  = max(0.0, temp_c - no_ac_t)
        energy  = 0.8 * d_cool + 0.5 * d_heat + 0.3 * fan_lv
        self.total_energy += energy

        over = int(d_cool >= 6) + int(d_heat >= 6) + int(fan_lv >= 5)
        self.total_overreaction += over

        # ---- penalties & rewards ------------------------
        rate_pen = self.K_RATE * (abs(set_tlv - cur_tlv) +
                                  0.5 * abs(set_flv - cur_flv))

        wrong = (comfort < -0.05 and (set_tlv < cur_tlv or set_flv > cur_flv)) or \
                (comfort >  0.05 and (set_tlv > cur_tlv or set_flv < cur_flv))
        wrong_pen = self.K_WRONG * int(wrong)

        ideal_tlv, ideal_flv = self._comfort_to_target(comfort)
        mismatch  = abs(ideal_tlv - set_tlv) + 0.5 * abs(ideal_flv - set_flv)
        dir_rwd   = self.K_DIR * mismatch * (1 + abs(comfort))

        reward = (dir_rwd
                  + self.K_COMFORT * comfort**2
                  + self.K_ENERGY  * energy
                  + self.K_OVER    * over
                  + rate_pen
                  + wrong_pen)

        # ---- propagate environment ----------------------
        self.current_step += 1
        t     = (t + 1) % 24
        pax   = self._simulate_passengers(t)
        x     = np.clip(x + self.np_random.uniform(-0.05, 0.05), 0, 1)
        y     = np.clip(y + self.np_random.uniform(-0.05, 0.05), 0, 1)
        stay  = min(stay + 1.0, self.MAX_STAY)
        ext_t = float(np.nan_to_num(np.clip(ext_t + self.np_random.normal(0, 0.5),
                                            -10, 40)))
        ext_h = float(np.nan_to_num(np.clip(ext_h + self.np_random.normal(0, 2.0),
                                            1, 100)))   # ⬅️ 1 % 이상 유지

        self.state = np.array(
            [t, pax, comfort, x, y, stay, ext_t, ext_h, cur_tlv, cur_flv],
            dtype=np.float32
        )

        terminated = False
        truncated  = self.current_step >= self.max_steps
        if truncated:
            comfort_ratio = self.num_comfort / self.max_steps
            reward += (+1.0 * comfort_ratio
                       + self.K_ENERGY * (self.total_energy / self.max_steps)
                       + self.K_OVER   * (self.total_overreaction / self.max_steps))

        return self.state, float(reward), terminated, truncated, {}

    # ----------------------------------------------------
    def _simulate_passengers(self, hour: int) -> int:
        hour = int(hour)
        return self.np_random.integers(60, 100) if 7 <= hour <= 9 or 17 <= hour <= 19 \
               else self.np_random.integers(10, 50)

    # ----------------------------------------------------
    @staticmethod
    def _comfort_to_target(c: float) -> tuple[int, int]:
        """
        comfort(-1~1) → (temp_level 0-12, fan_level 0-4)
        24 ℃·3단이 기준점
        """
        return (int(round(np.clip(24 - 4 * c, 18, 30) - 18)),
                int(round(np.clip(3 + 2 * c, 1, 5) - 1)))