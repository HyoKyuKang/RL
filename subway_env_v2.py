"""
subway_env_v2.py  (위치 × 체류시간 가중 버전)
───────────────────────────────────────────────────────
state = [time, passengers, comfort, x_pos, y_pos, enter_time]
  · x_pos, y_pos   ∈ [0,1]   (0.5,0.5 = 객실 중앙)
  · enter_time     ∈ [0,60]  (객실에 머문 시간, 분 단위)
냉방 민감도 가중치 = (1 + w_pos) · (1 + w_time)

    w_pos  = 2·L∞(중심 거리)         (0~1)
    w_time = max(1 − enter_time/30, 0) (0~1)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from collections import Counter


class SubwayCoolingEnv(gym.Env):
    # ── 보상 계수 ───────────────────────────────────────
    K_DIR      = -0.25
    K_COMFORT  = -0.5
    K_ENERGY   = -0.1
    K_OVER     = -0.2
    MAX_STAY   = 60.0   # enter_time 상한 (분)

    def __init__(self, max_steps: int = 60):
        super().__init__()

        # --- 관측 공간 (6차원) ---------------------------
        low  = np.array([0, 0, -1.0, 0.0, 0.0, 0.0],   dtype=np.float32)
        high = np.array([23, 100,  1.0, 1.0, 1.0, self.MAX_STAY], dtype=np.float32)
        self.observation_space = spaces.Box(low, high)

        # --- 행동 공간 ----------------------------------
        self.action_space = spaces.MultiDiscrete([13, 5])

        self.max_steps = max_steps
        self.action_counter: Counter[tuple[int, int]] = Counter()
        self.np_random, _ = seeding.np_random(None)

        self.reset()

    # ---------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)

        self.current_step      = 0
        self.num_comfort       = 0
        self.total_energy      = 0.0
        self.total_overreaction = 0

        # 초기 state: 객실 중앙, 입장 직후
        self.state = np.array([8, 50, 0.0, 0.5, 0.5, 0.0], dtype=np.float32)
        self.action_counter.clear()
        return self.state, {}

    # ---------------------------------------------------------------------
    def step(self, action):
        t, pax, comfort, x_pos, y_pos, stay = self.state
        tlv, flv = map(int, action)

        temp, fan = 18 + tlv, 1 + flv

        # ---------- 위치·시간 가중치 ----------------------
        w_pos  = 2.0 * max(abs(x_pos - 0.5), abs(y_pos - 0.5))          # 0~1
        w_time = max(1.0 - stay / 30.0, 0.0)                            # 0~1

        d_c = self._simulate_comfort_change(pax, temp, fan)
        comfort = np.clip(comfort + d_c * (1 + w_pos) * (1 + w_time), -1.0, 1.0)
        if abs(comfort) < 0.3:
            self.num_comfort += 1

        # ---------- 비용 -------------------------------
        energy = abs(temp - 24) + abs(fan - 3)
        self.total_energy += energy
        over = int(temp <= 20) + int(temp >= 28) + int(fan == 5)
        self.total_overreaction += over

        # ---------- 방향성 보상 ------------------------
        ideal_tlv, ideal_flv = self._comfort_to_target(comfort)
        mismatch   = abs(ideal_tlv - tlv) + 0.5 * abs(ideal_flv - flv)
        dir_reward = self.K_DIR * mismatch * (1 + abs(comfort))

        reward = (
            dir_reward
            + self.K_COMFORT * comfort * comfort
            + self.K_ENERGY  * energy
            + self.K_OVER    * over
        )

        # ---------- 다음 상태 -------------------------
        self.current_step += 1
        t   = (t + 1) % 24
        pax = self._simulate_passengers(t)

        # 임의 위치 이동(예시)
        x_pos = np.clip(x_pos + self.np_random.uniform(-0.05, 0.05), 0.0, 1.0)
        y_pos = np.clip(y_pos + self.np_random.uniform(-0.05, 0.05), 0.0, 1.0)
        stay  = min(stay + 1.0, self.MAX_STAY)   # 1분 경과

        self.state = np.array([t, pax, comfort, x_pos, y_pos, stay], dtype=np.float32)

        terminated = False
        truncated  = self.current_step >= self.max_steps
        if truncated:
            comfort_ratio = self.num_comfort / self.max_steps
            reward += (
                +1.0 * comfort_ratio
                -0.3 * (self.total_energy / self.max_steps)
                -0.5 * (self.total_overreaction / self.max_steps)
            )

        return self.state, float(reward), terminated, truncated, {}

    # =================== 헬퍼 ============================================
    def _simulate_passengers(self, tt):
        tt = int(tt)
        if 7 <= tt <= 9 or 17 <= tt <= 19:
            return self.np_random.integers(60, 100)
        return self.np_random.integers(10, 50)

    @staticmethod
    def _simulate_comfort_change(pax, temp, fan):
        temp_eff = +0.015 * pax * (temp - 24.0)
        fan_eff  = -0.01  * pax * (fan  - 3.0)
        return 0.01 * (temp_eff + fan_eff)

    @staticmethod
    def _comfort_to_target(c):
        temp_lv = int(round(np.clip(24 - 4 * c, 18, 30) - 18))
        fan_lv  = int(round(np.clip(3 + 2 * c, 1, 5) - 1))
        return temp_lv, fan_lv