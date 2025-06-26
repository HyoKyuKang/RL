"""
subway_env_v2.py  (위치 감쇠 버전)
───────────────────────────────────────────────────────
state = [time, passengers, comfort, x_pos, y_pos]
  · x_pos, y_pos ∈ [0,1]  (0.5,0.5 = 객실 가운데)
  · 끝으로 갈수록 냉방 영향 가중치 w ↑  (여기선 w = 2·d,  d: 중심과의 L∞ 거리)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from collections import Counter


class SubwayCoolingEnv(gym.Env):
    # ── 보상 계수 ───────────────────────────────────────
    K_DIR = -0.25
    K_COMFORT = -0.5
    K_ENERGY = -0.1
    K_OVER = -0.2

    def __init__(self, max_steps: int = 60):
        super().__init__()

        # --- 관측 공간: x_pos, y_pos 추가 (0~1) ----------
        low  = np.array([0, 0, -1.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([23, 100, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low, high)

        # --- 행동 공간 ----------------------------------
        self.action_space = spaces.MultiDiscrete([13, 5])

        self.max_steps = max_steps
        self.action_counter: Counter[tuple[int, int]] = Counter()
        self.np_random, _ = seeding.np_random(None)

        self.reset()

    # ============= 핵심 로직 =============================================

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)

        self.current_step = 0
        self.num_comfort = 0
        self.total_energy = 0.0
        self.total_overreaction = 0

        # 초기 위치: 객실 중앙
        self.state = np.array([8, 50, 0.0, 0.5, 0.5], dtype=np.float32)
        self.action_counter.clear()
        return self.state, {}

    # ---------------------------------------------------------------------

    def step(self, action):
        t, passengers, comfort, x_pos, y_pos = self.state
        tlv, flv = map(int, action)

        temp = 18 + tlv
        fan  = 1 + flv

        # -------- 위치 가중치 w -------------------------------------------
        #   w = 0 (중앙) → 1 (가장자리)  : L∞ 거리 * 2
        w = 2.0 * max(abs(x_pos - 0.5), abs(y_pos - 0.5))

        d_comfort = self._simulate_comfort_change(passengers, temp, fan)
        comfort = np.clip(comfort + d_comfort * (1 + w), -1.0, 1.0)
        if abs(comfort) < 0.3:
            self.num_comfort += 1

        # -------- 비용 -----------------
        energy = abs(temp - 24) + abs(fan - 3)
        self.total_energy += energy
        overreaction = int(temp <= 20) + int(temp >= 28) + int(fan == 5)
        self.total_overreaction += overreaction

        # -------- 방향성 보상 ----------
        ideal_tlv, ideal_flv = self._comfort_to_target(comfort)
        mismatch = abs(ideal_tlv - tlv) + 0.5 * abs(ideal_flv - flv)
        dir_reward = self.K_DIR * mismatch * (1 + abs(comfort))

        reward = (
            dir_reward
            + self.K_COMFORT * comfort * comfort
            + self.K_ENERGY * energy
            + self.K_OVER * overreaction
        )

        # -------- 다음 상태 ------------
        self.current_step += 1
        t = (t + 1) % 24
        passengers = self._simulate_passengers(t)

        # 위치 임의 이동(예시): ±0.05 무작위
        x_pos = np.clip(x_pos + self.np_random.uniform(-0.05, 0.05), 0.0, 1.0)
        y_pos = np.clip(y_pos + self.np_random.uniform(-0.05, 0.05), 0.0, 1.0)

        self.state = np.array([t, passengers, comfort, x_pos, y_pos], dtype=np.float32)

        terminated = False
        truncated = self.current_step >= self.max_steps
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
    def _simulate_comfort_change(passengers, temp, fan):
        ideal_temp, ideal_fan = 24.0, 3
        temp_eff = +0.015 * passengers * (temp - ideal_temp)
        fan_eff  = -0.01  * passengers * (fan  - ideal_fan)
        return 0.01 * (temp_eff + fan_eff)

    @staticmethod
    def _comfort_to_target(c):
        temp_lv = int(round(np.clip(24 - 4 * c, 18, 30) - 18))
        fan_lv  = int(round(np.clip(3 + 2 * c, 1, 5) - 1))
        return temp_lv, fan_lv