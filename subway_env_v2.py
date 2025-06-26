"""
subway_env_v2.py
────────────────────────────────────────────────────────
지하철 객실 냉방 제어를 위한 Gymnasium 환경 (강화학습용)

주요 개선 사항
─────────────
1. RNG 일원화: self.np_random 사용 → seed 재현성 확보
2. `terminated`/`truncated` 구분: 타임리밋 도달 시 truncated=True
3. 보상 계수(K_*) 상수화로 튜닝 편의성 ↑
4. fan / temp 스케일 편차 보정(0.5 가중치)로 균형 학습
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from collections import Counter


class SubwayCoolingEnv(gym.Env):
    """지하철 냉방 환경
    - comfort ∈ [-1, 1]  (−1: 매우 춥다, 0: 쾌적, +1: 매우 덥다)
    - action = [temp_level 0-12 → 18-30 ℃,  fan_level 0-4 → 1-5단]
    """

    # ---------- 보상 계수 --------------------------------------------------
    K_DIR = -0.2      # 방향성 불일치 패널티
    K_COMFORT = -0.5  # |comfort|²
    K_ENERGY = -0.1   # 에너지 사용
    K_OVER = -0.2     # 과도 조작

    def __init__(self, max_steps: int = 60):
        super().__init__()
        # 관측치: [시간(0-23), 승객 수(0-100), comfort(-1-1)]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -1.0], dtype=np.float32),
            high=np.array([23, 100, 1.0], dtype=np.float32),
        )
        # 행동: [temp_level(0-12), fan_level(0-4)]
        self.action_space = spaces.MultiDiscrete([13, 5])

        self.max_steps = max_steps
        self.action_counter: Counter[tuple[int, int]] = Counter()

        # RNG
        self.np_random, _ = seeding.np_random(None)

        # 상태 초기화
        self.reset()

    # ---------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)

        self.current_step = 0
        self.num_comfort = 0
        self.total_energy = 0.0
        self.total_overreaction = 0

        self.state = np.array([8, 50, 0.0], dtype=np.float32)  # 초기 상태
        self.action_counter.clear()

        return self.state, {}

    # ---------------------------------------------------------------------

    def step(self, action):
        """Gymnasium step"""
        time, passengers, comfort = self.state
        temp_level, fan_level = map(int, action)

        # 실제 제어값 (온도 °C, 풍량 단계)
        temp = 18.0 + temp_level     # 18-30 ℃
        fan = 1 + fan_level          # 1-5단

        # ---------- 체감 온도 변화 ----------------------------------------
        d_comfort = self._simulate_comfort_change(passengers, comfort, temp, fan)
        comfort = np.clip(comfort + d_comfort, -1.0, 1.0)
        if abs(comfort) < 0.3:
            self.num_comfort += 1

        # ---------- 비용 및 패널티 ----------------------------------------
        energy = abs(temp - 24.0) + abs(fan - 3)
        self.total_energy += energy

        overreaction = int(temp <= 20) + int(temp >= 28) + int(fan == 5)
        self.total_overreaction += overreaction

        # ---------- 방향성 보상 -------------------------------------------
        ideal_tlv, ideal_flv = self._comfort_to_target(comfort)
        # fan 레벨 스케일이 temp(0-12)에 비해 작으므로 0.5 가중치
        mismatch = abs(ideal_tlv - temp_level) + 0.5 * abs(ideal_flv - fan_level)
        dir_reward = self.K_DIR * mismatch * (1 + abs(comfort))

        # ---------- step reward -------------------------------------------
        reward = (
            dir_reward
            + self.K_COMFORT * comfort * comfort
            + self.K_ENERGY * energy
            + self.K_OVER * overreaction
        )

        # ---------- 다음 상태 --------------------------------------------
        self.current_step += 1
        time = (time + 1) % 24
        passengers = self._simulate_passengers(time)
        self.state = np.array([time, passengers, comfort], dtype=np.float32)

        # ---------- 종료 조건 --------------------------------------------
        terminated = False                         # 자연 종료 없음
        truncated = self.current_step >= self.max_steps

        if truncated:  # 에피소드 후 보너스/패널티
            comfort_ratio = self.num_comfort / self.max_steps
            norm_energy = self.total_energy / self.max_steps
            over_rate = self.total_overreaction / self.max_steps
            reward += (
                + 1.0 * comfort_ratio
                - 0.3 * norm_energy
                - 0.5 * over_rate
            )

        return self.state, float(reward), terminated, truncated, {}

    # ================== 내부 함수 =========================================

    def _simulate_passengers(self, t):
        """시간대별 승객 수 시뮬레이션 (정수 t)"""
        t = int(t)
        if 7 <= t <= 9 or 17 <= t <= 19:
            return self.np_random.integers(60, 100)
        return self.np_random.integers(10, 50)

    @staticmethod
    def _simulate_comfort_change(passengers, comfort, temp, fan):
        ideal_temp, ideal_fan = 24.0, 3
        temp_effect = +0.015 * passengers * (temp - ideal_temp)
        fan_effect = -0.01 * passengers * (fan - ideal_fan)
        return 0.01 * (temp_effect + fan_effect)

    @staticmethod
    def _comfort_to_target(comfort: float):
        """
        comfort → (temp_level, fan_level) 목표치
        -1 → temp_lv 10 (28 ℃), fan_lv 0
        0  → temp_lv  6 (24 ℃), fan_lv 2
        +1 → temp_lv  2 (20 ℃), fan_lv 4
        """
        desired_temp = np.clip(24 - 4 * comfort, 18, 30)
        temp_level = int(round(desired_temp - 18))

        desired_fan = np.clip(3 + 2 * comfort, 1, 5)
        fan_level = int(round(desired_fan - 1))

        return temp_level, fan_level