import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import Counter


class SubwayCoolingEnv(gym.Env):
    """
    강화학습용 지하철 객실 냉방 환경
    - comfort ∈ [-1, 1]:  −1(매우 춥다) ← 0(쾌적) → +1(매우 덥다)
    - temp_level 0 ~ 12  → 18 ℃ ~ 30 ℃
    - fan_level  0 ~ 4   → 1단(약) ~ 5단(강)
    """

    def __init__(self):
        super().__init__()

        # [시간, 승객 수, comfort]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -1.0], dtype=np.float32),
            high=np.array([23, 100, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # temp_level, fan_level
        self.action_space = spaces.MultiDiscrete([13, 5])

        self.max_steps = 60
        self.action_counter = Counter()
        self.reset()

    # ---------- public API -------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.num_comfort = 0
        self.total_energy = 0.0
        self.total_overreaction = 0
        self.state = np.array([8, 50, 0.0], dtype=np.float32)
        self.action_counter.clear()
        return self.state, {}

    def step(self, action):
        """
        action = [temp_level(0-12), fan_level(0-4)]
        """
        now_t, passengers, comfort = self.state
        temp_level, fan_level = map(int, action)
        self.current_step += 1
        self.action_counter.update([tuple(action)])

        # 실제 제어값
        actual_temp = 18.0 + temp_level          # 18-30 ℃
        actual_fan = 1 + fan_level               # 1-5단

        # ------------------------------ comfort 시뮬레이션
        comfort_delta = self._simulate_comfort_change(
            passengers, comfort, actual_temp, actual_fan
        )
        comfort = np.clip(comfort + comfort_delta, -1.0, 1.0)

        if abs(comfort) < 0.3:                   # 쾌적 판정
            self.num_comfort += 1

        # ------------------------------ 비용 계산
        energy = abs(actual_temp - 24.0) + abs(actual_fan - 3)
        self.total_energy += energy

        overreaction = (
            int(actual_temp <= 20.0)
            + int(actual_temp >= 28.0)
            + int(actual_fan == 5)
        )
        self.total_overreaction += overreaction

        # ------------------------------ 방향성 보상 --------------------------------
        #   └ comfort 부호(±)와 temp/fan 조절 방향이 일치하면 추가 보상,
        #     반대면 페널티 (정도(|comfort|)에 비례)
        ideal_temp_lv, ideal_fan_lv = self._comfort_to_target(comfort)
        direction_mismatch = (
            abs(ideal_temp_lv - temp_level) + abs(ideal_fan_lv - fan_level)
        )
        direction_reward = -0.1 * direction_mismatch * (1 + abs(comfort))

        # ------------------------------ step reward
        reward = (
            + direction_reward          # 방향성 보상
            - 0.5 * comfort * comfort   # |comfort|²  (쾌적에 가까울수록 ↑)
            - 0.1 * energy              # 에너지 절약
            - 0.2 * overreaction        # 과도 조작 억제
        )

        # ------------------------------ 다음 상태
        now_t = (now_t + 1) % 24
        passengers = self._simulate_passengers(now_t)
        self.state = np.array([now_t, passengers, comfort], dtype=np.float32)

        # ------------------------------ 에피소드 종료
        terminated = self.current_step >= self.max_steps
        if terminated:
            comfort_ratio = self.num_comfort / self.max_steps
            normalized_energy = self.total_energy / self.max_steps
            overreaction_rate = self.total_overreaction / self.max_steps

            reward += (
                + 1.0 * comfort_ratio
                - 0.3 * normalized_energy
                - 0.5 * overreaction_rate
            )

        return self.state, float(reward), terminated, False, {}

    # ---------- internal helpers ------------------------------------------

    @staticmethod
    def _simulate_passengers(time):
        # 러시 아워(7-9, 17-19) 승객 ↑
        if 7 <= time <= 9 or 17 <= time <= 19:
            return np.random.randint(60, 100)
        return np.random.randint(10, 50)

    @staticmethod
    def _simulate_comfort_change(passengers, comfort, temp, fan):
        """온도·풍량 차이가 승객 체감에 미치는 영향"""
        ideal_temp, ideal_fan = 24.0, 3
        temp_effect = -0.015 * passengers * (temp - ideal_temp)
        fan_effect = -0.01 * passengers * (fan - ideal_fan)
        return 0.01 * (temp_effect + fan_effect)

    @staticmethod
    def _comfort_to_target(comfort: float):
        """
        comfort ∈ [-1,1] → (temp_level, fan_level) 목표치
        · comfort←-1(춥다) → temp↑  fan↓
        · comfort→+1(덥다) → temp↓  fan↑
        맵핑:
            -1 → 28 ℃(lv=10), fan=1(0)
            -0.5 → 26 ℃(lv=8), fan=2(1)
            0   → 24 ℃(lv=6), fan=3(2)
            +0.5→ 22 ℃(lv=4), fan=4(3)
            +1  → 20 ℃(lv=2), fan=5(4)
        """
        # 온도: comfort 비례로 ±4 ℃
        desired_temp = np.clip(24 - 4 * comfort, 18, 30)
        temp_level = int(round(desired_temp - 18))

        # 풍량: comfort 비례로 ±2단
        desired_fan = np.clip(3 + 2 * comfort, 1, 5)
        fan_level = int(round(desired_fan - 1))

        return temp_level, fan_level