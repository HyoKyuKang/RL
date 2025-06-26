import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SubwayCoolingEnv(gym.Env):
    def __init__(self):
        super(SubwayCoolingEnv, self).__init__()

        self.observation_space = spaces.Box(
            low=np.array([0, 0, -1.0], dtype=np.float32),
            high=np.array([23, 100, 1.0], dtype=np.float32)
        )

        # 행동 공간: 온도 18~30도 (13단계), 풍량 1~5단계 (5단계)
        self.action_space = spaces.MultiDiscrete([13, 5])

        self.max_steps = 60
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.num_comfort = 0
        self.num_discomfort = 0
        self.total_energy = 0.0
        self.total_overreaction = 0
        self.state = np.array([8, 50, 0.0], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        time, passengers, comfort = self.state
        temp_level, fan_level = action
        self.current_step += 1

        actual_temp = 18.0 + temp_level
        actual_fan = 1 + fan_level

        # 쾌적도 변화
        comfort_delta = self._simulate_comfort_change(passengers, actual_temp, actual_fan)
        comfort = np.clip(comfort + comfort_delta, -1.0, 1.0)

        # 쾌적도 기록
        if abs(comfort) < 0.3:
            self.num_comfort += 1
        else:
            self.num_discomfort += 1

        # 에너지 소비
        energy = abs(actual_temp - 24.0) + abs(actual_fan - 3)
        self.total_energy += energy / 12.0

        # 과잉 조절 판단
        overreaction = int(actual_temp < 20.0) + int(actual_temp > 28.0) + int(actual_fan == 5)
        self.total_overreaction += overreaction

        # 즉시 보상 계산 (comfort 중심 + penalty)
        reward = -abs(comfort)                         # 쾌적도 0에서 멀어질수록 패널티
        reward -= 0.2 * (energy / 12.0)                # 에너지 패널티
        reward -= 0.5 * overreaction                   # 과잉 조절 패널티

        # 상태 갱신
        time = (time + 1) % 24
        passengers = self._simulate_passengers(time)
        self.state = np.array([time, passengers, comfort], dtype=np.float32)

        # 종료 보상
        terminated = self.current_step >= self.max_steps
        if terminated:
            comfort_ratio = self.num_comfort / self.max_steps
            discomfort_ratio = self.num_discomfort / self.max_steps
            normalized_energy_use = self.total_energy / self.max_steps
            overreaction_penalty = self.total_overreaction / self.max_steps

            reward += (
                1.0 * comfort_ratio
                - 1.0 * discomfort_ratio
                - 0.3 * normalized_energy_use
                - 0.5 * overreaction_penalty
            )

        return self.state, reward, terminated, False, {}

    def _simulate_passengers(self, time):
        if 7 <= time <= 9 or 17 <= time <= 19:
            return np.random.randint(60, 100)
        else:
            return np.random.randint(10, 50)

    def _simulate_comfort_change(self, passengers, temp, fan):
        ideal_temp = 24.0
        ideal_fan = 3

        # 덥거나 추운 방향을 반영한 discomfort 계산
        temp_diff = temp - ideal_temp
        if self.state[2] > 0:  # 현재 덥다 (comfort > 0)
            temp_discomfort = 0.01 * passengers * max(0, temp_diff)  # 온도가 높을수록 불쾌
        else:  # 현재 춥다 (comfort < 0)
            temp_discomfort = 0.01 * passengers * max(0, -temp_diff)  # 온도가 낮을수록 불쾌

        fan_discomfort = 0.005 * passengers * abs(fan - ideal_fan)

        total_discomfort = temp_discomfort + fan_discomfort
        return -total_discomfort