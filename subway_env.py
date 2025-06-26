import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SubwayCoolingEnv(gym.Env):
    def __init__(self):
        super(SubwayCoolingEnv, self).__init__()

        # 상태: [시간(0~23), 승객 수(0~100), 쾌적도(-1~1)]
        self.observation_space = spaces.Box(low=np.array([0, 0, -1.0]),
                                            high=np.array([23, 100, 1.0]),
                                            dtype=np.float32)

        # 행동: 온도(0~2), 풍량(0~2)
        self.action_space = spaces.MultiDiscrete([3, 3])

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
        temp_setting, fan_setting = action
        self.current_step += 1

        # 쾌적도 변화 계산
        comfort_delta = self._simulate_comfort_change(passengers, temp_setting, fan_setting)
        comfort = np.clip(comfort + comfort_delta, -1.0, 1.0)

        # 쾌적/불쾌 카운트
        if abs(comfort) < 0.3:
            self.num_comfort += 1
        else:
            self.num_discomfort += 1

        # 에너지 사용량
        energy = temp_setting + fan_setting
        self.total_energy += energy / 4  # 최대 4로 normalize

        # 과잉 조절 여부
        overreaction = int(abs(temp_setting - 1) > 1) + int(abs(fan_setting - 1) > 1)
        self.total_overreaction += overreaction

        # 시간 및 승객 수 갱신
        time = (time + 1) % 24
        passengers = self._simulate_passengers(time)
        self.state = np.array([time, passengers, comfort], dtype=np.float32)

        terminated = self.current_step >= self.max_steps
        reward = 0.0

        if terminated:
            comfort_ratio = self.num_comfort / self.max_steps
            discomfort_ratio = self.num_discomfort / self.max_steps
            normalized_energy_use = self.total_energy / self.max_steps
            overreaction_penalty = self.total_overreaction / self.max_steps

            reward = (
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

    def _simulate_comfort_change(self, passengers, temp_setting, fan_setting):
        ideal = 1
        deviation = abs(temp_setting - ideal) + abs(fan_setting - ideal)
        direction = -1 if temp_setting < ideal else 1
        return -0.01 * passengers * deviation * direction