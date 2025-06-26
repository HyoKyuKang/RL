import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SubwayCoolingEnv(gym.Env):
    def __init__(self, max_passengers=10):
        super().__init__()
        self.max_passengers = max_passengers

        # 상태: [시간, 외부 온도, 외부 습도, 에어컨 온도, 풍량, 승객들 (위치, 투표)...]
        low = [0.0, 0.0, 0.0, 0.0, 0.0] + [0, -2] * self.max_passengers
        high = [23.0, 40.0, 100.0, 2.0, 2.0] + [4, 2] * self.max_passengers

        self.observation_space = spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            dtype=np.float32
        )

        self.action_space = spaces.MultiDiscrete([3, 3])
        self.max_steps = 60
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.total_energy = 0.0

        self.current_time = np.random.randint(6, 22)
        self.ac_temp = 1
        self.ac_fan = 1
        self.outside_temp = np.random.uniform(20, 35)
        self.outside_humidity = np.random.uniform(30, 80)
        self.passengers = self._generate_passengers()

        return self._get_state(), {}

    def step(self, action):
        self.ac_temp, self.ac_fan = action
        self.current_step += 1

        self.total_energy += (self.ac_temp + self.ac_fan) / 4.0
        self.current_time = (self.current_time + 1) % 24

        for p in self.passengers:
            p["vote"] = self._generate_vote(p["position"])

        terminated = self.current_step >= self.max_steps
        reward, comfort_score, energy_penalty = self._calculate_reward()

        info = {
            "time": self.current_time,
            "votes": [p["vote"] for p in self.passengers],
            "positions": [p["position"] for p in self.passengers],
            "mean_vote": np.mean([p["vote"] for p in self.passengers]),
            "comfort_score": comfort_score,
            "energy_penalty": energy_penalty
        }

        return self._get_state(), reward, terminated, False, info

    def _generate_passengers(self):
        passengers = []
        num = np.random.randint(4, self.max_passengers + 1)
        for _ in range(num):
            pos = np.random.randint(0, 5)
            vote = self._generate_vote(pos)
            passengers.append({"position": pos, "vote": vote})
        return passengers

    def _generate_vote(self, pos):
        perceived_temp = (self.ac_temp - 1) + np.random.normal(0, 0.2)
        if perceived_temp > 0.7:
            return -2  # 매우 춥다
        elif perceived_temp > 0.3:
            return -1
        elif perceived_temp > -0.3:
            return 0
        elif perceived_temp > -0.7:
            return 1
        else:
            return 2  # 매우 덥다

    def _get_state(self):
        base_state = [
            self.current_time,
            self.outside_temp,
            self.outside_humidity,
            self.ac_temp,
            self.ac_fan
        ]
        passenger_state = []
        for p in self.passengers:
            passenger_state.extend([p["position"], p["vote"]])
        while len(passenger_state) < self.max_passengers * 2:
            passenger_state.extend([0, 0])
        return np.array(base_state + passenger_state, dtype=np.float32)

    def _calculate_reward(self):
        votes = [p["vote"] for p in self.passengers]
        comfort_score = -np.mean(np.abs(votes)) / 2.0
        energy_penalty = -0.3 * (self.ac_temp + self.ac_fan) / 4.0
        reward = comfort_score + energy_penalty
        return reward, comfort_score, energy_penalty