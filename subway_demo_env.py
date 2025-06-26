import gymnasium as gym
from gymnasium import spaces
import numpy as np

TEMP_VALUES = list(np.arange(18.0, 31.0, 1.0))
FAN_VALUES = [0.5, 1.0, 1.5]

class SubwayCoolingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        max_passengers: int = 10,
        ac_position: np.ndarray = np.array([2.5, 0.0]),
        sigma: float = 1.0,
        perceived_temp_limits: tuple = (18, 40),
        target_temp: float = 25.0
    ):
        super().__init__()
        self.max_passengers = max_passengers
        self.ac_position = np.array(ac_position, dtype=np.float32)
        self.sigma = sigma
        self.temp_min, self.temp_max = perceived_temp_limits
        self.target_temp = target_temp

        self.action_space = spaces.MultiDiscrete([5, 5])  # delta -2~+2 for both temp and fan

        dim = 5 + 2 + max_passengers * 3
        low = np.array([0.0, 0.0, 0.0, 0.0, 0.0] + 
                       list(self.ac_position) + 
                       [0.0, 0.0, -1.5] * max_passengers, dtype=np.float32)
        high = np.array([23.0, 40.0, 100.0, len(TEMP_VALUES)-1, len(FAN_VALUES)-1] + 
                        list(self.ac_position) + 
                        [5.0, 5.0, 1.5] * max_passengers, dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.max_steps = 60
        self.vote_thresholds = [22, 23, 26.5, 28]
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.total_energy = 0.0

        self.current_time = np.random.randint(6, 22)
        self.ac_temp_idx = TEMP_VALUES.index(25.0)
        self.ac_fan_idx = FAN_VALUES.index(1.0)
        self.prev_ac_temp_idx = self.ac_temp_idx
        self.prev_ac_fan_idx = self.ac_fan_idx

        self.ac_temp = TEMP_VALUES[self.ac_temp_idx]
        self.ac_fan = FAN_VALUES[self.ac_fan_idx]

        self.outside_temp = np.random.uniform(20, 35)
        self.outside_humidity = np.random.uniform(30, 80)
        self.room_temp = self._estimate_room_temp()
        self.passengers = self._generate_passengers()

        return self._get_state(), {}

    def step(self, action):
        temp_delta = action[0] - 2
        fan_delta = action[1] - 2

        self.prev_ac_temp_idx = self.ac_temp_idx
        self.prev_ac_fan_idx = self.ac_fan_idx

        self.ac_temp_idx = int(np.clip(self.ac_temp_idx + temp_delta, 0, len(TEMP_VALUES) - 1))
        self.ac_fan_idx = int(np.clip(self.ac_fan_idx + fan_delta, 0, len(FAN_VALUES) - 1))

        self.ac_temp = TEMP_VALUES[self.ac_temp_idx]
        self.ac_fan = FAN_VALUES[self.ac_fan_idx]

        self.current_step += 1
        self.total_energy += (30.0 - self.ac_temp) / 12.0 + (self.ac_fan - 0.5)
        self.current_time = (self.current_time + 1) % 24

        self.room_temp = self._estimate_room_temp()

        for p in self.passengers:
            dist = np.linalg.norm(p["position"] - self.ac_position)
            attenuation = np.exp(-dist / self.sigma)

            delta = (
                (self.room_temp - p["perceived_temp"]) * 0.15 * attenuation + np.random.normal(-0.03, 0.03)
            )
            p["perceived_temp"] = np.clip(p["perceived_temp"] + delta, self.temp_min, self.temp_max)
            p["vote"] = self._vote_from_temp(p["perceived_temp"])

        terminated = self.current_step >= self.max_steps
        truncated = False
        reward, comfort_ratio, _, energy_penalty, overreaction_penalty = self._calculate_reward()

        obs = self._get_state()
        info = {
            "time": self.current_time,
            "votes": [p["vote"] for p in self.passengers],
            "positions": [p["position"].tolist() for p in self.passengers],
            "mean_vote": np.mean([p["vote"] for p in self.passengers]),
            "comfort_ratio": comfort_ratio,
            "energy_penalty": energy_penalty,
            "overreaction_penalty": overreaction_penalty,
            "reward": reward,
            "ac_temp": self.ac_temp,
            "ac_fan": self.ac_fan
        }
        return obs, reward, terminated, truncated, info

    def _generate_passengers(self):
        passengers = []
        n = np.random.randint(3, self.max_passengers + 1)
        for _ in range(n):
            x, y = np.random.uniform(0.0, 5.0), np.random.uniform(0.0, 5.0)
            humidity_factor = self.outside_humidity / 100.0
            init_temp = ((self.outside_temp - self.target_temp) / 10.0) * (1 + 0.5 * humidity_factor) + np.random.normal(0, 0.1)
            vote = self._vote_from_temp(init_temp)
            passengers.append({
                "position": np.array([x, y], dtype=np.float32),
                "perceived_temp": np.clip(init_temp, self.temp_min, self.temp_max),
                "vote": vote
            })
        return passengers

    def _vote_from_temp(self, t):
        th = self.vote_thresholds
        if t < th[0]: return -2
        elif t < th[1]: return -1
        elif t < th[2]: return 0
        elif t < th[3]: return 1
        else: return 2

    def _estimate_room_temp(self):
        alpha = 0.8 * self.ac_fan / max(FAN_VALUES)
        return self.outside_temp * (1 - alpha) + self.ac_temp * alpha

    def _get_state(self):
        base = [self.current_time, self.outside_temp, self.outside_humidity,
                self.ac_temp_idx, self.ac_fan_idx, *self.ac_position]
        ps = []
        for p in self.passengers:
            ps.extend(p["position"].tolist())
            ps.append(p["perceived_temp"])
        while len(ps) < self.max_passengers * 3:
            ps.extend([0.0, 0.0, 0.0])
        return np.array(base + ps, dtype=np.float32)

    def _calculate_reward(self):
        votes = [p["vote"] for p in self.passengers]
        total = len(votes)

        comfort_ratio = sum(1 for p in self.passengers if abs(p["perceived_temp"]) <= 0.5) / total
        energy_penalty = (30.0 - self.ac_temp) / 12.0 + (self.ac_fan - 0.5)
        overreaction = (abs(self.ac_temp_idx - self.prev_ac_temp_idx) + abs(self.ac_fan_idx - self.prev_ac_fan_idx)) / 2.0

        too_hot_penalty = sum((p["perceived_temp"] - self.target_temp) ** 2 for p in self.passengers if  self._vote_from_temp(p["perceived_temp"]) > 0 )
        too_cold_penalty = sum((p["perceived_temp"] - self.target_temp) ** 2 for p in self.passengers if  self._vote_from_temp(p["perceived_temp"]) < 0 )

        reward = (1.0 * comfort_ratio - 0.2 * energy_penalty - 0.2 * overreaction -
                  0.02 * too_hot_penalty - 0.05 * too_cold_penalty)

        return reward, comfort_ratio, None, energy_penalty, overreaction
