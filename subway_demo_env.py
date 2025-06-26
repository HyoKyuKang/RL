import gymnasium as gym
from gymnasium import spaces
import numpy as np

# AC 온도/풍량 값 (index → 실제 값)
TEMP_VALUES = list(np.arange(18.0, 31.0, 1.0))  # 18.0 ~ 30.0
FAN_VALUES = [0.5, 1.0, 1.5]

class SubwayCoolingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        max_passengers: int = 10,
        ac_position: np.ndarray = np.array([2.5, 0.0]),
        sigma: float = 1.0,
        perceived_temp_limits: tuple = (-1.5, 1.5),
        target_temp: float = 25.0
    ):
        super().__init__()
        self.max_passengers = max_passengers
        self.ac_position = np.array(ac_position, dtype=np.float32)
        self.sigma = sigma
        self.temp_min, self.temp_max = perceived_temp_limits
        self.target_temp = target_temp

        self.action_space = spaces.MultiDiscrete([len(TEMP_VALUES), len(FAN_VALUES)])

        dim = 5 + 2 + max_passengers * 3
        low = np.array([0.0, 0.0, 0.0, 0.0, 0.0] + 
                       list(self.ac_position) + 
                       [0.0, 0.0, -1.5] * max_passengers,
                       dtype=np.float32)
        high = np.array([23.0, 40.0, 100.0, len(TEMP_VALUES)-1, len(FAN_VALUES)-1] + 
                        list(self.ac_position) + 
                        [5.0, 5.0, 1.5] * max_passengers,
                        dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.max_steps = 60
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
        self.room_temp = self.ac_temp
        self.passengers = self._generate_passengers()

        return self._get_state(), {}

    def step(self, action):
        self.prev_ac_temp_idx = self.ac_temp_idx
        self.prev_ac_fan_idx = self.ac_fan_idx

        self.ac_temp_idx, self.ac_fan_idx = action
        self.ac_temp = TEMP_VALUES[self.ac_temp_idx]
        self.ac_fan = FAN_VALUES[self.ac_fan_idx]

        self.current_step += 1
        self.total_energy += (self.ac_temp - 18.0) / 12.0 + (self.ac_fan - 0.5) / 1.0
        self.current_time = (self.current_time + 1) % 24

        self.room_temp = self._estimate_room_temp()

        for p in self.passengers:
            dist = np.linalg.norm(p["position"] - self.ac_position)
            attenuation = np.exp(-dist / self.sigma)

            delta = (
                (self.room_temp - self.target_temp) * -0.2 * attenuation +
                #(self.outside_temp - self.target_temp) * 0.02 * (1 - attenuation) +
                np.random.normal(0, 0.05)
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
            x = np.random.uniform(0.0, 5.0)
            y = np.random.uniform(0.0, 5.0)
            init_temp = (self.room_temp - self.target_temp) + np.random.normal(0, 0.2)
            vote = self._vote_from_temp(init_temp)
            passengers.append({
                "position": np.array([x, y], dtype=np.float32),
                "perceived_temp": np.clip(init_temp, self.temp_min, self.temp_max),
                "vote": vote
            })
        return passengers

    def _vote_from_temp(self, t):
        if t < -0.7:
            return -2
        elif t < -0.3:
            return -1
        elif t < 0.3:
            return 0
        elif t < 0.7:
            return 1
        else:
            return 2

    def _estimate_room_temp(self):
        return self.ac_temp
        #ac_power = (self.ac_temp - 18.0) / 12.0 * (self.ac_fan / 1.5)
        #weight_ac = min(1.0, 0.9 * ac_power)
        #weight_out = 1.0 - weight_ac
        #return self.target_temp * weight_ac + self.outside_temp * weight_out

    def _get_state(self):
        base = [
            self.current_time,
            self.outside_temp,
            self.outside_humidity,
            self.ac_temp_idx,
            self.ac_fan_idx,
            *self.ac_position
        ]
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

        comfort_ratio = votes.count(0) / total
        energy_penalty = (self.ac_temp - 18.0) / 12.0 + (self.ac_fan - 0.5) / 1.0

        overreaction = (
            abs(self.ac_temp_idx - self.prev_ac_temp_idx) +
            abs(self.ac_fan_idx - self.prev_ac_fan_idx)
        ) / 2.0

        reward = (
            +1.0 * comfort_ratio
            -0.2 * energy_penalty
            -0.2 * overreaction
        )
        reward = np.clip(reward, -2.0, 1.0)

        return reward, comfort_ratio, None, energy_penalty, overreaction
