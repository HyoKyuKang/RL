import gymnasium as gym
from gymnasium import spaces
import numpy as np

#TEMP_VALUES = list(np.arange(18.0, 31.0, 1.0))
#FAN_VALUES = [0.5, 1.0, 1.5]

class SubwayCoolingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        max_votes: int = 10,
        ac_position: np.ndarray = np.array([0.0, 0.0]),
        perceived_temp_limits: tuple = (10, 40),
        target_temp: float = 25.0
    ):
        super().__init__()
        self.ac_temp = 25
        self.ac_fan = 2
        self.prev_ac_temp = self.ac_temp
        self.prev_ac_fan = self.ac_fan

        self.max_votes = max_votes
        self.ac_position = np.array(ac_position, dtype=np.float32)
        self.temp_min, self.temp_max = perceived_temp_limits
        self.target_temp = target_temp
        self.passengers_num = 0

        self.action_space = spaces.MultiDiscrete([5, 5])  # delta -2~+2 for both temp and fan

        dim = 4 + 2 + max_votes * 4

        # 외부 온도 / 외부 습도 / 에어컨 설정 온도 / 에어컨 설정 풍량 / 에어컨 위치 (x,y) / 사람들 위치 (x,y) [-7~7][-1~1] / 들어온지얼마(0~5) /평가 (-2~2) 
        low = np.array([10.0, 0.0, 18, 0] + list(self.ac_position) +  [-7, -1, 0, -2] * max_votes, dtype=np.float32)
        high = np.array([40.0, 100.0, 31, 4] + list(self.ac_position) + [7, 1, 5, 2] * max_votes, dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.max_steps = 45
        self.vote_thresholds = [23, 25, 27, 28]
        self.reset()

    def reset(
        self, *,
        seed: int | None = None,
        options=None,
        ac_temp: int | None = None,
        ac_fan: int  | None = None,
        outside_temp: float | None = None,
        outside_humidity: float | None = None,
        passengers_num: float | None = None
    ):
        """
        에피소드(1회 시뮬레이션) 초기화

        Parameters
        ----------
        ac_temp, ac_fan          : 초기 에어컨 온도(18‥31 °C), 풍량(0‥4단)
        outside_temp, humidity   : 초기 외기 조건
        passengers_num           : 탑승객 수 (실수; 내부 로직에선 연속값 사용)
        seed                     : 난수 시드
        options                  : Gym 표준 매개변수 (미사용)

        * 각 매개변수가 None 이면 아래 범위에서 난수로 결정됩니다.
        - ac_temp            : randint(18, 31)   → 18‥30 °C
        - ac_fan             : randint(0, 5)     → 0‥4 단
        - outside_temp       : uniform(10, 40)   → 10‥40 °C
        - outside_humidity   : uniform(0, 100)   →   0‥100 %
        - passengers_num     : uniform(10, 150)  → 10‥150 명
        """
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # ───────── None → 난수 / 기본값 ─────────
        self.ac_temp = np.random.randint(18, 31) if ac_temp  is None else int(ac_temp)
        self.ac_fan  = np.random.randint(0, 5)   if ac_fan   is None else int(ac_fan)

        self.outside_temp     = (np.random.uniform(10, 40)
                                if outside_temp     is None else float(outside_temp))
        self.outside_humidity = (np.random.uniform(0, 100)
                                if outside_humidity is None else float(outside_humidity))

        self.passengers_num   = (np.random.uniform(10, 150)
                                if passengers_num  is None else float(passengers_num))

        # 이전 설정(변동 패널티 계산용) -- 처음엔 같은 값으로 맞춤
        self.prev_ac_temp = self.ac_temp
        self.prev_ac_fan  = self.ac_fan

        # 기타 상태 리셋
        self.current_step  = 0
        self.total_energy  = 0.0
        self.room_temp     = 25.0
        self.comfort_votes = self._generate_votes()

        return self._get_state(), {}


    def step(self, action):
        temp_delta = action[0] - 2
        fan_delta = action[1] - 2

        

        # action 취하기 --> 에어컨 온도 바꿈
        self.ac_temp = int(np.clip(self.ac_temp + temp_delta, 18, 31))
        self.ac_fan = int(np.clip(self.ac_fan + fan_delta, 0, 4))

        self.current_step += 1
        temp_diff = abs(self.outside_temp - self.ac_temp)
        fan_factor = self.ac_fan / 5
        cooling_power = temp_diff * fan_factor

        self.total_energy += cooling_power * 0.1 #0.1 scailing facotr

        # 에어컨 온도 바꿨으니, 실내 온도 새로 계산
        self.room_temp = self._estimate_room_temp()
    
        # 승객들의 체감온도를 다시 계산 --> 그 체감온도에 대한 투표(쾌적 지수)
        for p in self.comfort_votes:
            dist = np.linalg.norm(p["position"] - self.ac_position)
            attenuation = np.exp(-dist / 1)

            delta = (
                (self.room_temp - p["perceived_temp"]) * 0.5 * (0.8+0.2*attenuation) # + np.random.normal(-0.03, 0.03)
            )
            p["perceived_temp"] = np.clip(p["perceived_temp"] + delta, self.temp_min, self.temp_max)
            p["vote"] = self._vote_from_temp(p["perceived_temp"])

        terminated = self.current_step >= self.max_steps
        truncated = False
        reward, comfort_ratio, _, energy_penalty, overreaction_penalty = self._calculate_reward()

        obs = self._get_state()
        info = {
            "votes": [p["vote"] for p in self.comfort_votes],
            "positions": [p["position"].tolist() for p in self.comfort_votes],
            "mean_vote": np.mean([p["vote"] for p in self.comfort_votes]),
            "comfort_ratio": comfort_ratio,
            "energy_penalty": energy_penalty,
            "overreaction_penalty": overreaction_penalty,
            "reward": reward,
            "ac_temp": self.ac_temp,
            "ac_fan": self.ac_fan,
            "passengers_num": self.passengers_num
        }
        return obs, reward, terminated, truncated, info

    def _generate_votes(self):
        passengers = []
        n = np.random.randint(3, self.max_votes + 1)
        for _ in range(n):
            x, y,entertime = np.random.uniform(-7, 7), np.random.uniform(-1, 1.0),np.random.uniform(0,5)
            humidity_factor = self.outside_humidity / 100.0

            strain = np.linalg.norm(np.array([x, y]) - self.ac_position)
            atten  = np.exp(-strain / 1)           
            ac_effect = self.ac_temp * self.ac_fan * atten
            init_temp =0.9* ( self.room_temp * (entertime/5) + self.outside_temp *((5-entertime)/5) ) + 0.1 * ac_effect

            vote = self._vote_from_temp(init_temp)

            passengers.append({
                "position": np.array([x, y], dtype=np.float32),
                "perceived_temp": np.clip(init_temp, self.temp_min, self.temp_max),
                "entertime": entertime,
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
        return (1+self.passengers_num/600)*self.ac_temp
        # return self.outside_temp * (1 - alpha) + self.ac_temp * alpha
    # def _estimate_room_temp(self):
    #     # α: 냉방효율, β: 외기열부하, γ: 인원열부하
    #     α = 0.7
    #     β = 0.3
    #     γ = 0.0008  # 승객 1명당 0.8 %
    #     mix = α * self.ac_temp + β * self.outside_temp
    #     return mix + γ * self.passengers_num

    def _get_state(self):
        base = [self.outside_temp, self.outside_humidity,
                self.ac_temp, self.ac_fan, *self.ac_position]
        ps = []
        for p in self.comfort_votes:
            ps.extend(p["position"].tolist())
            #ps.append(p["perceived_temp"])
            ps.append(p["entertime"])
            ps.append(p["vote"])
            
        while len(ps) < self.max_votes * 4:
            ps.extend([0.0, 0.0, 0.0, 0.0])
        return np.array(base + ps, dtype=np.float32)

    def _calculate_reward(self):
        votes = [p["vote"] for p in self.comfort_votes]
        total = len(votes)

        comfort_ratio = sum(1 for p in self.comfort_votes if abs(p["vote"]) == 0) / total
        
        temp_diff = abs(self.outside_temp - self.ac_temp)
        fan_factor = self.ac_fan / 5
        cooling_power = temp_diff * fan_factor
        energy_penalty = 0.1 * cooling_power

        overreaction = (0.9 * abs(self.ac_temp - self.prev_ac_temp) + 0.1 * abs(self.ac_fan - self.prev_ac_fan)) 

        too_hot_penalty = sum((p["perceived_temp"] - self.target_temp) ** 2 for p in self.comfort_votes if  self._vote_from_temp(p["perceived_temp"]) > 0 )
        too_cold_penalty = sum((p["perceived_temp"] - self.target_temp) ** 2 for p in self.comfort_votes if  self._vote_from_temp(p["perceived_temp"]) < 0 )
        # reward = (1.0 * comfort_ratio - 0.2 * energy_penalty - 0.2 * overreaction)
        
        reward = (
            1.75 * comfort_ratio
            - 0.2 * energy_penalty
            - 0.075 * overreaction
            - 0.1 * too_hot_penalty
            - 0.01 * too_cold_penalty
        )
        
        return reward, comfort_ratio, None, energy_penalty, overreaction
