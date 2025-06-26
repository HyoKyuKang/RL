"""
subway_env_v2.py  
────────────────────────────────────────────────────────
위치·체류시간·외부기상 + 관성 + 변화-속도 + 역방향 패널티
+ **현실적 에너지 모델**(무냉방 객실온도 기반) 버전
────────────────────────────────────────────────────────
state = [time, passengers, comfort, x_pos, y_pos,
         enter_time, ext_temp, ext_hum,
         cur_temp_lv, cur_fan_lv]

● comfort < 0  (춥다) → temp_lv ↑(따뜻) · fan_lv ↓(약)
● comfort > 0  (덥다) → temp_lv ↓(시원) · fan_lv ↑(강)

에너지 비용 =  
  냉방비(무냉방온도-현재온도)·0.8  +  난방비(현재온도-무냉방온도)·0.5  
  + 풍량비(cur_fan_lv·0.3)

무냉방 객실온도 ≈ ext_temp + 0.01·passengers (단순 선형 가정)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from collections import Counter


class SubwayCoolingEnv(gym.Env):
    # ── 보상 계수 --------------------------------------------------------
    K_DIR, K_COMFORT, K_ENERGY, K_OVER = -0.25, -0.5, -0.1, -0.2
    K_RATE  = -0.3     # 변화-속도
    K_WRONG = -1.0     # 역방향(미스매치) 패널티
    MAX_STAY = 60.0
    ALPHA = 0.2        # set-point → 실내공기 추종 속도

    def __init__(self, max_steps: int = 60):
        super().__init__()

        low  = np.array([0, 0, -1.0, 0.0, 0.0,  0.0, -10.0,   0.0, 0, 0], dtype=np.float32)
        high = np.array([23,100,  1.0, 1.0, 1.0, 60.0,  40.0,100.0,12, 4], dtype=np.float32)
        self.observation_space = spaces.Box(low, high)
        self.action_space = spaces.MultiDiscrete([13, 5])      # set-point (temp_lv, fan_lv)

        self.max_steps = max_steps
        self.np_random, _ = seeding.np_random(None)
        self.reset()

    # -------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)

        self.current_step   = 0
        self.num_comfort    = 0
        self.total_energy   = 0.0
        self.total_overreaction = 0

        # 초기 상태 (24 ℃·3단 → cur_lv=(6,2))
        self.state = np.array(
            [8, 50, 0.0, 0.5, 0.5, 0.0, 25.0, 50.0, 6.0, 2.0],
            dtype=np.float32
        )
        return self.state, {}

    # -------------------------------------------------------------------
    def step(self, action):
        (t, pax, comfort, x, y, stay,
         ext_t, ext_h, cur_tlv, cur_flv) = self.state
        set_tlv, set_flv = map(int, action)        # 새 set-point

        # ── 관성: 실내공기 값 업데이트 -----------------------------------
        cur_tlv += self.ALPHA * (set_tlv - cur_tlv)
        cur_flv += self.ALPHA * (set_flv - cur_flv)

        temp = 18 + cur_tlv               # °C
        fan  = 1  + cur_flv               # 단계

        # 위치·체류 가중치
        w_pos  = 2.0 * max(abs(x - 0.5), abs(y - 0.5))
        w_time = max(1.0 - stay / 30.0, 0.0)

        # 외기 영향
        w_weather = 0.002 * (ext_t - 24.0) + 0.001 * (ext_h - 50.0)

        # comfort 변화
        d_c = self._simulate_comfort_change(pax, temp, fan)
        comfort = np.clip(comfort + d_c * (1 + w_pos) * (1 + w_time) + w_weather,
                          -1.0, 1.0)
        if abs(comfort) < 0.3:
            self.num_comfort += 1

        # ── 에너지 비용 --------------------------------------------------
        no_ac_temp = self._predict_no_ac_temp(ext_t, pax)   # 무냉방 객실온도
        delta_cool = max(0.0, no_ac_temp - temp)            # 냉방
        delta_heat = max(0.0, temp - no_ac_temp)            # 난방
        temp_cost  = 0.8 * delta_cool + 0.5 * delta_heat
        fan_cost   = 0.3 * cur_flv
        energy     = temp_cost + fan_cost
        self.total_energy += energy

        # 과도한 난방/냉방 정의
        over = int(delta_cool >= 6.0) + int(delta_heat >= 6.0) + int(fan >= 5)
        self.total_overreaction += over

        # ── 변화-속도 패널티 -------------------------------------------
        d_temp = abs(set_tlv - cur_tlv)
        d_fan  = abs(set_flv - cur_flv)
        rate_penalty = self.K_RATE * (d_temp + 0.5 * d_fan)

        # ── 역방향 패널티 ----------------------------------------------
        wrong_dir = 0
        if comfort < -0.05:                          # 춥다 → temp ↑ fan ↓
            if set_tlv < cur_tlv: wrong_dir += 1
            if set_flv > cur_flv: wrong_dir += 1
        elif comfort > 0.05:                         # 덥다 → temp ↓ fan ↑
            if set_tlv > cur_tlv: wrong_dir += 1
            if set_flv < cur_flv: wrong_dir += 1
        wrong_penalty = self.K_WRONG * wrong_dir

        # ── 방향성 보상 -----------------------------------------------
        ideal_tlv, ideal_flv = self._comfort_to_target(comfort)
        mismatch = abs(ideal_tlv - set_tlv) + 0.5 * abs(ideal_flv - set_flv)
        dir_reward = self.K_DIR * mismatch * (1 + abs(comfort))

        reward = (dir_reward
                  + self.K_COMFORT * comfort * comfort
                  + self.K_ENERGY  * energy        # 새 에너지 모델
                  + self.K_OVER    * over
                  + rate_penalty
                  + wrong_penalty)

        # ── 다음 상태 --------------------------------------------------
        self.current_step += 1
        t   = (t + 1) % 24
        pax = self._simulate_passengers(t)

        x = np.clip(x + self.np_random.uniform(-0.05, 0.05), 0, 1)
        y = np.clip(y + self.np_random.uniform(-0.05, 0.05), 0, 1)
        stay = min(stay + 1.0, self.MAX_STAY)
        ext_t = np.clip(ext_t + self.np_random.normal(0, 0.5), -10, 40)
        ext_h = np.clip(ext_h + self.np_random.normal(0, 2.0),   0, 100)

        self.state = np.array(
            [t, pax, comfort, x, y, stay, ext_t, ext_h, cur_tlv, cur_flv],
            dtype=np.float32
        )

        terminated = False
        truncated  = self.current_step >= self.max_steps
        if truncated:
            comfort_ratio = self.num_comfort / self.max_steps
            reward += (+1.0 * comfort_ratio
                       + self.K_ENERGY * (self.total_energy / self.max_steps)
                       + self.K_OVER   * (self.total_overreaction / self.max_steps))

        return self.state, float(reward), terminated, truncated, {}

    # =================== 헬퍼들 ========================================
    def _simulate_passengers(self, tt):
        tt = int(tt)
        return self.np_random.integers(60, 100) if 7 <= tt <= 9 or 17 <= tt <= 19 \
               else self.np_random.integers(10, 50)

    @staticmethod
    def _simulate_comfort_change(pax, temp, fan):
        temp_eff = +0.015 * pax * (temp - 24.0)
        fan_eff  = -0.01  * pax * (fan  - 3.0)
        return 0.01 * (temp_eff + fan_eff)

    @staticmethod
    def _predict_no_ac_temp(ext_t, pax):
        """외기온·승객열을 고려한 ‘무냉방’ 객실온도 간단 추정식"""
        return ext_t + 0.01 * pax        # 1명당 +0.01 ℃

    @staticmethod
    def _comfort_to_target(c):
        return (int(round(np.clip(24 - 4 * c, 18, 30) - 18)),
                int(round(np.clip(3 + 2 * c, 1, 5) - 1)))