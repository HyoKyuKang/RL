import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SubwayCoolingEnv(gym.Env):
    def __init__(self):
        super(SubwayCoolingEnv, self).__init__()

        # --- 상태 공간: [현재 시간, 승객 수, 평균 쾌적도]
        self.observation_space = spaces.Box(low=np.array([0, 0, -1.0]),
                                            high=np.array([23, 100, 1.0]),
                                            dtype=np.float32)

        # --- 행동 공간: 온도 설정 (0~2단계), 풍량 설정 (0~2단계)
        self.action_space = spaces.MultiDiscrete([3, 3])  # 예: (온도, 풍량)

        # 내부 상태 초기화
        self.state = None
        self.current_step = 0
        self.max_steps = 60  # 1회 시뮬레이션은 60 step (예: 1시간)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # 예시 초기 상태: 오전 8시, 탑승객 50명, 중립 쾌적도
        self.state = np.array([8, 50, 0.0], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        time, passengers, comfort = self.state
        temp_setting, fan_setting = action

        self.current_step += 1

        # 쾌적도 업데이트: 단순 모델 (온도, 풍량, 인원수 기반)
        comfort_delta = self._simulate_comfort_change(passengers, temp_setting, fan_setting)
        comfort = np.clip(comfort + comfort_delta, -1.0, 1.0)

        # 보상 계산: 쾌적도 최대화 - 에너지 사용 최소화
        energy_penalty = 0.1 * (temp_setting + fan_setting)
        comfort_reward = 1 - abs(comfort)  # 0에 가까울수록 좋음
        reward = comfort_reward - energy_penalty

        # 시간 진행
        time = (time + 1) % 24
        passengers = self._simulate_passengers(time)

        self.state = np.array([time, passengers, comfort], dtype=np.float32)
        terminated = self.current_step >= self.max_steps

        return self.state, reward, terminated, False, {}

    def _simulate_passengers(self, time):
        # 시간대에 따른 탑승객 수 (예시)
        if 7 <= time <= 9 or 17 <= time <= 19:
            return np.random.randint(60, 100)  # 출퇴근 시간
        else:
            return np.random.randint(10, 50)

    def _simulate_comfort_change(self, passengers, temp_setting, fan_setting):
        # 매우 간단한 모델: 세팅이 낮으면 춥다, 높으면 덥다
        ideal_setting = 1  # 중간 세팅이 이상적이라고 가정
        deviation = abs(temp_setting - ideal_setting) + abs(fan_setting - ideal_setting)
        direction = -1 if temp_setting < ideal_setting else 1
        return -0.01 * passengers * deviation * direction  # 단순 불만 반영
