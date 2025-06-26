from stable_baselines3 import PPO
import numpy as np

# 온도 및 풍량 매핑 정의
temp_min, temp_max = 18.0, 30.0
fan_levels = ["약", "약중", "중", "중강", "강", "매우강"]

# 모델 불러오기
model = PPO.load("ppo_subway_model.zip")

# 테스트 상태: [시간, 승객 수, 쾌적도]
example_state = np.array([9, 70, 1], dtype=np.float32).reshape(1, -1)

# 예측 실행
action, _ = model.predict(example_state, deterministic=True)
temp_level, fan_level = action[0]

# 실제 값으로 변환
real_temp = temp_min + (temp_level / 5.0) * (temp_max - temp_min)
real_fan = fan_levels[fan_level]

# 출력
print("입력 상태:", example_state.flatten().tolist())
print(f"추천 조치 → 에어컨 온도: {real_temp:.1f}도, 풍량: {real_fan} ({fan_level}단계)")