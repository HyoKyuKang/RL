from stable_baselines3 import PPO
import numpy as np

# 모델 불러오기
model = PPO.load("ppo_subway_model.zip")

# 테스트 입력: [현재 시간, 승객 수, 현재 쾌적도]
example_state = np.array([9, 70, -0.2], dtype=np.float32)

# 추론 실행
action, _ = model.predict(example_state, deterministic=True)

print("입력 상태:", example_state.tolist())
print("추천 조치 (온도, 풍량):", action.tolist())
