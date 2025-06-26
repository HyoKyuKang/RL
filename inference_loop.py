from stable_baselines3 import PPO
import numpy as np

model = PPO.load("ppo_subway_model.zip")

state = np.array([7, 70, -0.3], dtype=np.float32)
total_reward = 0

print("시간\t승객\t쾌적도\t→ Action\tReward")
for i in range(10):
    action, _ = model.predict(state, deterministic=True)
    # 임의의 간단한 comfort 변화 시뮬레이션
    comfort_delta = -0.01 * state[1] * abs(action[0] - 1)  # deviation 영향
    comfort = np.clip(state[2] + comfort_delta, -1.0, 1.0)

    reward = 1 - abs(comfort) - 0.1 * sum(action)
    total_reward += reward

    print(f"{int(state[0])}\t{int(state[1])}\t{state[2]:+.2f}\t→ {action}\t{reward:.2f}")

    # 상태 업데이트
    state[0] = (state[0] + 1) % 24  # 시간
    state[1] = np.random.randint(60, 100) if 7 <= state[0] <= 9 or 17 <= state[0] <= 19 else np.random.randint(10, 50)
    state[2] = comfort

print(f"\n총 보상: {total_reward:.2f}")
