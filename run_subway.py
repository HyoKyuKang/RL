from stable_baselines3 import PPO
from subway_env import SubwayCoolingEnv

model = PPO.load("ppo_subway_model")
env = SubwayCoolingEnv()

obs, _ = env.reset()
total_reward = 0

for step in range(10):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward

    print(f"Step {step + 1} - Action: {action}, Reward: {reward:.2f}")

    if done:
        print("에피소드 종료. 환경 재시작.\n")
        obs, _ = env.reset()

print(f"\n총 보상 (Total reward): {total_reward:.2f}")
