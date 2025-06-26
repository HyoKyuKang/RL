from stable_baselines3 import PPO
import gymnasium as gym

# CartPole 환경 만들기
env = gym.make("CartPole-v1")

# 모델 생성 및 학습
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# 학습된 모델로 실행 테스트
obs, _ = env.reset()
for _ in range(200):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

# 모델 저장
model.save("ppo_cartpole_model")
