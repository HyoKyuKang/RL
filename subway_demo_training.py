from subway_demo_env import SubwayCoolingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# 환경 생성
env = SubwayCoolingEnv()

# 환경 체크 (에러 확인용)
check_env(env, warn=True)

# 모델 생성
model = PPO("MlpPolicy", env, verbose=1)

# 학습 시작t
model.learn(total_timesteps=100_000)

# 학습 후 테스트
obs, _ = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, _, info = env.step(action)
    print(f"Step: {env.current_step}, Reward: {reward:.3f}, Mean Vote: {info['mean_vote']:.2f}")
