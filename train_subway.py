from stable_baselines3 import PPO
from subway_env import SubwayCoolingEnv
from stable_baselines3.common.env_checker import check_env

# 환경 생성
env = SubwayCoolingEnv()

# 환경 검증 (선택)
check_env(env, warn=True)

# PPO 모델 생성
model = PPO("MlpPolicy", env, verbose=1)

# 학습 실행
model.learn(total_timesteps=20000)

# 모델 저장
model.save("ppo_subway_model")

print("학습 완료 및 모델 저장 완료.")
