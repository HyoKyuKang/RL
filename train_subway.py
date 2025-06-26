from stable_baselines3 import PPO
from subway_env import SubwayCoolingEnv  # 파일명이 subway_env.py가 아닌 경우 수정
from stable_baselines3.common.env_checker import check_env
from gymnasium.wrappers import FlattenObservation
import os

# 1. 환경 생성 및 flatten
base_env = SubwayCoolingEnv()
env = FlattenObservation(base_env)

# 2. 환경 검증 (디버깅용)
check_env(env, warn=True)

# 3. PPO 모델 생성
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log="./tensorboard_logs",  # 선택: 학습 로그 저장
)

# 4. 학습 실행
model.learn(total_timesteps=100_000)

# 5. 모델 저장
os.makedirs("models", exist_ok=True)
model.save("models/ppo_subway_model")

print("✅ 학습 완료 및 모델 저장 완료.")