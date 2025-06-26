from subway_demo_env import SubwayCoolingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os

# 1. 환경 설정
env = SubwayCoolingEnv()
env = Monitor(env)  # 로그 저장을 위해 Monitor 래핑

# 2. 모델 설정
#model = PPO("MlpPolicy", env, verbose=1)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tb_logs")

# 3. 콜백: 정기 저장 + 성능 평가
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./checkpoints",
    name_prefix="ppo_subway"
)

eval_env = SubwayCoolingEnv()
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model",
    log_path="./logs",
    eval_freq=10000,
    deterministic=True,
    render=False
)

# 4. 학습 시작
model.learn(
    total_timesteps=1_000_00,  # 시간 늘림
    callback=[checkpoint_callback, eval_callback]
)

# 5. 최종 저장
model.save("ppo_subway_v3")
