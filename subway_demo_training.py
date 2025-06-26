from subway_demo_env import SubwayCoolingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
import os

env = Monitor(SubwayCoolingEnv())

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tb_logs")

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

callback = CallbackList([checkpoint_callback, eval_callback])

model.learn(
    total_timesteps=150_000,
    callback=callback
)

model.save("ppo_subway_v3")
