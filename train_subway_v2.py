import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env
from subway_env_v2 import SubwayCoolingEnv   # ← 수정한 환경

# 1. 환경 래핑 --------------------------------------------------------------
def make_env():
    env = SubwayCoolingEnv()
    check_env(env, warn=True)                # 한 번만 수행
    return env

v_env = DummyVecEnv([make_env])
v_env = VecNormalize(v_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# 2. 네트워크 크기 키우기 ---------------------------------------------------
policy_kwargs = dict(net_arch=[256, 256])

model = PPO(
    policy="MlpPolicy",
    env=v_env,
    verbose=1,
    tensorboard_log="./tb",
    n_steps=2048,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    policy_kwargs=policy_kwargs,
)

# 3. 학습 ------------------------------------------------------------------
model.learn(total_timesteps=500_000)         # 필요하면 더 늘리기

# 4. VecNormalize 스케일 정보까지 같이 저장
save_path = "models/ppo_subway_model_latest"
model.save(save_path)
v_env.save(f"{save_path}_vecnorm.pkl")

print("✅ 학습 및 저장 완료.")