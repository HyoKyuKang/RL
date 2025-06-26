from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from subway_demo_env import SubwayCoolingEnv
import os

# 1) 벡터 환경 + 자동 정규화
def make_env():
    return Monitor(SubwayCoolingEnv())

train_env = DummyVecEnv([make_env])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                         clip_reward=10.0)     # ★ 리턴 스케일 자동 0~±10

eval_env  = DummyVecEnv([make_env])
eval_env  = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

# 2) 하이퍼파라미터
policy_kwargs = dict(net_arch=[256, 256])      # 값 예측 용량↑

model = PPO(
    "MlpPolicy",
    train_env,
    n_steps      = 4096,        # rollout 길이↑ → GAE 분산↓
    gamma        = 0.995,
    learning_rate= 1e-4,        # 느린 LR로 진동 방지
    vf_coef      = 1.0,         # value_loss 비중↑
    clip_range_vf= None,        # value clipping 해제
    tensorboard_log="./tb_logs",
    verbose=1,
    policy_kwargs=policy_kwargs,
)

# 3) 콜백(기존과 동일)
checkpoint_cb = CheckpointCallback(save_freq=50_000, save_path="./checkpoints", name_prefix="ppo_subway")
eval_cb = EvalCallback(eval_env, best_model_save_path="./best_model",
                       log_path="./logs", eval_freq=10_000, deterministic=True)
callback = CallbackList([checkpoint_cb, eval_cb])

model.learn(total_timesteps=1_000_000, callback=callback)
model.save("ppo_subway_v3")

# ───── 추론 시 ─────
# eval_env.training = False;  eval_env.norm_reward = False