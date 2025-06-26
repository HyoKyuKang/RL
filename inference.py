"""
inference.py ― SubwayCooling PPO (위치 + 체류시간 버전)
────────────────────────────────────────────────────────
state = [time, passengers, comfort, x_pos, y_pos, enter_time]
  · x_pos, y_pos ∈ [0,1]    (0.5,0.5 = 객실 중앙)
  · enter_time   ∈ [0,60]   (분 단위 체류 시간)
VecNormalize 통계 복원 → 학습 시 정규화 일치
무작위 60개(기본) 상태로 추론 → 액션 빈도 요약
"""

import os
import argparse
import numpy as np
from collections import Counter

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from subway_env_v2 import SubwayCoolingEnv   # enter_time 포함된 최신 v2

# ── 매핑 상수 ───────────────────────────────────────────
TEMP_MIN, TEMP_MAX = 18.0, 30.0
FAN_LEVELS = ["약", "약중", "중", "중강", "강"]

MODEL_PATH   = "models/ppo_subway_model_latest"
VECNORM_PATH = f"{MODEL_PATH}_vecnorm.pkl"

# ── 환경 / 모델 로드 ────────────────────────────────────
def _make_env():
    return SubwayCoolingEnv()

def load_model_and_env():
    base = DummyVecEnv([_make_env])
    if os.path.isfile(VECNORM_PATH):
        env = VecNormalize.load(VECNORM_PATH, base)
        env.training, env.norm_reward = False, False
    else:
        print("⚠️ VecNormalize 통계 누락 → 정규화 없이 진행")
        env = base
    model = PPO.load(MODEL_PATH, env=env, print_system_info=False)
    return model, env

model, env = load_model_and_env()

# ── 단일 추론 함수 ─────────────────────────────────────
def run_inference(state: np.ndarray, deterministic: bool = True):
    """
    state shape=(6,)
      [time, passengers, comfort, x_pos, y_pos, enter_time]
    returns: (temp_level, fan_level)
    """
    obs = state.reshape(1, -1)
    if isinstance(env, VecNormalize):
        obs = env.normalize_obs(obs)

    action, _ = model.predict(obs, deterministic=deterministic)
    tlv, flv = map(int, action[0])

    real_temp = TEMP_MIN + (tlv / 12) * (TEMP_MAX - TEMP_MIN)
    real_fan  = FAN_LEVELS[flv]

    print(
        f"[t={state[0]:02.0f}, pax={state[1]:3.0f}, "
        f"c={state[2]:+5.2f}, x={state[3]:.2f}, y={state[4]:.2f}, "
        f"stay={state[5]:4.0f}m] "
        f"→ {real_temp:4.1f}℃(lv{tlv:2}), fan {real_fan}({flv})"
    )
    return tlv, flv

# ── 메인: 무작위 테스트 ─────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=60,
                        help="무작위 테스트 케이스 수 (default=60)")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    hist = Counter()

    for _ in range(args.num):
        state = np.array([
            rng.integers(0, 24),      # time
            rng.integers(10, 101),    # passengers
            rng.uniform(-1.0, 1.0),   # comfort
            rng.uniform(0.0, 1.0),    # x_pos
            rng.uniform(0.0, 1.0),    # y_pos
            rng.uniform(0.0, 60.0),   # enter_time (minutes)
        ], dtype=np.float32)

        tlv, flv = run_inference(state)
        hist[(tlv, flv)] += 1

    # ── 액션 빈도 요약 -----------------------------------------------------
    print("\n―――――― Action frequency ――――――")
    for (tlv, flv), cnt in hist.most_common():
        print(f"temp_lv {tlv:2}, fan_lv {flv} : {cnt}회")
    print("총 학습 스텝 =", model.num_timesteps)