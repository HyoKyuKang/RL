"""
inference.py ― SubwayCooling PPO 모델 다중-테스트 스크립트
──────────────────────────────────────────────────────────
• VecNormalize 복원 → 관측치 정규화 일관성 유지
• 60개(기본) 무작위 상태를 생성해 일괄 추론
    - 시간   : 0 ~ 23   (정수)
    - 승객수 : 10 ~ 100 (러시 아워 포함 범위)
    - comfort: 균등분포 U[-1,1]
• 추론 결과를 한 줄씩 출력하고, 마지막에 액션 빈도 요약
"""

import os
import argparse
import numpy as np
from collections import Counter
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from subway_env_v2 import SubwayCoolingEnv   # ← 최신 환경 필요

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
    state: shape=(3,)  [time, passengers, comfort]
    returns: (temp_level, fan_level)
    """
    if isinstance(env, VecNormalize):
        obs = env.normalize_obs(state.reshape(1, -1))
    else:
        obs = state.reshape(1, -1)

    action, _ = model.predict(obs, deterministic=deterministic)
    tlv, flv = map(int, action[0])

    real_temp = TEMP_MIN + (tlv / 12) * (TEMP_MAX - TEMP_MIN)
    real_fan  = FAN_LEVELS[flv]

    print(f"[time={state[0]:.0f}, passengers={state[1]:.0f}, "
          f"comfort={state[2]:+.2f}]  "
          f"→ temp {real_temp:4.1f} ℃ (lv {tlv:2}), "
          f"fan {real_fan} ({flv}단)")
    return tlv, flv

# ── 메인: 무작위 테스트 ─────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=60,
                        help="test case 수 (default=60)")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    action_hist = Counter()

    for _ in range(args.num):
        t  = rng.integers(0, 24)
        p  = rng.integers(10, 101)               # 10-100명
        c  = rng.uniform(-1.0, 1.0)              # comfort
        s  = np.array([t, p, c], dtype=np.float32)
        tlv, flv = run_inference(s)
        action_hist[(tlv, flv)] += 1

    # ── 요약 통계 ───────────────────────────────────────
    print("\n―――――― Action frequency ――――――")
    for (tlv, flv), cnt in action_hist.most_common():
        print(f"temp_lv {tlv:2}, fan_lv {flv} : {cnt}회")
    print("총 학습 스텝 =", model.num_timesteps)