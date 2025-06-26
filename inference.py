"""
inference_seq.py ― 연속 시각 · 실시간 set-point 피드백 테스트
────────────────────────────────────────────────────────────
state = [time, passengers, comfort, x_pos, y_pos,
         enter_time, ext_temp, ext_hum,
         cur_temp_lv, cur_fan_lv]        # ← 10-dim

첫 스텝 set-point : 26 ℃(lv=8), 3단(lv=2)
각 step에서 나온 set-point를 다음 step의 cur_* 로 넘겨
N step(기본 60) 연속 시뮬레이션 후 set-point 빈도 요약.
"""

import os, argparse
import numpy as np
from collections import Counter
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from subway_env_v2 import SubwayCoolingEnv   # 10-dim 환경

TEMP_MIN, TEMP_MAX = 18.0, 30.0
FAN_KOR = ["약", "약중", "중", "중강", "강"]

MODEL_PATH   = "models/ppo_subway_model_latest"
VEC_PATH     = f"{MODEL_PATH}_vecnorm.pkl"

# ─────────────────────────────────────────────────────────
def make_env():
    return SubwayCoolingEnv()

base = DummyVecEnv([make_env])
if os.path.isfile(VEC_PATH):
    env = VecNormalize.load(VEC_PATH, base)
    env.training, env.norm_reward = False, False
else:
    print("⚠️ VecNormalize 통계 누락 → 정규화 없이 진행")
    env = base

model = PPO.load(MODEL_PATH, env=env, print_system_info=False)

# ─────────────────────────────────────────────────────────
def run_inference(state):
    """
    state: [t, pax, comfort, x, y, stay, extT, extH, cur_tlv, cur_flv]
    returns: 새 set-point (tlv, flv)
    """
    obs = state.reshape(1, -1)
    if isinstance(env, VecNormalize):
        obs = env.normalize_obs(obs)

    act, _ = model.predict(obs, deterministic=True)
    tlv, flv = map(int, np.asarray(act).squeeze())

    # 변환 & 출력 (모든 입력 변수 + 새 set-point)
    real_temp = TEMP_MIN + (tlv / 12) * (TEMP_MAX - TEMP_MIN)
    print(
        "⏱ t={:02.0f}h | pax {:3.0f} | c {:+.2f} | xy({:.2f},{:.2f}) | "
        "stay {:4.1f}m | ext {:5.1f}℃ / {:5.1f}% | "
        "cur_lv {:2.0f}/{:1.0f} → set {:4.1f}℃(lv{:2}), fan {:}".format(
            state[0],      # time
            state[1],      # passengers
            state[2],      # comfort
            state[3],      # x_pos
            state[4],      # y_pos
            state[5],      # enter_time
            state[6],      # ext_temp
            state[7],      # ext_hum
            state[8],      # cur_temp_lv
            state[9],      # cur_fan_lv
            real_temp, tlv, FAN_KOR[flv]   # 새 set-point
        )
    )
    return tlv, flv

# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, default=60, help="step 수")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # 초기 조건 ----------------------------------------------------------
    t0 = 0
    pax = rng.integers(30, 81)
    comfort = rng.uniform(-0.2, 0.2)
    x, y = 0.5, 0.5
    stay = 0.0
    ext_t, ext_h = 25.0, 50.0
    cur_tlv, cur_flv = 8, 2    # 26℃·3단

    hist = Counter()

    for k in range(args.num):
        t = (t0 + k) % 24
        state = np.array([t, pax, comfort, x, y,
                          stay, ext_t, ext_h,
                          cur_tlv, cur_flv], dtype=np.float32)

        tlv, flv = run_inference(state)
        hist[(tlv, flv)] += 1

        # 다음 step으로 현재 set-point 전달
        cur_tlv, cur_flv = tlv, flv

        # (예시) 환경 변수 약간 업데이트
        comfort = np.clip(comfort + rng.normal(0, 0.05), -1, 1)
        pax = int(np.clip(pax + rng.normal(0, 5), 10, 100))
        stay = min(stay + 1.0, 60.0)
        ext_t = np.clip(ext_t + rng.normal(0, 0.3), -10, 40)
        ext_h = np.clip(ext_h + rng.normal(0, 1.0), 0, 100)

    # 요약 --------------------------------------------------------------
    print("\n―――――― Action frequency ――――――")
    for (tlv, flv), cnt in hist.most_common():
        print(f"temp_lv {tlv:2}, fan_lv {flv} : {cnt}회")
    print("총 학습 스텝 =", model.num_timesteps)