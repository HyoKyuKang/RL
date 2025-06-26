"""
inference_seq_votes.py – real‑time roll‑out for SubwayCoolingEnv (minute resolution)
──────────────────────────────────────────────────────────────────────────────
• Works with **subway_env_v2_votes.SubwayCoolingEnv** (12‑dim state, minute time).
• Prints HH:MM timestamp instead of raw minute index.
• All indices unchanged: cur_tlv = state[10], cur_flv = state[11].
"""

import os
import argparse
from collections import Counter

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from subway_env_v2 import SubwayCoolingEnv  # ← updated import

TEMP_MIN, TEMP_MAX = 18.0, 30.0
FAN_KOR = ["약", "약중", "중", "중강", "강"]

MODEL_PATH = "models/ppo_subway_model_latest"
VEC_PATH = f"{MODEL_PATH}_vecnorm.pkl"


# ──────────────────────────────────────────────────────────

def make_env():
    return SubwayCoolingEnv(max_steps=1_000_000)


base = DummyVecEnv([make_env])
if os.path.isfile(VEC_PATH):
    env = VecNormalize.load(VEC_PATH, base)
    env.training = env.norm_reward = False
else:
    print("⚠️ VecNormalize stats not found – running unnormalized")
    env = base

model = PPO.load(MODEL_PATH, env=env, print_system_info=False)


# ──────────────────────────────────────────────────────────

def run_inference(state: np.ndarray, return_extra: bool = False):
    """Execute **one** step, return (set_tlv, set_flv [, energy, reward, next_state])."""

    # ① observation (normalize if needed)
    obs = state.reshape(1, -1)
    if isinstance(env, VecNormalize):
        obs = env.normalize_obs(obs)

    # ② agent prediction
    act, _ = model.predict(obs, deterministic=True)
    ask_tlv, ask_flv = map(int, np.asarray(act).squeeze())

    # ③ sync env state & step
    env.envs[0].state = state.copy()
    next_state, reward, *_ = env.envs[0].step([ask_tlv, ask_flv])

    # ④ applied set‑point
    set_tlv, set_flv = int(round(next_state[10])), int(round(next_state[11]))

    # ⑤ energy delta
    prev_total_e = getattr(env.envs[0], "_energy_prev", 0.0)
    step_energy = float(env.envs[0].total_energy - prev_total_e)
    env.envs[0]._energy_prev = env.envs[0].total_energy

    # ⑥ log pretty
    t_min = int(state[0])
    hh, mm = divmod(t_min, 60)
    time_str = f"{hh:02d}:{mm:02d}"
    real_temp = TEMP_MIN + (set_tlv / 12) * (TEMP_MAX - TEMP_MIN)
    print(
        "⏱ t={} | pax {:3.0f} | c {:+.2f} | xy({:.2f},{:.2f}) | "
        "stay {:4.1f}m | ext {:5.1f}℃ / {:5.1f}% | "
        "cur_lv {:2.0f}/{:1.0f} → ask lv{:2},fan{} | set {:4.1f}℃(lv{:2}), fan {}".format(
            time_str,
            state[1],  # passengers
            state[2],  # comfort_avg
            state[5], state[6],  # x, y
            state[7],  # stay
            state[8], state[9],  # ext T / H
            state[10], state[11],  # current tlv / flv
            ask_tlv, FAN_KOR[ask_flv],
            real_temp, set_tlv, FAN_KOR[set_flv],
        )
    )

    if return_extra:
        return set_tlv, set_flv, step_energy, float(reward), next_state
    return set_tlv, set_flv


# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, default=60, help="roll‑out steps")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # initial snapshot (minute‑based)
    state = np.array(
        [
            0,                       # time (min)
            rng.integers(30, 81),    # passengers
            rng.uniform(-1.0, 1.0),  # comfort_avg
            0.0,                     # comfort_std
            1.0,                     # ratio_comfy
            0.5, 0.5,                # x, y
            0.0,                     # stay
            25.0, 50.0,              # ext T/H
            8, 2,                    # cur tlv / flv
        ],
        dtype=np.float32,
    )

    hist = Counter()
    stats = dict(comfort_hit=0, wrong_dir=0, dtemp_sum=0.0, dfan_sum=0.0,
                 energy_sum=0.0, reward_sum=0.0)

    # simulation loop
    for _ in range(args.num):
        tlv, flv, en, rew, next_state = run_inference(state, return_extra=True)
        hist[(tlv, flv)] += 1

        stats["comfort_hit"] += int(abs(state[2]) < 0.3)
        wrong = (
            (state[2] < -0.05 and (tlv < state[10] or flv > state[11]))
            or (state[2] > 0.05 and (tlv > state[10] or flv < state[11]))
        )
        stats["wrong_dir"] += int(wrong)
        stats["dtemp_sum"] += abs(tlv - state[10])
        stats["dfan_sum"] += abs(flv - state[11])
        stats["energy_sum"] += en
        stats["reward_sum"] += rew

        state = next_state.copy()

    # summary
    print("\n―――――― Action frequency ――――――")
    for (tlv, flv), cnt in hist.most_common():
        print(f"temp_lv {tlv:2}, fan_lv {flv} : {cnt}회")
    print("총 학습 스텝 =", model.num_timesteps)

    n = args.num
    print("\n―――――― Quick metrics ――――――")
    print(f"Comfort in band   : {stats['comfort_hit']/n:6.1%}")
    print(f"Wrong‑dir ratio   : {stats['wrong_dir']/n:6.1%}")
    print(f"Δtemp avg (lv)    : {stats['dtemp_sum']/n:6.2f}")
    print(f"Δfan  avg (lv)    : {stats['dfan_sum']/n:6.2f}")
    print(f"Energy / step     : {stats['energy_sum']/n:6.2f}")
    print(f"Avg step reward   : {stats['reward_sum']/n:6.2f}")
