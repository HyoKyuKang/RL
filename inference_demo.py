"""
ppo_subway_inference.py  –  SubwayCoolingEnv v2 추론 스크립트
──────────────────────────────────────────────────────────────
· action: MultiDiscrete([7, 5]) → ΔT(-2~+2), ΔF(-2~+2)
· env.info 에서 ac_temp, ac_fan, votes, mean_vote 등을 직접 읽어와 로그 작성
"""

from collections import Counter

from stable_baselines3 import PPO
from subway_demo_env import SubwayCoolingEnv

# ────────────────────── 투표 레이블/풍량 레이블 ──────────────────────
VOTE_LABELS = {
    -2: "매우춥다",
    -1: "약간춥다",
     0: "보통이다",
     1: "약간덥다",
     2: "매우덥다",
}

FAN_LABELS = ["약", "약중", "중", "중강", "강"]  # ac_fan = 0‥4


def fmt_action(t: int, f: int) -> str:
    """현재 AC 설정을 문자열로 변환"""
    return f"{t:2d}°C / 풍량 {f}단({FAN_LABELS[f]})"


# ───────────────────────────── main ──────────────────────────────
env = SubwayCoolingEnv()
model = PPO.load("ppo_subway_v3", env=env)

N_EPISODES = 20
cur_temp, cur_fan = 25, None # 나중에 건너서 받기

for ep in range(N_EPISODES):
    ext_temp = 30           # °C
    ext_hum  = 48.0            # %
    obs, _ = env.reset(outside_temp=ext_temp,
                       outside_humidity=ext_hum, ac_temp=cur_temp, ac_fan=cur_fan)
    done = False
    total_r = 0.0

    prev_temp, prev_fan = env.ac_temp, env.ac_fan

    print(f"\n🔁 [Episode {ep + 1}]")
    print(f"외부 {env.outside_temp:.1f}°C / {env.outside_humidity:.1f}% | "
          f"초기 설정 → {fmt_action(prev_temp, prev_fan)}\n")

    #while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    total_r += reward

       
   
    


    votes = info["votes"]
    mean_vote = info["mean_vote"]
    vc = Counter(votes)
    vote_str = " | ".join(f"{VOTE_LABELS[v]}: {vc.get(v, 0)}" for v in range(-2, 3))


    d_temp, d_fan = info["ac_temp"] - prev_temp, info["ac_fan"] - prev_fan

    # 4. 규칙 적용
    #    vote_sum > 0  → 덥다 의견 우세 → temp_code ↓ (온도 낮춤)
    #    vote_sum < 0  → 춥다 의견 우세 → temp_code ↑ (온도 높임)
    if  mean_vote > 0 and d_temp > 0:          # 덥다 → temp_code 1 (ΔT = −1 °C)
        print("덥다 의견 우세 → 온도 낮춤")
        action[0] = 1
        cur_temp, cur_fan = cur_temp-1, info["ac_fan"]
          
    elif mean_vote < 0 and d_temp < 0:        # 춥다 → temp_code 3 (ΔT = +1 °C)
        print("춥다 의견 우세 → 온도 높임")
        action[0] = 3
        cur_temp, cur_fan = cur_temp+1, info["ac_fan"]
    else:
        cur_temp, cur_fan = info["ac_temp"], info["ac_fan"]
    

    d_temp, d_fan = cur_temp - prev_temp, cur_fan - prev_fan
    prev_temp, prev_fan = cur_temp, cur_fan  

    print(f"Step {env.current_step:02d} | {vote_str} | Mean {mean_vote:+.2f} | "
            f"{fmt_action(cur_temp, cur_fan)} | "
            f"ΔT {d_temp:+.1f}°C, ΔF {d_fan:+d} | R {reward:+.3f}")

    print(f"\n✅ Total Reward: {total_r:+.3f}\n")
