from subway_demo_env import SubwayCoolingEnv
from stable_baselines3 import PPO
from collections import Counter
import numpy as np

# 투표 값 → 레이블 매핑
VOTE_LABELS = {
    -2: "매우춥다",
    -1: "약간춥다",
     0: "보통이다",
     1: "약간덥다",
     2: "매우덥다"
}

TEMP_VALUES = list(np.arange(18.0, 31.0, 1.0))  # 총 13단계
FAN_VALUES = [0.5, 1.0, 1.5]                    # 총 3단계

def format_action(action):
    temp_idx, fan_idx = action
    temp_val = TEMP_VALUES[temp_idx]
    fan_val = FAN_VALUES[fan_idx]
    return f"(온도 설정: {temp_val:.1f}°C, 풍량: {fan_val:.1f}단계)"

# 환경 및 모델 로드
env = SubwayCoolingEnv()
model = PPO.load("ppo_subway_v3", env=env)

n_eval_episodes = 1

for episode in range(n_eval_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    prev_action = (env.ac_temp, env.ac_fan)

    print(f"\n🔁 [Episode {episode + 1} 시작]")
    print(f"▶️ 외부 온도: {env.outside_temp:.1f}°C, 습도: {env.outside_humidity:.1f}%")
    print(f"▶️ 초기 에어컨 설정: 온도 {TEMP_VALUES[env.ac_temp]:.1f}°C / 풍량 {FAN_VALUES[env.ac_fan]:.1f}단계\n")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward

        temp_idx, fan_idx = action
        delta_temp = TEMP_VALUES[temp_idx] - TEMP_VALUES[prev_action[0]]
        delta_fan = FAN_VALUES[fan_idx] - FAN_VALUES[prev_action[1]]
        prev_action = action

        votes = info.get("votes", [])
        mean_vote = info.get("mean_vote", 0.0)

        vote_counter = Counter(votes)
        vote_summary = " | ".join(
            f"{VOTE_LABELS[v]}: {vote_counter.get(v, 0)}"
            for v in range(-2, 3)
        )

        print(f"Step {env.current_step:2d} | {vote_summary} | "
              f"Mean: {mean_vote:+.2f} | {format_action(action)} | "
              f"Δ온도: {delta_temp:+.1f}, Δ풍량: {delta_fan:+.1f} | "
              f"Reward: {reward:.3f}")

    print(f"\n✅ [Episode {episode + 1} 종료] Total Reward: {total_reward:.3f}")
