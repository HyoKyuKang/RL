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

# 온도/풍량 값 리스트 (env와 동일하게 정의)
TEMP_VALUES = list(np.arange(18.0, 31.0, 1.0))  # 13개: 18.0 ~ 30.0
FAN_VALUES = [0.5, 1.0, 1.5]                    # 3개

def format_action(action):
    temp_idx, fan_idx = action
    temp_value = TEMP_VALUES[temp_idx]
    fan_value = FAN_VALUES[fan_idx]
    return f"(온도: {temp_value:.1f}°C, 풍량: {fan_value:.1f}단계)"

# 환경 및 학습된 PPO 모델 불러오기
env = SubwayCoolingEnv()
model = PPO.load("ppo_subway_v2", env=env)

n_eval_episodes = 1  # 원하는 에피소드 수만큼 평가

for episode in range(n_eval_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    print(f"\n🔁 [Episode {episode + 1} 시작]")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward

        # 투표 처리
        votes = info.get("votes", [])
        mean_vote = info.get("mean_vote", 0.0)
        vote_counter = Counter(votes)
        vote_summary = " | ".join(
            f"{VOTE_LABELS[v]}: {vote_counter.get(v, 0)}"
            for v in range(-2, 3)
        )

        # 출력 정리
        action_str = format_action(action)
        print(f"Step {env.current_step:2d} | {vote_summary} | "
              f"Mean Vote: {mean_vote:+.2f} | Action: {action_str} | "
              f"AC 설정: {env.ac_temp:.1f}°C / {env.ac_fan:.1f}단계 | Reward: {reward:.3f}")

    print(f"\n✅ [Episode {episode + 1} 종료] Total Reward: {total_reward:.3f}")
