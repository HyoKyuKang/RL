import json
import numpy as np
from stable_baselines3 import PPO
from subway_env import SubwayCoolingEnv

# 전역 초기화 (Azure가 시작될 때 한 번만 실행됨)
def init():
    global model, env
    env = SubwayCoolingEnv()
    model = PPO.load("subway_model.zip", env=env)

# 추론 요청이 올 때마다 실행됨
def run(raw_data):
    state = np.array(raw_data["state"], dtype=np.float32)
    action, _ = model.predict(state, deterministic=True)
    return json.dumps({"action": action.tolist()})
