from flask import Flask, request, jsonify
import numpy as np
from stable_baselines3 import PPO
from subway_env import SubwayCoolingEnv

# ✅ Flask 앱 객체 먼저 선언해야 함
app = Flask(__name__)

# 모델과 환경 로드
env = SubwayCoolingEnv()
model = PPO.load("subway_base.zip", env=env)

# 라우팅은 그 다음
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if "state" not in data:
        return jsonify({"error": "Missing 'state' key"}), 400

    try:
        # ─────────────────────────────────────────────
        # 1. 입력 상태 벡터 → numpy
        obs = np.array(data["state"], dtype=np.float32)

        # 2. 모델이 제안한 원본 action
        action, _ = model.predict(obs, deterministic=True)   # [temp_code, fan_code]

        vote_sum = obs[9::4].sum()     
        if vote_sum > 0 and action[0] >= 2:        
            action[0] = 1
        elif vote_sum < 0 and action[0] <= 2:       
            action[0] = 3
        elif vote_sum == 0:
            action[0] = 2

        # ─────────────────────────────────────────────
        print(f"[REQUEST]  state  = {obs}")
        print(f"[VOTES]    sum    = {vote_sum}")
        print(f"[RESPONSE] action = {action}")

        return jsonify({"action": action.tolist()})
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500

# 메인 엔트리 포인트
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
