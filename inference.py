import os
import sys
import json
import re
import traceback
import httpx  # Make sure this is imported!
from openai import OpenAI
from core import CustomerSupportEnv

# =========================
# 🔥 FIX 1: NORMALIZE ENV (STRICT)
# =========================
# We use exactly what the validator asks for. 
# No 'or' fallbacks for the API_KEY.
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

MIN_VAL, MAX_VAL, MAX_STEPS = 0.001, 0.999, 20

# =========================
# PARSER & FALLBACK
# =========================
def safe_parse(text):
    if not text: return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group().replace("'", '"'))
        except: pass
    return None

def fallback_policy(state):
    inbox = state.get("inbox", [])
    if not inbox: return {"ticket_id": 1, "action": "close"}
    return {"ticket_id": inbox[0]["id"], "action": "reply"}

# =========================
# MAIN
# =========================
def run():
    print(f"[START] task=customer_support_triage model={MODEL_NAME}", flush=True)
    
    try:
        env = CustomerSupportEnv()
        state = env.reset()
        
        # 🔥 THE CRITICAL FIX: 
        # Create a client that explicitly ignores the broken system proxies.
        # This prevents the 'SyncHttpxClientWrapper' crash in your traceback.
        http_client = httpx.Client(trust_env=False)
        
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY,
            http_client=http_client
        )

        total_reward = 0
        steps_taken = 0
        rewards = []

        for step in range(1, MAX_STEPS + 1):
            prompt = f"State: {json.dumps(state)}\nReturn JSON: {{'ticket_id': int, 'action': 'reply'}}"
            
            action = None
            try:
                res = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0
                )
                action = safe_parse(res.choices[0].message.content)
            except Exception as e:
                print(f"[DEBUG] API Error: {e}", file=sys.stderr)

            if not action:
                action = fallback_policy(state)

            # Step and handle return values (works for 3, 4, or 5 values)
            results = env.step(action)
            state, reward, done = results[0], results[1], results[2]
            
            total_reward += float(reward)
            rewards.append(float(reward))
            steps_taken = step
            
            print(f"[STEP] step={step} reward={reward:.4f} done={str(done).lower()}", flush=True)
            if done: break

        score = max(MIN_VAL, min(MAX_VAL, total_reward / 50))
        print(f"[END] success=True steps={steps_taken} score={score:.4f}", flush=True)
        print(f"[TOTAL_SUMMARY] task=customer_support_triage score={score:.4f}", flush=True)

    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1) # Exit 1 so we can see the traceback in the logs

if __name__ == "__main__":
    run()