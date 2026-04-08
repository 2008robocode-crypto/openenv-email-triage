import os
import json
import re
from openai import OpenAI
from core import CustomerSupportEnv

# ===== CONFIG =====
MAX_STEPS = 20
MODEL_NAME = os.getenv("MODEL_NAME")

# ===== SAFE JSON PARSER =====
def safe_parse(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return None

# ===== FALLBACK POLICY =====
def fallback_policy(state):
    for t in state["inbox"]:
        if not t["resolved"] and t["issue_type"] == "spam":
            return {"ticket_id": t["id"], "action": "mark_spam"}

    for t in state["inbox"]:
        if not t["resolved"] and t["customer_type"] == "vip":
            return {"ticket_id": t["id"], "action": "escalate"}

    for t in state["inbox"]:
        if not t["resolved"]:
            return {"ticket_id": t["id"], "action": "close"}

    return {"ticket_id": 1, "action": "reply"}

# ===== CREATE CLIENT SAFELY =====
def get_client():
    try:
        return OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"],
        )
    except Exception as e:
        print(f"[CLIENT ERROR] {e}", flush=True)
        return None

# ===== LLM POLICY =====
def llm_policy(state, client):
    if client is None:
        return fallback_policy(state)

    prompt = f"""
You are an AI agent solving a customer support email triage task.

State:
{state}

Return ONLY valid JSON:
{{
    "ticket_id": int,
    "action": "reply" | "close" | "escalate" | "mark_spam"
}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=80,
        )

        # 🔥 SAFE ACCESS
        if not response or not response.choices:
            print("[WARN] Empty response", flush=True)
            return fallback_policy(state)

        text = response.choices[0].message.content

        if not text:
            print("[WARN] Empty content", flush=True)
            return fallback_policy(state)

        action = safe_parse(text)

        if not action or "ticket_id" not in action or "action" not in action:
            print("[WARN] Bad JSON", flush=True)
            return fallback_policy(state)

        return action

    except Exception as e:
        print(f"[LLM ERROR] {e}", flush=True)
        return fallback_policy(state)

# ===== MAIN RUN =====
def run():
    env = CustomerSupportEnv()
    state = env.reset()

    client = get_client()

    print("[START] task=customer_support_triage", flush=True)

    total_reward = 0

    for step in range(1, MAX_STEPS + 1):
        action = llm_policy(state, client)

        state, reward, done, _ = env.step(action)

        total_reward += reward

        print(f"[STEP] step={step} reward={reward}", flush=True)

        if done:
            break

    score = max(0.0, min(1.0, total_reward / 50))

    print(f"[END] score={score}", flush=True)

if __name__ == "__main__":
    run()