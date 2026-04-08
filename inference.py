import os
import json
import re
from openai import OpenAI
from core import CustomerSupportEnv

# ===== STRICT ENV (NO FALLBACKS) =====
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# ===== CLIENT (GLOBAL, REQUIRED) =====
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

# ===== CONFIG =====
MAX_STEPS = 20

# ===== SAFE PARSER =====
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


# ===== FALLBACK =====
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


# ===== LLM POLICY =====
def llm_policy(state):
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
            max_tokens=50,
        )

        # 🔴 CRITICAL: DO NOT FAIL HERE
        text = response.choices[0].message.content.strip()

        action = safe_parse(text)

        if action and "ticket_id" in action and "action" in action:
            return action

        return fallback_policy(state)

    except Exception as e:
        print(f"[LLM ERROR] {e}", flush=True)
        return fallback_policy(state)


# ===== MAIN =====
def run():
    env = CustomerSupportEnv()
    state = env.reset()

    print("[START] task=customer_support_triage", flush=True)

    total_reward = 0

    for step in range(1, MAX_STEPS + 1):
        action = llm_policy(state)

        state, reward, done, _ = env.step(action)

        total_reward += reward

        print(f"[STEP] step={step} reward={reward}", flush=True)

        if done:
            break

    score = max(0.0, min(1.0, total_reward / 50))

    print(f"[END] score={score}", flush=True)


if __name__ == "__main__":
    run()