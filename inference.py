import os
import json
import re
from openai import OpenAI
from core import CustomerSupportEnv

# ===== CONFIG =====
MAX_STEPS = 20
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

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

# ===== LLM POLICY =====
def llm_policy(state):
    try:
        # 🔥 DIRECT CLIENT (NO WRAPPER)
        client = OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"],
        )

        prompt = f"""
You are an AI agent solving a customer support email triage task.

State:
{state}

Rules:
- Spam → mark_spam
- VIP → escalate
- Urgency >= 4 → prioritize
- Otherwise → close or reply

Return ONLY valid JSON:
{{
    "ticket_id": int,
    "action": "reply" | "close" | "escalate" | "mark_spam"
}}
"""

        # 🔥 FORCE API CALL
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=50,
        )

        # ✅ SAFE RESPONSE HANDLING
        if not response or not hasattr(response, "choices") or len(response.choices) == 0:
            return fallback_policy(state)

        message = response.choices[0].message

        if not message or not message.content:
            return fallback_policy(state)

        text = message.content.strip()

        action = safe_parse(text)

        if action and "ticket_id" in action and "action" in action:
            return action

        return fallback_policy(state)

    except Exception as e:
        print(f"[LLM ERROR] {e}", flush=True)
        return fallback_policy(state)

# ===== MAIN RUN =====
def run():
    env = CustomerSupportEnv()
    state = env.reset()

    print("[START] task=customer_support_triage", flush=True)

    total_reward = 0
    steps = 0

    for step in range(1, MAX_STEPS + 1):
        action = llm_policy(state)

        state, reward, done, _ = env.step(action)

        total_reward += reward
        steps += 1

        print(f"[STEP] step={step} reward={reward}", flush=True)

        if done:
            break

    score = max(0.0, min(1.0, total_reward / 50))

    print(f"[END] task=customer_support_triage score={score} steps={steps}", flush=True)

if __name__ == "__main__":
    run()