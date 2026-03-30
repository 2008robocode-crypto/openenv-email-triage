import os
import json
import re
from openai import OpenAI

from core import CustomerSupportEnv

# ===== REQUIRED ENV VARIABLES =====
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

# ===== CONFIG =====
MAX_STEPS = 20
TEMPERATURE = 0.2
MAX_TOKENS = 150

# ===== OPENAI CLIENT =====
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

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


# ===== FALLBACK POLICY (SMART) =====
def fallback_policy(state):
    # handle spam first
    for t in state["inbox"]:
        if not t["resolved"] and t["issue_type"] == "spam":
            return {"ticket_id": t["id"], "action": "mark_spam"}

    # handle VIP
    for t in state["inbox"]:
        if not t["resolved"] and t["customer_type"] == "vip":
            return {"ticket_id": t["id"], "action": "escalate"}

    # close remaining
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

Rules:
- Spam → mark_spam
- VIP → escalate before closing
- Urgency >= 4 → prioritize
- Queries → reply

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
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        text = response.choices[0].message.content.strip()
        action = safe_parse(text)

        if not action or "ticket_id" not in action or "action" not in action:
            return fallback_policy(state)

        return action

    except Exception:
        return fallback_policy(state)


# ===== MAIN RUN =====
def run():
    env = CustomerSupportEnv()
    state = env.reset()

    total_reward = 0

    for _ in range(MAX_STEPS):
        action = llm_policy(state)

        state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    score = max(0.0, min(1.0, total_reward / 50))

    print("Evaluation Result")
    print({
        "total_reward": total_reward,
        "score": score
    })


if __name__ == "__main__":
    run()