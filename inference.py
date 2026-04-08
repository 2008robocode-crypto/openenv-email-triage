import os
import json
import re
from openai import OpenAI
from core import CustomerSupportEnv

MAX_STEPS = 20


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


# ===== SAFE CLIENT CREATION =====
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
def llm_policy(state):
    client = get_client()

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
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=50,
        )

        # ✅ SAFE access (fixes "list index out of range")
        if not response or not response.choices:
            return fallback_policy(state)

        text = response.choices[0].message.content
        if not text:
            return fallback_policy(state)

        action = safe_parse(text)

        if not action or "ticket_id" not in action or "action" not in action:
            return fallback_policy(state)

        return action

    except Exception as e:
        print(f"[LLM ERROR] {e}", flush=True)
        return fallback_policy(state)


# ===== MAIN RUN =====
def run():
    env = CustomerSupportEnv()
    state = env.reset()

    print("[START] task=customer_support_triage", flush=True)

    total_reward = 0

    for step in range(1, MAX_STEPS + 1):
        action = llm_policy(state)

        try:
            state, reward, done, _ = env.step(action)
        except Exception as e:
            print(f"[ENV ERROR] {e}", flush=True)
            break

        total_reward += reward

        print(f"[STEP] step={step} reward={reward}", flush=True)

        if done:
            break

    score = max(0.0, min(1.0, total_reward / 50))
    print(f"[END] score={score}", flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"[FATAL ERROR] {e}", flush=True)