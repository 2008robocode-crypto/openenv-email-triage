import os
import sys
import json
import re
from openai import OpenAI
from core import CustomerSupportEnv

# ===== CONFIG =====
MAX_STEPS = 20

# ===== STRICT ENV (DO NOT CHANGE) =====
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")


# ===== SAFE JSON PARSER =====
def safe_parse(text):
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return None


# ===== FALLBACK POLICY =====
def fallback_policy(state):
    for t in state.get("inbox", []):
        if not t.get("resolved") and t.get("issue_type") == "spam":
            return {"ticket_id": t["id"], "action": "mark_spam"}

    for t in state.get("inbox", []):
        if not t.get("resolved") and t.get("customer_type") == "vip":
            return {"ticket_id": t["id"], "action": "escalate"}

    for t in state.get("inbox", []):
        if not t.get("resolved"):
            return {"ticket_id": t["id"], "action": "close"}

    return {"ticket_id": 1, "action": "reply"}


# ===== LLM CALL =====
def call_llm(client, state):
    prompt = f"""You are an AI agent solving a customer support email triage task.

State:
{json.dumps(state)}

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
            max_tokens=100,
        )

        text = response.choices[0].message.content if response.choices else None
        return safe_parse(text)

    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return None


# ===== MAIN RUN =====
def run():
    env = CustomerSupportEnv()
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    state = env.reset()

    rewards = []
    steps = 0

    # ✅ START LOG (STRICT FORMAT)
    print(
        f"[START] task=customer_support_triage env=openenv model={MODEL_NAME}",
        flush=True,
    )

    try:
        for step in range(1, MAX_STEPS + 1):

            action = call_llm(client, state)

            if not action:
                action = fallback_policy(state)

            try:
                state, reward, done, info = env.step(action)
                error = None
            except Exception as e:
                reward = 0.0
                done = False
                error = str(e)

            rewards.append(float(reward))
            steps = step

            # ✅ STEP LOG (STRICT FORMAT)
            print(
                f"[STEP] step={step} action={action} reward={reward:.2f} "
                f"done={str(done).lower()} error={error if error else 'null'}",
                flush=True,
            )

            if done:
                break

        # ✅ SCORE CALC
        total_reward = sum(rewards)
        score = max(0.0, min(1.0, total_reward / 50))
        success = score > 0

    except Exception as e:
        print(f"[FATAL] {e}", flush=True)
        success = False
        score = 0.0

    # ✅ END LOG (STRICT FORMAT)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


if __name__ == "__main__":
    run()