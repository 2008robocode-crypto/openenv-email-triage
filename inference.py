import os
import sys
import json
import re
import traceback
from openai import OpenAI
from core import CustomerSupportEnv

# STRICT (NO FALLBACKS — REQUIRED)
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

MIN_VAL, MAX_VAL, MAX_STEPS = 0.001, 0.999, 20


def log_start(task, env_name, model):
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(step, action_str, reward, done, error=None):
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={str(done).lower()} "
        f"error={error if error else 'null'}",
        flush=True
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True
    )


def safe_parse(text):
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return None


def fallback_policy(state):
    inbox = state.get("inbox", [])
    for t in inbox:
        if not t.get("resolved") and t.get("issue_type") == "spam":
            return {"ticket_id": t["id"], "action": "mark_spam"}
    for t in inbox:
        if not t.get("resolved"):
            return {"ticket_id": t["id"], "action": "reply"}
    return {"ticket_id": 1, "action": "close"}


def run():
    success = False
    steps_taken = 0
    rewards = []

    log_start("customer_support_triage", "openenv", MODEL_NAME)

    try:
        env = CustomerSupportEnv()
        state = env.reset()

        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY
        )

        # 🔥 CRITICAL: HARD API CALL (NO TRY/EXCEPT)
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )

        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            prompt = f"""
State:
{json.dumps(state)}

Return ONLY JSON:
{{"ticket_id": int, "action": "reply|close|escalate|mark_spam"}}
"""

            # 🚨 NO SILENT FAIL
            res = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0,
            )

            text = res.choices[0].message.content.strip()

            try:
                action = safe_parse(text)
            except:
                action = None

            error = None
            if not action:
                action = fallback_policy(state)
                error = "parse_failed"

            state, reward, done, _ = env.step(action)

            reward = float(reward)
            rewards.append(reward)
            steps_taken = step

            log_step(step, json.dumps(action), reward, done, error)

        score = sum(rewards) / 50 if rewards else 0
        score = max(MIN_VAL, min(MAX_VAL, score))
        success = score > 0.01

    except Exception as e:
        # ONLY catch here (top level)
        print(f"[FATAL] {e}", flush=True)
        traceback.print_exc(file=sys.stdout)

    finally:
        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    run()
    sys.exit(0)