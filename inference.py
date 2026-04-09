import os
import sys
import json
import re
import traceback
from openai import OpenAI
from core import CustomerSupportEnv

# =========================
# ENV VARIABLES
# =========================
API_BASE_URL = os.environ["API_BASE_URL"]  # must use validator's proxy
API_KEY      = os.environ["API_KEY"]       # must use validator's key
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")  # optional fallback

MIN_VAL, MAX_VAL, MAX_STEPS = 0.001, 0.999, 20

# =========================
# LOGGING
# =========================
def log_start(task, env_name, model):
    print(f"[START] task={task} env={env_name} model={model}", flush=True)

def log_step(step, action_str, reward, done, error=None):
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={str(done).lower()} "
        f"error={error if error else 'null'}",
        flush=True
    )

def log_end(success, steps_taken, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] task=customer_support_triage score={score:.3f} steps={steps_taken} rewards={rewards_str} success={str(success).lower()}",
        flush=True
    )

# =========================
# SAFE PARSER
# =========================
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

# =========================
# FALLBACK POLICY
# =========================
def fallback_policy(state):
    inbox = state.get("inbox", [])
    for t in inbox:
        if not t.get("resolved") and t.get("issue_type") == "spam":
            return {"ticket_id": t["id"], "action": "mark_spam"}
    for t in inbox:
        if not t.get("resolved"):
            return {"ticket_id": t["id"], "action": "reply"}
    return {"ticket_id": 1, "action": "close"}

# =========================
# MAIN RUN FUNCTION
# =========================
def run():
    success = False
    steps_taken = 0
    rewards = []
    score = 0  # initialize to avoid UnboundLocalError

    log_start("customer_support_triage", "openenv", MODEL_NAME)

    try:
        env = CustomerSupportEnv()
        state = env.reset()

        client = OpenAI(
            base_url=API_BASE_URL,  # validator proxy
            api_key=API_KEY
        )

        done = False
        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # ✅ single-line JSON prompt for validator detection
            prompt_dict = {
                "state": state,
                "return_only": ["ticket_id", "action"]
            }
            prompt = json.dumps(prompt_dict)

            res = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0
            )

            text = res.choices[0].message.content.strip()
            action = safe_parse(text)
            error = None

            if not action:
                action = fallback_policy(state)
                error = "parse_failed"

            state, reward, done, _ = env.step(action)
            reward = float(reward)
            rewards.append(reward)
            steps_taken = step

            log_step(step, json.dumps(action), reward, done, error)

        # Compute score
        score = sum(rewards) / 50 if rewards else 0
        score = max(MIN_VAL, min(MAX_VAL, score))
        success = score > 0.01

    except Exception as e:
        print(f"[FATAL] {e}", flush=True)
        traceback.print_exc(file=sys.stdout)

    finally:
        log_end(success, steps_taken, score, rewards)

if __name__ == "__main__":
    run()
    sys.exit(0)