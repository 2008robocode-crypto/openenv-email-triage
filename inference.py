import os
import sys
import json
import re
import traceback
from openai import OpenAI
from core import CustomerSupportEnv

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
        try:
            return json.loads(match.group())
        except:
            pass
    return None

def fallback_policy(state):
    inbox = state.get("inbox", [])
    for t in inbox:
        if not t.get("resolved") and t.get("issue_type") == "spam":
            return {"ticket_id": t["id"], "action": "mark_spam"}
    for t in inbox:
        if not t.get("resolved") and t.get("customer_type") == "vip":
            return {"ticket_id": t["id"], "action": "escalate"}
    for t in inbox:
        if not t.get("resolved"):
            return {"ticket_id": t["id"], "action": "close"}
    return {"ticket_id": 1, "action": "reply"}

def run():
    # Read env vars INSIDE run() exactly as validator instructions say
    api_base_url = os.environ["API_BASE_URL"]
    api_key      = os.environ["API_KEY"]
    model_name   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

    success = False
    steps_taken = 0
    rewards = []
    score = MIN_VAL

    log_start("customer_support_triage", "openenv", model_name)
    print(f"[DEBUG] base_url={api_base_url} model={model_name}", flush=True)

    try:
        env = CustomerSupportEnv()
        state = env.reset()

        client = OpenAI(
            base_url=api_base_url,
            api_key=api_key,
            timeout=20.0,
        )

        done = False
        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            prompt = (
                f"You are a customer support triage agent.\n"
                f"State:\n{json.dumps(state, indent=2)}\n\n"
                f"Return ONLY valid JSON, no explanation:\n"
                f'{{ "ticket_id": <int>, "action": "<reply|close|escalate|mark_spam>" }}'
            )

            action = None
            error = None
            try:
                res = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0,
                )
                text = res.choices[0].message.content.strip()
                print(f"[DEBUG] step={step} llm={text[:80]}", flush=True)
                action = safe_parse(text)
            except Exception as e:
                print(f"[DEBUG] LLM error step={step}: {type(e).__name__}: {e}", flush=True)
                error = "llm_failed"

            if not action:
                action = fallback_policy(state)
                if not error:
                    error = "parse_failed"

            step_result = env.step(action)
            state  = step_result[0]
            reward = float(step_result[1])
            done   = step_result[2]

            rewards.append(reward)
            steps_taken = step
            log_step(step, json.dumps(action), reward, done, error)

        score = max(MIN_VAL, min(MAX_VAL, sum(rewards) / 50)) if rewards else MIN_VAL
        success = score > 0.01

    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)

    finally:
        log_end(success, steps_taken, score, rewards)

if __name__ == "__main__":
    run()
    sys.exit(0)