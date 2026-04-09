import os
import sys
import json
import re
import traceback
from openai import OpenAI
from core import CustomerSupportEnv

# ── Use ONLY the exact variable names the validator injects ──
# Do NOT check HF_TOKEN — it may be set to your personal token
# which would bypass the validator's proxy entirely
API_KEY      = os.environ.get("API_KEY")
API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

MIN_VAL, MAX_VAL, MAX_STEPS = 0.001, 0.999, 20

# ── If no proxy creds → running on HF Space, start web server ──
if not API_KEY or not API_BASE_URL:
    try:
        import uvicorn
        from fastapi import FastAPI, Request
        app = FastAPI()
        _env = CustomerSupportEnv()

        @app.get("/")
        def root():
            return {"status": "running"}

        @app.post("/reset")
        def reset():
            global _env
            _env = CustomerSupportEnv()
            return _env.reset()

        @app.post("/step")
        async def step(request: Request):
            global _env
            action = await request.json()
            state, reward, done, info = _env.step(action)
            return {"state": state, "reward": reward, "done": done, "info": info}

        if __name__ == "__main__":
            uvicorn.run(app, host="0.0.0.0", port=7860)
    except Exception as e:
        print(f"[INFO] Space mode failed: {e}", flush=True)
    sys.exit(0)

# ── Validator mode: proxy creds are present ──

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
        if not t.get("resolved"):
            return {"ticket_id": t["id"], "action": "reply"}
    return {"ticket_id": inbox[0]["id"] if inbox else 1, "action": "close"}

def run():
    success = False
    steps_taken = 0
    rewards = []
    score = MIN_VAL

    log_start("customer_support_triage", "openenv", MODEL_NAME)
    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)

    try:
        env = CustomerSupportEnv()
        state = env.reset()

        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY,
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
                    model=MODEL_NAME,
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