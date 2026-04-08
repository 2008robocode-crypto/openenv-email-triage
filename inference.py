import os
import sys
import json
import re
import traceback
from openai import OpenAI
from core import CustomerSupportEnv

try:
    API_KEY = os.environ["API_KEY"]   # validator
except KeyError:
    API_KEY = os.environ.get("HF_TOKEN")  # HF fallback

try:
    API_BASE_URL = os.environ["API_BASE_URL"]
except KeyError:
    API_BASE_URL = "https://router.huggingface.co/v1"

MIN_VAL, MAX_VAL, MAX_STEPS = 0.001, 0.999, 20

# ── If no proxy creds → we're on the HF Space, run the web server instead ──
if not API_KEY or not API_BASE_URL:
    import uvicorn
    from fastapi import FastAPI, Request

    app = FastAPI()
    env = CustomerSupportEnv()

    @app.get("/")
    def root():
        return {"status": "running"}

    @app.post("/reset")
    def reset():
        global env
        env = CustomerSupportEnv()
        return env.reset()

    @app.post("/step")
    async def step(request: Request):
        global env
        action = await request.json()
        state, reward, done, info = env.step(action)
        return {"state": state, "reward": reward, "done": done, "info": info}

    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=7860)
    sys.exit(0)

# ── Otherwise → validator mode ──
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
    if not text: return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try: return json.loads(match.group())
        except: pass
    return None

def fallback_policy(state):
    inbox = state.get("inbox", [])
    for t in inbox:
        if not t.get("resolved") and t.get("issue_type") == "spam":
            return {"ticket_id": t["id"], "action": "mark_spam"}
    for t in inbox:
        if not t.get("resolved"):
            return {"ticket_id": t["id"], "action": "reply"}
    tid = inbox[0]["id"] if inbox else 1
    return {"ticket_id": tid, "action": "close"}

def call_llm(client, prompt):
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0,
            timeout=20.0,
        )
        text = res.choices[0].message.content.strip()
        print(f"[DEBUG] LLM response: {text[:100]}", flush=True)
        return text
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {type(e).__name__}: {e}", flush=True)
        return None

def run():
    success = False
    steps_taken = 0
    final_score = MIN_VAL
    rewards = []

    log_start(task="customer_support_triage", env_name="openenv", model=MODEL_NAME)
    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] API_KEY=SET", flush=True)

    try:
        env = CustomerSupportEnv()
        state = env.reset()
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY, timeout=20.0)

        print("[DEBUG] Smoke test...", flush=True)
        smoke = call_llm(client, 'Reply with the single word: ok')
        print(f"[DEBUG] Smoke test: {smoke}", flush=True)

        done = False
        for step in range(1, MAX_STEPS + 1):
            if done: break

            prompt = (
                f"You are a customer support triage agent.\n"
                f"Current state:\n{json.dumps(state, indent=2)}\n\n"
                f"Choose the best action for one unresolved ticket.\n"
                f"Return ONLY valid JSON, no explanation, no markdown:\n"
                f'{{ "ticket_id": <int>, "action": "<reply|close|escalate|mark_spam>" }}'
            )

            text = call_llm(client, prompt)
            action = safe_parse(text)
            error = None

            if not action:
                action = fallback_policy(state)
                error = "parse_failed"

            step_result = env.step(action)
            state   = step_result[0]
            reward  = float(step_result[1])
            done    = step_result[2]

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action_str=json.dumps(action), reward=reward, done=done, error=error)

        raw_score = sum(rewards) / 50 if rewards else MIN_VAL
        final_score = max(MIN_VAL, min(MAX_VAL, raw_score))
        success = final_score > 0.01

    except Exception as e:
        print(f"[DEBUG] FATAL: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)

    finally:
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

if __name__ == "__main__":
    run()
    sys.exit(0)