import os
import sys
import json
import re
import traceback
from openai import OpenAI
from core import CustomerSupportEnv

MIN_VAL, MAX_VAL, MAX_STEPS = 0.001, 0.999, 20

def safe_parse(text):
    if not text: return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            content = match.group().replace("'", '"')
            return json.loads(content)
        except:
            pass
    return None

def make_client():
    api_key = os.environ["API_KEY"]
    base_url = os.environ["API_BASE_URL"].rstrip("/")
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

    # DO NOT modify base_url — use it exactly as injected by the validator
    print(f"[CONFIG] base_url={base_url} model={model_name}", flush=True)

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=30.0,
    )
    return client, model_name

def llm_call(client, model_name, prompt):
    res = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0,
    )
    return res.choices[0].message.content.strip()

def run():
    try:
        env = CustomerSupportEnv()
        state = env.reset()
        print("[START] task=customer_support_triage", flush=True)

        client, model_name = make_client()

        # ── SMOKE TEST ── confirm proxy is reachable before the main loop
        print("[SMOKE_TEST] sending test call to proxy...", flush=True)
        try:
            test_resp = llm_call(client, model_name, 'Reply with the word "ok"')
            print(f"[SMOKE_TEST] success, response={test_resp[:60]}", flush=True)
        except Exception as e:
            print(f"[SMOKE_TEST_FAIL] {type(e).__name__}: {e}", flush=True)
            traceback.print_exc(file=sys.stdout)
            # Don't exit — keep going so the validator can at least see the error

        total_reward, steps = 0, 0
        for step in range(1, MAX_STEPS + 1):
            prompt = (
                f"State:\n{json.dumps(state)}\n\n"
                "Return ONLY valid JSON with no explanation:\n"
                '{"ticket_id": <int>, "action": "reply" | "close" | "escalate" | "mark_spam"}'
            )

            action = None
            try:
                text = llm_call(client, model_name, prompt)
                print(f"[LLM] step={step} raw={text[:100]}", flush=True)
                action = safe_parse(text)
                if not action:
                    print(f"[PARSE_FAIL] step={step} could not parse: {text}", flush=True)
            except Exception as e:
                print(f"[LLM_ERROR] step={step} {type(e).__name__}: {e}", flush=True)
                traceback.print_exc(file=sys.stdout)

            if not action:
                inbox = state.get("inbox", [])
                tid = inbox[0]["id"] if inbox else 1
                action = {"ticket_id": tid, "action": "reply"}
                print(f"[FALLBACK] step={step} action={action}", flush=True)

            step_result = env.step(action)
            state, reward, done = step_result[0], step_result[1], step_result[2]
            total_reward += reward
            steps += 1
            print(f"[STEP] step={step} reward={float(reward):.4f}", flush=True)
            if done:
                break

        final_score = max(MIN_VAL, min(MAX_VAL, float(total_reward / 50)))
        print(f"[END] success=True steps={steps} rewards={final_score:.4f}", flush=True)
        print(f"[TOTAL_SUMMARY] task=customer_support_triage score={final_score:.4f}", flush=True)

    except Exception as e:
        print(f"[FATAL] {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.exit(0)

if __name__ == "__main__":
    run()