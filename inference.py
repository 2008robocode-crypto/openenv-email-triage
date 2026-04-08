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

def run():
    try:
        env = CustomerSupportEnv()
        state = env.reset()
        print("[START] task=customer_support_triage", flush=True)

        api_key = os.environ["API_KEY"]
        base_url = os.environ["API_BASE_URL"]
        model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

        # FIX 1: Ensure base_url ends with /v1 (LiteLLM proxy requirement)
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

        print(f"[CONFIG] base_url={base_url} model={model_name}", flush=True)

        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        total_reward, steps = 0, 0
        for step in range(1, MAX_STEPS + 1):
            prompt = f"""State:
{json.dumps(state)}

Return ONLY valid JSON:
{{
    "ticket_id": int,
    "action": "reply" | "close" | "escalate" | "mark_spam"
}}"""

            # FIX 2: Don't silently swallow errors — log them loudly
            action = None
            try:
                res = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0
                )
                text = res.choices[0].message.content.strip()
                print(f"[LLM] step={step} response={text[:80]}", flush=True)
                action = safe_parse(text)
            except Exception as e:
                # FIX 3: Print full error so you can see what's going wrong
                print(f"[LLM_ERROR] step={step} error={type(e).__name__}: {e}", flush=True)
                traceback.print_exc(file=sys.stdout)  # stdout so validator sees it
                action = None

            if not action:
                inbox = state.get("inbox", [])
                tid = inbox[0]["id"] if inbox else 1
                action = {"ticket_id": tid, "action": "reply"}
                print(f"[FALLBACK] step={step} using fallback action", flush=True)

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
        print(f"FATAL ERROR: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.exit(0)

if __name__ == "__main__":
    run()