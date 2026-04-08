import os
import sys
import json
import re
import traceback
from openai import OpenAI
from core import CustomerSupportEnv

# Configuration
MIN_VAL, MAX_VAL, MAX_STEPS = 0.001, 0.999, 20

def safe_parse(text):
    if not text: return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            # Standardizing to double quotes for json.loads
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

        # MANDATORY: Explicitly use environment variables as requested
        # If these are missing, the script SHOULD crash here.
        api_key = os.environ["API_KEY"]
        base_url = os.environ["API_BASE_URL"]
        model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

        client = OpenAI(base_url=base_url, api_key=api_key)

        total_reward, steps = 0, 0
        for step in range(1, MAX_STEPS + 1):
            # 1. Prepare Prompt
            prompt = f"State: {json.dumps(state)}\nReturn JSON: {{'ticket_id': int, 'action': 'reply'}}"
            
            # 2. LLM Call - NO internal try/except here. 
            # We want to see the error if the proxy rejects us.
            res = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0
            )
            
            text = res.choices[0].message.content.strip()
            action = safe_parse(text)
            
            # 3. Step logic
            if not action:
                # Basic fallback if LLM returns non-JSON text
                action = {"ticket_id": 1, "action": "close"}

            # Flexible unpacking to handle 3, 4, or 5 return values
            step_result = env.step(action)
            state = step_result[0]
            reward = step_result[1]
            done = step_result[2]
            
            total_reward += reward
            steps += 1
            print(f"[STEP] step={step} reward={float(reward):.4f}", flush=True)
            if done: break

        final_score = max(MIN_VAL, min(MAX_VAL, float(total_reward / 50)))
        print(f"[END] success=True steps={steps} rewards={final_score:.4f}", flush=True)
        print(f"[TOTAL_SUMMARY] task=customer_support_triage score={final_score:.4f}", flush=True)

    except Exception as e:
        # Log the full error to stderr so you can read it in the validator logs
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1) # Exit with error code so the validator knows it failed

if __name__ == "__main__":
    run()