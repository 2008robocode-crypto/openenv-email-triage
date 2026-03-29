import os
from openai import OpenAI

from core import CustomerSupportEnv

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)


def llm_policy(state):
    prompt = f"""
    You are an agent solving email triage.

    State:
    {state}

    Choose action as JSON:
    {{
        "ticket_id": int,
        "action": "reply" | "close" | "escalate" | "mark_spam"
    }}
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=100,
    )

    text = response.choices[0].message.content

    try:
        import json
        return json.loads(text)
    except:
        # fallback safe action
        return {"ticket_id": 1, "action": "reply"}


def run():
    env = CustomerSupportEnv()
    state = env.reset()

    total_reward = 0

    for _ in range(20):
        action = llm_policy(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break

    score = max(0.0, min(1.0, total_reward / 50))

    print("Evaluation Result")
    print({
        "total_reward": total_reward,
        "score": score
    })


if __name__ == "__main__":
    run()