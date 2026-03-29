import os
from grader import evaluate
from baseline_agent import baseline_policy

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

def run():
    result = evaluate(baseline_policy)
    print("Evaluation Result")
    print(result)

if __name__ == "__main__":
    run()