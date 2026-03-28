from fastapi import FastAPI
from grader import evaluate
from baseline_agent import baseline_policy

app = FastAPI()

@app.get("/")
def home():
    return {"status": "running"}

@app.get("/baseline")
def run_baseline():
    result = evaluate(baseline_policy)
    return result