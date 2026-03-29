from fastapi import FastAPI
from core import CustomerSupportEnv
from grader import evaluate
from baseline_agent import baseline_policy

app = FastAPI()

env = CustomerSupportEnv()

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: dict):
    return env.step(action)

@app.get("/state")
def state():
    return env.state()

@app.get("/baseline")
def baseline():
    return evaluate(baseline_policy)

@app.get("/tasks")
def tasks():
    return {
    "tasks": [
        {"name": "easy", "description": "Handle spam emails"},
        {"name": "medium", "description": "Prioritize urgent tickets"},
        {"name": "hard", "description": "Handle VIP workflow correctly"}
    ]
}

@app.get("/grader")
def grader():
    return {"info": "Use baseline endpoint for score"}