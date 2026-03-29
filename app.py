from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "running", "service": "openenv-email-triage-v1"}

@app.get("/reset")
def reset():
    from core import CustomerSupportEnv
    env = CustomerSupportEnv()
    return env.reset()

@app.get("/state")
def state():
    from core import CustomerSupportEnv
    env = CustomerSupportEnv()
    return env.state()

@app.post("/step")
def step(action: dict):
    from core import CustomerSupportEnv
    env = CustomerSupportEnv()
    return env.step(action)

@app.get("/baseline")
def baseline():
    from grader import evaluate
    from baseline_agent import baseline_policy
    return evaluate(baseline_policy)

@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"name": "easy"},
            {"name": "medium"},
            {"name": "hard"}
        ]
    }

@app.get("/grader")
def grader():
    return {"info": "use /baseline"}