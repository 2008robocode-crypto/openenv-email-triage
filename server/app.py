
from fastapi import FastAPI, Request

app = FastAPI()

@app.get("/")
def root():
    return "ok"

@app.head("/")
def root_head():
    return {}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/reset")
def reset_get():
    from core import CustomerSupportEnv
    env = CustomerSupportEnv()
    return env.reset()

@app.post("/reset")
def reset_post():
    from core import CustomerSupportEnv
    env = CustomerSupportEnv()
    return env.reset()

@app.get("/state")
def state():
    from core import CustomerSupportEnv
    env = CustomerSupportEnv()
    return env.state()

@app.post("/step")
async def step(request: Request):
    from core import CustomerSupportEnv

    action = await request.json()
    env = CustomerSupportEnv()

    state, reward, done, info = env.step(action)

    return {
        "state": state,
        "reward": reward,
        "done": done,
        "info": info
    }

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


@app.get("/favicon.ico") # to remove noice from logs
def favicon():
    return {}