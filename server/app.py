from fastapi import FastAPI, Request
import uvicorn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core import CustomerSupportEnv

app = FastAPI()

# ✅ Persist environment
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

    return {
        "state": state,
        "reward": reward,
        "done": done,
        "info": info
    }


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()