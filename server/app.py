from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def root():
    return {"status": "running"}

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

@app.post("/step")
async def step(request):
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

# ✅ ADD THIS
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


# ✅ REQUIRED
if __name__ == "__main__":
    main()