from core import CustomerSupportEnv

env = CustomerSupportEnv(task_mode="hard")

state = env.reset()

print("Initial State:", state)