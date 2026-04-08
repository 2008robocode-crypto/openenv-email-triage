from core import CustomerSupportEnv

def run_episode(policy_fn):
    env = CustomerSupportEnv()
    state = env.reset()
    total_reward = 0

    for _ in range(20):
        action = policy_fn(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break

    return total_reward


def evaluate(policy_fn):
    total = 0
    episodes = 5

    for _ in range(episodes):
        total += run_episode(policy_fn)

    avg_reward = total / episodes
    score = max(0.0, min(1.0, avg_reward / 50))

    return {
        "total_reward": total,
        "score": score
    }