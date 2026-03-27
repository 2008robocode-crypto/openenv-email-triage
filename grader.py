from core import CustomerSupportEnv


def run_episode(policy_fn, task_mode="hard"):

    env = CustomerSupportEnv(task_mode=task_mode)

    state = env.reset()

    total_reward = 0
    done = False

    while not done:

        action = policy_fn(state)

        state, reward, done, info = env.step(action)

        total_reward += reward

    return total_reward


def normalize_score(total_reward):

    # simple heuristic scaling
    min_r = -50
    max_r = 80

    score = (total_reward - min_r) / (max_r - min_r)

    score = max(0.0, min(1.0, score))

    return score


def evaluate(policy_fn):

    reward = run_episode(policy_fn)

    score = normalize_score(reward)

    return {
        "total_reward": reward,
        "score": score
    }