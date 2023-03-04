import gym
import numpy as np

env = gym.make("FrozenLake-v1")


def value_evaluation(policy):
    num_iterations = 5000
    threshold = 1e-20
    gamma = 0.9
    value_table = np.zeros(env.observation_space.n)

    for i in range(num_iterations):
        updated_value_table = np.copy(value_table)
        for s in range(env.observation_space.n):
            a = policy[s]
            value_table[s] = sum([prob * (r + gamma * updated_value_table[s_])
                                  for prob, s_, r, _ in env.P[s][a]])

        if(np.sum(np.fabs(updated_value_table - value_table)) < threshold):
            break
    return value_table


def policy_improvement(value_table):
    gamma = 0.9
    policy = np.ones(env.observation_space.n)

    for s in range(env.observation_space.n):
        q_values=[sum([prob*(r+gamma*value_table[s_])
                          for prob,s_,r,_ in env.P[s][a]])
                        for a in range(env.action_space.n)]
        policy[s]=np.argmax(np.array(q_values))

    return policy


def policy_iteration(env):
    num_iterations = 3000
    policy = np.zeros(env.observation_space.n)

    for i in range(num_iterations):
        value_function = value_evaluation(policy)
        new_policy = policy_improvement(value_function)
        if (np.all(policy == new_policy)):
            break
        else:
            policy = new_policy
    return value_function, policy


optimal_value, optimal_policy = policy_iteration(env)
print(optimal_value)
print(optimal_policy)

obs = env.reset()
episode_reward = 0.0
for _ in range(20):
    action = np.int_(optimal_policy[obs])
    obs, reward, done, info = env.step(action)
    env.render()
    episode_reward += reward
    if done:
        print("reward:", episode_reward)
        episode_reward = 0.0
        obs = env.reset()