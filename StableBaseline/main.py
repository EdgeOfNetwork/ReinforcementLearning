import gym

env = gym.make("FrozenLake-v1")
env.render()
env.close()

print(env.observation_space)
print(env.action_space)
print(env.observation_space.n)
print(env.action_space.n)