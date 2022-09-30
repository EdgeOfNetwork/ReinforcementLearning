import gym

env = gym.make('CartPole-v0')
observation = env.reset()
print(observation)

observation, reward, done, info = env.step(0)
print(observation)

env.close()