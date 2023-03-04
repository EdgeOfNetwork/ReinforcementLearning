import gym
import numpy as np

env = gym.make("FrozenLake-v1")

#1

# env.render()
# env.close()
#
# print(env.observation_space)
# print(env.action_space)
# print(env.observation_space.n)
# print(env.action_space.n)
#
# print(env.P[0][0])
# print(env.P[15][1])
# print(env.P[5][0])

#2
# state = env.reset()
# print(env.step(1))
# env.render()

#3
state = env.reset()

print("Time step 0 :")
env.render()

num_timesteps = 20
for t in range(num_timesteps):
    random_action = env.action_space.sample()

    new_state, reward, done, _ = env.step(random_action)
    print('timestep {}:'.format(t+1))

    env.render()

    if done:
        break

