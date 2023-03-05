import gym
import matplotlib.pyplot as plt


env = gym.make("CartPole-v0")
# render = lambda: plt.imshow(env.render(mode = 'rgb_array'))
# env.reset()
#
# for _ in range(100):
#     env.step(env.action_space.sample())
#     env.render()
#
# env.close()
#
# print(env.observation_space)
# print(env.observation_space.shape)
# print(env.action_space)
#
# print(env.reset())
# print(env.observation_space.high)
# print(env.observation_space.low)


num_episodes = 100
num_timesteps = 50

for i in range(num_episodes):

    Return = 0
    state = env.reset()

    for t in range(num_timesteps):
        random_action = env.action_space.sample()
        next_state, reward, done, _ = env.step(random_action)
        Return += reward
        if done:
            break

    if i % 20 == 0:
        print('Episode : {}, Return : {}'.format(i,Return))

env.close()