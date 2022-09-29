import gym

env = gym.make('CartPole-v0')

for i_episode in range(20): #20회의 에피소드와
    observation = env.reset()

    for t in range(100): # 100 시간 스텝동안 게임을 진행함.
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()