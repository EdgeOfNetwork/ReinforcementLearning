import gym

env = gym.make('CartPole-v0')
observation = env.reset()

print(observation)
#카트 위치/ 카트 속도 / 막대기의 각도 / 막대기의 회전율