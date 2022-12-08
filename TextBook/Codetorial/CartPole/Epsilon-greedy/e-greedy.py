import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque


model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(24, input_dim=4, activation=tf.nn.relu),
  tf.keras.layers.Dense(24, activation=tf.nn.relu),
  tf.keras.layers.Dense(2, activation='linear')
])

# 모델 컴파일
#model.compile(optimizer='adam', loss='mean_squared_error', learning_rate=0.001) #deprecated
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
              loss = tf.keras.losses.MeanSquaredError())

score = [] #에피소드의 점수 저장
memory = deque(maxlen= 2000) # 현재 상태와 행동, 다음상태, 보상 등을 보관

env = gym.make("CartPole-v0")

#에피소드 시작
for i in range(1000):
    state = env.reset()
    state = np.reshape(state, [1, 4])
    eps = 1 / (i / 50 + 10) #왜 상수값으로 정해놨지?

    for t in range(200):

        #200 timesteps
        if np.random.rand() < eps:
            action = np.random.randint(0, 2)
        else:
            predict = model.predict(state)
            action = np.argmax(predict)

        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4]) #뉴럴넷으로 들어가기 적합한 형태인 input_dim 사이즈에 맞춘다.

        memory.append((state, action, reward, next_state, done)) #상태 정보 저장
        state = next_state # update

        if done or t == 199:
            print("Episode", i, "Score", t + 1)
            score.append(t + 1)
            break

    #Training : 에피소드를 10회 이상 진행하는 경우
    if i > 10:
        minibatch = random.sample(memory, 16)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + 0.9 * np.amax(model.predict(next_state)[0])
            target_outputs = model.predict(state)
            target_outputs[0][action] = target
            model.fit(state, target_outputs, epochs = 1, verbose = 0)

env.close()
print(score)