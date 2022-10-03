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
model.compile(optimizer='adam', loss='mean_squared_error', learning_rate=0.001)

score = [] #에피소드의 점수 저장
memory = deque(maxlen= 2000) # 현재 상태와 행동, 다음상태, 보상 등을 보관

env = gym.make("CartPole-v0")

#에피소드 시작
for i in range(1000):
    state = env.reset()
    state = np.reshape(state, [1, 4])
    eps = 1 / (i / 50 + 10)
