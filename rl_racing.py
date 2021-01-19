import gym
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from copy import deepcopy
from cv2 import cv2
from utils import process_image

action_space    = [
            (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), #           Action Space Structure
            (-1, 1,   0), (0, 1,   0), (1, 1,   0), #        (Steering Wheel, Gas, Break)
            (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # Range        -1~1       0~1   0~1
            (-1, 0,   0), (0, 0,   0), (1, 0,   0)
]
epsilon = 1.0
EPSILON_DECAY  = 0.999
GAMMA = 0.95
STEP_VERIFICATION = 300



# def process_state_image(state):
#     state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
#     state = state.astype(float)
#     state /= 255.0
#     return state

def new_model():
     # Neural Net for Deep-Q learning Model
    model = Sequential()
    #model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(96, 96, self.frame_stack_num)))
    model.add(Conv2D(filters=16, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(84, 96, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(len(action_space), activation=None))
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001, epsilon=1e-7))
    return model


def train_model(nb_episodes, model_path=None):
    global epsilon
    env = gym.make('CarRacing-v0')
    if model_path == None:
        model = new_model()
        model_path = 'model/test1'
    else:
        model = load_model(model_path)

    global_memory = deque(maxlen=1000)

    for k in range(nb_episodes):
        print(k)
        total_reward = 0
        current_state = env.reset()
        current_state = np.reshape(process_image(current_state), (84, 96, 1))
        episode_history = [current_state]
        done = False
        steps = 0

        while not done:
            if random.random() < epsilon:
                action = random.choice(range(len(action_space)))
            else:
                prediction = model.predict(np.expand_dims(current_state, axis=0))[0]
                action = np.argmax(prediction)
            next_state, r, done, _ = env.step(action_space[action])
            next_state = np.reshape(process_image(next_state), (84, 96, 1))
            total_reward += r
            episode_history.append([deepcopy(current_state), action, r, deepcopy(next_state), done])

            if steps > 300 and total_reward < 0:
                break

            current_state = next_state

        global_memory.extend(episode_history)

        #Create minibatch and train network
        if len(global_memory) > 128:
            minibatch = random.sample(global_memory, 128)
            states_train = []
            expected_reward_train = []
            for s, a, r, ns, d in minibatch:
                target = model.predict(np.expand_dims(s, axis=0))[0]
                if d:
                    target[a] = r
                else:
                    next_state_rewards = model.predict(np.expand_dims(ns, axis=0))[0]
                    target[a] = r + GAMMA * np.amax(next_state_rewards)
                states_train.append(s)
                expected_reward_train.append(target)
            model.fit(np.array(states_train), np.array(expected_reward_train), epochs=1, verbose=0)
            epsilon = epsilon if epsilon < 0.1 else epsilon*EPSILON_DECAY
    #model.save(model_path)

train_model(2)
