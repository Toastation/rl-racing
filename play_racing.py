import gym
import tensorflow as tf
from tensorflow.keras.models import load_model
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


def play_model(model_path):
    env = gym.make('CarRacing-v0')
    model = load_model(model_path)
    
    current_state = env.reset()
    current_state = np.reshape(process_image(current_state), (84, 96, 1))
    done = False
    #steps = 0

    while not done:
        env.render()
        prediction = model.predict(np.expand_dims(current_state, axis=0))[0]
        action = np.argmax(prediction)
        print(action)
        next_state, r, done, _ = env.step(action_space[action])
        next_state = np.reshape(process_image(next_state), (84, 96, 1))

        #if steps > 300 and total_reward < 0:
        #    break

        current_state = next_state

play_model("model/test1")