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
CUMUL_FRAMES = 1


def play_model(model_path):
    env = gym.make('CarRacing-v0')
    model = load_model(model_path)
    
    
    current_state = env.reset()
    current_state = process_image(current_state)
    state_stack = [deepcopy(current_state), deepcopy(current_state), deepcopy(current_state)]
    done = False
    #steps = 0

    while not done:
        env.render()
        input_stack_state = np.transpose(np.array([state_stack[-3], state_stack[-2], state_stack[-1]]), (1, 2, 0))
        prediction = model.predict(np.expand_dims(input_stack_state, axis=0))[0]
        action = np.argmax(prediction)
        print(action)
        for _ in range(CUMUL_FRAMES):
                next_state, _, done, _ = env.step(action_space[action])
                if done:
                    break
        next_state = process_image(next_state)
        state_stack.append(next_state)

        #if steps > 300 and total_reward < 0:
        #    break

play_model("model/test1")