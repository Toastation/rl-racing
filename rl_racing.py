import gym
import cv2
from utils import process_image

env = gym.make('CarRacing-v0')
init_state = env.reset()

# Note: au début du jeu la caméra est loin du terrain et zoom progressivement pendant ~1 seconde
# faudrait peut être ne pas faire d'observation sur cette période là

tot_reward = 0
frame_count = 0
while True:
    env.render()
    state, reward, finish, _ = env.step(env.action_space.sample()) # random action
    processed_state = process_image(state)
    # if frame_count == 120:
    #     cv2.imshow("hello", processed_state)
    #     cv2.waitKey(0)
    tot_reward += reward
    frame_count += 1
    if finish:
        print(f"Frames: {frame_count}   |   Total reward: {tot_reward}")
        break
env.close()
