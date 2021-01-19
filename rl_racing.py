import gym
import cv2
from utils import process_image

env = gym.make('CarRacing-v0')
init_state = env.reset()

# Note: au début du jeu la caméra est loin du terrain et zoom progressivement pendant ~1 seconde
# faudrait peut être ne pas faire d'observation sur cette période là

for i in range(1000):
    env.render()
    state,_,_,_ = env.step(env.action_space.sample()) # take a random action
    if (i == 120): 
        cv2.imshow("hello", process_image(state))
        cv2.waitKey(0)
env.close()