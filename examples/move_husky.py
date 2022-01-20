import gym
import pix_main_arena
import time
import pybullet as p
import pybullet_data
import cv2
import os

if __name__=="__main__":
    parent_path = os.path.dirname(os.getcwd())
    os.chdir(parent_path)
    env = gym.make("pix_main_arena-v0")
    time.sleep(2)
    while True:
        p.stepSimulation()
        env.move_husky(0.2, 0.2, 0.2, 0.2)