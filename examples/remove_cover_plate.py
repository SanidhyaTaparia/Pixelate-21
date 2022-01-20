import gym
import pix_main_arena
import time
import pybullet as p
import os

if __name__=="__main__":
    parent_path = os.path.dirname(os.getcwd())
    os.chdir(parent_path)
    env = gym.make("pix_main_arena-v0")
    x=0
    while True:
        p.stepSimulation()
        if x==10000:
            env.remove_cover_plate(0, 2)
        x+=1
    time.sleep(100)