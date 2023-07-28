import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from DQN_Brain import DQN_Brain

class Gym:
    def __init__(self, env_name):
        self.env = gym.make(env_name, render_mode='human')
        self.brain = DQN_Brain(self.env.action_space.n, self.env.observation_space.shape[0], memory_size=2000, e_greedy_increment=0.001)
        
    def run(self):
        his_steps = []
        last_step = 0
        step = 0
        observation, info = self.env.reset()

        while True:
            action = self.brain.choose_action(observation)

            observation_, reward, terminated, truncated, info = self.env.step(action)
            
            # the smaller theta and closer to center the better
            position, velocity = observation_
            reward = abs(velocity) * 100 / 7

            self.brain.store_transaction(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                self.brain.learn_DQ()

            if terminated or truncated:
                observation, info = self.env.reset()
                his_steps.append(step-last_step)
                last_step = step
                #self.plot_cost(his_steps)
            else:
                observation = observation_
                step +=1

    def plot_cost(self, his_steps):
        plt.plot(np.arange(len(his_steps)), his_steps)
        plt.ylabel('Steps')
        plt.xlabel('Episodes')
        plt.show()

    def close(self):
        self.env.close()
        
        
gym = Gym("MountainCar-v0")

gym.run()

gym.close()
