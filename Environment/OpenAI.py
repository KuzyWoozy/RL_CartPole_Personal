import gym
import numpy as np

class Environment:
    def __init__(self, environment_name, bot, debug=True):
        self.env = gym.make(environment_name)
        self.observation = self.env.reset()
        self.bot = bot(0.9, self.observation)
        self.debug = debug

    def train(self, time_steps):
        
        for time in range(time_steps):
            self.env.render()

            choice = np.argmax(self.bot.action())
            
            new_observation, reward, done, _ = self.env.step(choice)
            
            reward = 1
            if done:
                reward = -3
            
            
            self.bot.learn(self.observation, choice, new_observation, reward, time/(time_steps/2))
            
            self.observation = new_observation
            if done:
                self.observation = self.env.reset()

            self.bot.policy.print_stability()



        
    def test(self):
        while True:
            self.env.render()
            choice = np.argmax(self.bot.action())
            #print(choice)
            self.observation, reward, done, _ = self.env.step(choice)

            if done:
                self.env.reset()

