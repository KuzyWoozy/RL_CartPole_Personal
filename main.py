import gym
import Environment.OpenAI
import Agent.CartPole

environment = Environment.OpenAI.Environment("CartPole-v1", Agent.CartPole.Agent, debug=False)


environment.train(100000)
environment.test()








