from NeuralNetworks import forward
import numpy as np

class Agent:
    def __init__(self, gamma, init_state):
        self.state = init_state
        self.policy = forward.ReinforceTraitNetwork((0.0001,0.9),(4, ["tanh",24],["tanh",24],["linear",2]))
        self.gamma = gamma

    def action(self):
        # Take an action
        return self.policy.predict(np.array([self.state]))[0]

    def learn(self, state, choice, next_state, reward, effect):
        # Learning math
        self.state = next_state
        # The action we are training on update
        truth = reward+(effect*self.gamma*np.max(self.policy.predict(np.array([next_state]))[0]))

        self.policy.train(state, [(choice, truth)])
