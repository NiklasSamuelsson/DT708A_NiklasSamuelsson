import torch
import numpy as np
from ann_designer import ANN


class NFQ:

    def __init__(self, env, discount, epsilon):

        self.env = env
        self.discount = discount
        self.epsilon = epsilon
        self.experience = []
        self.Q = ANN().to("cpu")
        self.batch_size = None

        self.no_actions = env.action_space.n
        self.actions_oh = [0 for a in range(self.no_actions)]
        self.all_actions = [a for a in range(self.no_actions)]


    def train(self, no_episodes, no_epochs, no_steps=1):
        for ep in range(no_episodes):

            s = self.env.reset()
            done = False

            ep_len = 0
            while not done:
                # The number of steps to take before performing an update
                for cstep in range(no_steps):
                    #print("TRAIN")
                    a = self.get_action(s)
                    s_, r, done, _ = self.env.step(a)
                    self.experience.append([s, a, r, s_])

                x, y = self.create_training_data()

                self.refit_Q(x, y, no_epochs)

                ep_len += 1
            
            print("Episode", ep, "\tLen", ep_len)

    def play_episodes(self, no_episodes):
        for ep in range(no_episodes):
            s = self.env.reset()
            done = False
            self.env.render()
            while not done:
                a = self.get_greedy_action(s)
                s_, r, done, _ = self.env.step(a)
                self.env.render()

    def get_action(self, s):
        action_type = np.random.choice(a=["greedy", "random"], 
            size=1, 
            p=[1-self.epsilon, self.epsilon])[0]

        if action_type == "random":
            a = np.random.choice(self.all_actions, size=1)[0]
        else:
            a = self.get_greedy_action(s)
        
        return a

    def get_greedy_action(self, s):
        max_action = -99
        max_action_v = -np.inf
        for action in self.all_actions:
            v = self.get_state_action_value(s, action)

            if v > max_action_v:
                max_action = action
                max_action_v = v
                
        a = max_action
        #print(max_action_v)

        return a

    def get_state_action_value(self, s, a):
        x = self.create_one_prediction_sample(s, a)
        x = torch.as_tensor(x, dtype=torch.float)
        v = self.Q(x).item()

        return v

    def create_one_prediction_sample(self, s, a):
        actions = self.actions_oh.copy()
        actions[a] = 1
        sample = np.append(s, actions)

        return sample

    def create_training_data(self):
        # TODO: implement batch size
        self.batch_size = len(self.experience)

        # Input: (s, a)
        x = [self.create_one_prediction_sample(item[0], item[1]) for item in self.experience]
        x = np.stack(x)
        x = torch.as_tensor(x, dtype=torch.float)

        # Target: r + discount * max_a(Q(s', a))
        #print("CREATE DATA")
        y = [item[2] + self.discount * self.get_state_action_value(item[3], self.get_greedy_action(item[3])) for item in self.experience]
        y = torch.as_tensor(y, dtype=torch.float)

        return x, y

    def refit_Q(self, x, y, no_epochs):

        self.Q = ANN().to("cpu")

        for e in range(no_epochs):
            self.Q.fit_one_epoch(x, y)





    

    