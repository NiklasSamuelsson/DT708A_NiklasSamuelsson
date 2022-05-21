import random
import numpy as np
import copy
import torch


class DQN:

    def __init__(
        self, 
        env, 
        epsilon, 
        discount, 
        replay_memory_size, 
        batch_size, 
        loss_threshold, 
        reset_target_ANN_updates,
        ANN
    ):

        self.env = env
        self.epsilon = epsilon
        self.discount = discount
        self.replay_memory_size = replay_memory_size
        self.batch_size = batch_size
        self.loss_threshold = loss_threshold
        self.reset_target_ANN_updates = reset_target_ANN_updates
        self.Q = ANN
        
        self.replay_memory = [[], [], [], [], []]
        self.total_updates = 0
        self.Q_hat = copy.deepcopy(self.Q)
        self.all_actions = [a for a in range(self.env.action_space.start, self.env.action_space.n)]

    def train(self, no_episodes, init_replay_memory=False):
        episode = 0
        for episode in range(no_episodes):
            ep_len = self.train_one_episode(init_replay_memory)
            print("Episode:", episode, "\tLength:", ep_len)
            episode += 1

    def train_one_episode(self, init_replay_memory=False):
        ep_len = 0
        s = self.env.reset()
        done = False
        while not done:
            if init_replay_memory:
                a = self.env.action_space.sample()
            else:
                a = self.get_action(s)
            s_, r, done, _ = self.env.step(a)
            self.replay_memory[0].append(s)
            self.replay_memory[1].append(a)
            self.replay_memory[2].append(r)
            self.replay_memory[3].append(s_)
            # Hack to set the right reward later
            if ep_len >= 500:
                self.replay_memory[4].append(False)
            else:
                self.replay_memory[4].append(done)

            if len(self.replay_memory[0]) == self.replay_memory_size:
                for i in range(len(self.replay_memory)):
                    self.replay_memory[i].pop(0)

            if not init_replay_memory:
                self.fit_ANN()
                self.total_updates += 1
                if self.total_updates%self.reset_target_ANN_updates == 0:
                    self.Q_hat = copy.deepcopy(self.Q)

            s = s_
            ep_len += 1
        
        return ep_len

    def get_action(self, s):
        if random.uniform(0, 1) <= self.epsilon:
            a = random.choice(self.all_actions)
        else:
            a = self.get_greedy_action(s)
        
        return a

    def get_greedy_action(self, s):
        s = torch.tensor(s).reshape((1, s.shape[0]))
        _, a = self.Q.predict_max(s)

        return a.item()

    def fit_ANN(self):
        s, a, r, s_, done = self.sample_random_batch()
        s_ = torch.tensor(np.array(s_))
        target = self.calculate_target(
            r=r,
            s_=s_,
            done=done
        )

        s = torch.tensor(np.array(s))
        a = torch.tensor(np.array(a))
        target = torch.tensor(target, dtype=torch.float)

        # TODO: make sure so that a's actions is reindexed to start at zero (in case env don't start at 0)
        loss = 99
        n_epochs = 0
        while loss > self.loss_threshold:
            loss = self.Q.train_one_epoch(s, a, target)
            n_epochs += 1
        #print("Epochs:", n_epochs)

    def sample_random_batch(self):
        idx = random.sample(range(len(self.replay_memory[0])), self.batch_size)
        s = [self.replay_memory[0][i] for i in idx]
        a = [self.replay_memory[1][i] for i in idx]
        r = [self.replay_memory[2][i] for i in idx]
        s_ = [self.replay_memory[3][i] for i in idx]
        done = [self.replay_memory[4][i] for i in idx]

        return s, a, r, s_, done
        
    def calculate_target(self, r, s_, done):
        v, _ = self.Q_hat.predict_max(s_)
        target = np.add(r, np.multiply(self.discount, v.detach().numpy()))
        # Don't reward agent when ending episode
        target = np.where(done, 0, target)
        
        return target
