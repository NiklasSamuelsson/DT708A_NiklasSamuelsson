import random
import numpy as np
import copy

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

    def train(self, no_episodes):
        episode = 0
        for episode in range(no_episodes):
            ep_len = self.train_one_episode()
            print("Episode:", episode, "\tLenght:", ep_len)
            episode += 1

    def train_one_episode(self):
        ep_len = 0
        s = self.env.reset()
        done = False
        while not done:
            a = self.get_action(s)
            s_, r, done, _ = self.env.step(a)
            self.replay_memory[0].append(s)
            self.replay_memory[1].append(a)
            self.replay_memory[2].append(r)
            self.replay_memory[3].append(s_)
            self.replay_memory[4].append(done)

            if len(self.replay_memory[0]) == self.replay_memory_size:
                for i in range(len(self.replay_memory)):
                    self.replay_memory[i].pop(0)

            self.fit_ANN()
            self.total_updates += 1
            if self.total_updates%self.reset_target_ANN_updates == 0:
                # TODO: make sure this creates a copy and not a reference
                self.Q_hat = copy.deepcopy(self.Q)

            s = s_
            ep_len += 1
        
        return ep_len

    def get_action(self, s):

        if random.uniform(0, 1) <= self.epsilon:
            # Select random action
            a = 0
        else:
            a = self.get_greedy_action(s)
        
        return a

    def get_greedy_action(self, s):
        return self.Q(s)

    def fit_ANN(self):
        s, a, r, s_, done = self.sample_random_batch()
        target = self.calculate_target(
            a=a,
            r=r,
            s_=s_,
            done=done
        )
        loss = 99
        n_epochs = 0
        while loss > self.loss_threshold:
        # Loop epochs until convergence
            loss = self.Q.train_one_epoch(s, target)
            n_epochs += 1

    def sample_random_batch(self):
        idx = random.sample(range(len(self.replay_memory)), self.batch_size)
        s = [self.replay_memory[0][i] for i in idx]
        a = [self.replay_memory[1][i] for i in idx]
        r = [self.replay_memory[2][i] for i in idx]
        s_ = [self.replay_memory[3][i] for i in idx]
        done = [self.replay_memory[4][i] for i in idx]

        return s, a, r, s_, done
        
    def calculate_target(self, a, r, s_, done):
        pred = self.Q_hat(s_)
        target = np.add(r, np.multiply(self.discount, pred))
        target = np.where(done, r, target)
        
        return target
