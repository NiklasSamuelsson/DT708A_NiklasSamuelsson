import random
import numpy as np
import copy
import torch


class DQN:

    def __init__(
        self, 
        env, 
        epsilon,
        epsilon_min,
        epsilon_decay,
        discount, 
        replay_memory_size, 
        batch_size, 
        loss_threshold, 
        reset_target_ANN_updates,
        ANN
    ):

        self.env = env
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
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
        self.one_hot = []
        for a in range(env.action_space.n):
            self.one_hot.append([0 for i in range(env.action_space.n)])
            self.one_hot[a][a] = 1

    def train(self, no_episodes, init_replay_memory=False):
        sum_ep_len = 0
        for episode in range(no_episodes):
            ep_len = self.train_one_episode(init_replay_memory)
            sum_ep_len += ep_len
            avg_ep_len = sum_ep_len/(episode+1)
            if not init_replay_memory:
                print("Episode:", episode, "\tLength:", ep_len, "\tAvg length:", avg_ep_len, "\tEpsilon:", self.epsilon, "\tExp buffer size:", len(self.replay_memory[0]))

    def train_one_episode(self, init_replay_memory):
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
            # Hack to set the right reward later for CartPole-v1
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

    def play_one_episode(self, render=False):
        ep_len = 0
        s = self.env.reset()
        done = False
        while not done:
            if render:
                self.env.render()
            a = self.get_greedy_action(s)
            s_, r, done, _ = self.env.step(a)
            s = s_
            ep_len += 1
        
        print("Episode length:", ep_len)

    def get_action(self, s):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)
        if random.uniform(0, 1) <= self.epsilon:
            a = random.choice(self.all_actions)
        else:
            a = self.get_greedy_action(s)
        
        return a

    def get_greedy_action(self, s):
        best_av = 0
        best_a = 0
        for a, oh in enumerate(self.one_hot):
            s_mod = np.append(s, oh)
            s_mod = torch.tensor(s_mod, dtype=torch.float).reshape((1, s_mod.shape[0]))
            v = self.Q(s_mod).item()

            if v > best_av:
                best_av = v
                best_a = a

        return best_a

    def fit_ANN(self):
        s, a, r, s_, done = self.sample_random_batch()
        state_actions = []
        values = []
        # Predict all state-action values for selected states to select the maximizing action
        for oh in self.one_hot:
            s_mod = np.append(np.array(s_), [oh for i in range(np.array(s_).shape[0])], axis=1)
            state_actions.append(s_mod)
            s_mod_t = torch.tensor(s_mod, dtype=torch.float)
            v = self.Q_hat(s_mod_t)
            values.append(v.detach().numpy())

        # Create target data
        max_av = np.max(values, axis=0).reshape(len(values[0]))
        target = self.calculate_target(
            max_av=max_av,
            r=r,
            done=done
        )
        target = torch.tensor(target, dtype=torch.float)

        # Create input data
        oh_a = np.array([self.one_hot[i] for i in a])
        x = np.hstack((np.array(s), oh_a))
        x = torch.tensor(x, dtype=torch.float)

        loss = self.loss_threshold + 1
        n_epochs = 0
        while loss > self.loss_threshold and n_epochs < 200:
            loss = self.Q.train_one_epoch(x, target)
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
        
    def calculate_target(self, max_av, r, done):
        target = np.add(r, np.multiply(self.discount, max_av))
        # Don't reward agent when ending episode
        target = np.where(done, 0, target)
        
        return target
