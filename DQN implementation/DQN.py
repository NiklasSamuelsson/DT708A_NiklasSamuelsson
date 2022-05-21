import random
import numpy as np

class DQN:

    def __init__(self, env, epsilon, discount, replay_memory_size, batch_size, loss_threshold):

        self.env = env
        self.epsilon = epsilon
        self.discount = discount
        self.replay_memory_size = replay_memory_size
        self.batch_size = self.batch_size
        self.loss_threshold = loss_threshold
        
        self.replay_memory = [[], [], [], [], []]

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
            # Reset Q_hat to Q every C steps

            s = s_
            ep_len += 1
        
        return ep_len

    def get_action(self):
        pass

    def fit_ANN(self):
        state, action, reward, state_, done = self.sample_random_batch()
        target = self.calculate_target(
            action=action,
            reward=reward,
            state_=state_,
            done=done
        )
        loss = 99
        n_epochs = 0
        while loss > self.loss_threshold:
        # Loop epochs until convergence
            loss = self.Q.fit(state, action, target)
            n_epochs += 1

    def sample_random_batch(self):
        idx = random.sample(range(len(self.replay_memory)), self.batch_size)
        state = [self.replay_memory[0][i] for i in idx]
        action = [self.replay_memory[1][i] for i in idx]
        reward = [self.replay_memory[2][i] for i in idx]
        state_ = [self.replay_memory[3][i] for i in idx]
        done = [self.replay_memory[4][i] for i in idx]

        return state, action, reward, state_, done
        
    def calculate_target(self, action, reward, state_, done):
        # TODO: fix input to ANN
        pred = self.Q_hat(state_, action)
        target = np.add(reward, pred)
        target = np.where(done, reward, target)
        
        return target
