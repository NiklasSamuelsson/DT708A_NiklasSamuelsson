import random
import numpy as np
import copy
import torch


class DQN:
    """
    An implementation of Deep-Q-learning using artificial artificial neural networks 
    as state-action value function approximators.

    Parameters
    ----------
    env : Gym environement
        OpenAI gym style environment.
    epsilon : float
        The starting probability of taking a random action.
    epsilon_min : float
        The minimum value that epsilon can take.
    epsilon_decay : float
        The rate at which epsilon will decay with each timestep.
    discount : float
        The discount rate used for next state's maximum state-action value. Also known as gamma.
    replay_memory_size : int
        The size of the replay memory in which experience is stored.
    batch_size : int
        The number of samples to use when updating the policy network (Q).
    reset_target_ANN_updates : int
        The number of timesteps between each synchronization of Q_hat and Q.
    ANN : Artificial Neural Network, implemented in PyTorch
        The network to use for Q and Q_hat.
    """

    def __init__(
        self, 
        env, 
        epsilon,
        epsilon_min,
        epsilon_decay,
        discount, 
        replay_memory_size, 
        batch_size, 
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
        self.reset_target_ANN_updates = reset_target_ANN_updates
        self.Q = ANN
        
        self.replay_memory = [[], [], [], [], []]
        self.total_updates = 0
        self.Q_hat = copy.deepcopy(self.Q)
        self.all_actions = [a for a in range(self.env.action_space.start, self.env.action_space.n)]

    def train(self, no_episodes, init_replay_memory=False):
        """
        Trains the agent or generated experience.
        Will stop if the critiera for solving the enviroment is met.

        Parameters
        ----------
        no_episodes : int
            The number of episodes to train the agent for.
        init_replay_memory : bool, optional
            Whether to train the agent (False) or just generate random experience (True), by default False
        """
        sum_train_ep_len = 0
        sum_test_ep_len = 0
        for episode in range(no_episodes):
            train_ep_len = self.train_one_episode(init_replay_memory)
            sum_train_ep_len += train_ep_len
            avg_train_ep_len = sum_train_ep_len/(episode+1)

            test_ep_len = self.play_one_episode()
            sum_test_ep_len += test_ep_len
            avg_test_ep_len = sum_test_ep_len/(episode+1)
            if not init_replay_memory:
                print(
                    "Episode:", episode, 
                    "\tTraining length:", train_ep_len, 
                    "\tAvg train length:", avg_train_ep_len, 
                    "\tTest length:", test_ep_len,
                    "\tAvg test length:", avg_test_ep_len,
                    "\tEpsilon:", self.epsilon, 
                    "\tExp buffer size:", len(self.replay_memory[0]))

            # Evaluate if solved
            # Currently very CartPole-v1 specific
            if episode%100 == 0:
                solved_sum = 0
                solved_no_ep = 100
                for i in range(solved_no_ep):
                    solved_sum += self.play_one_episode(verbose=False)
                if solved_sum/solved_no_ep >= 495:
                    print("SOLVED")
                    break

    def train_one_episode(self, init_replay_memory):
        """
        Trains the agent or generated random experience for one epsiode.

        Parameters
        ----------
        init_replay_memory : bool
            Whether to actually train the agent (Fale) or just generate random experience (True.)

        Returns
        -------
        int
            The length of the episode (number of timesteps).
        """
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

    def play_one_episode(self, render=False, verbose=False):
        """
        Plays one episode by greedy action selection.

        Parameters
        ----------
        render : bool, optional
            Whether to render the enviroment each timestep, by default False
        verbose : bool, optional
            Whether to print statistics, by default False

        Returns
        -------
        int
            The length of the episode (number of timesteps).
        """
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
        
        if verbose:
            print("Episode length:", ep_len)

        return ep_len

    def get_action(self, s):
        """
        Get an action from the current policy.

        Parameters
        ----------
        s : numpy array
            An array describing the state in which to take an action.

        Returns
        -------
        int
            The action to take.
        """
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)
        if random.uniform(0, 1) <= self.epsilon:
            a = random.choice(self.all_actions)
        else:
            a = self.get_greedy_action(s)
        
        return a

    def get_greedy_action(self, s):
        """
        Gets the greedy action from the current policy.

        Parameters
        ----------
        s : numpy array
            An array describing the state in which to take an action.

        Returns
        -------
        int
            The greedy action.
        """
        s = torch.tensor(s).reshape((1, s.shape[0]))
        _, a = self.Q.predict_max(s)

        return a.item()

    def fit_ANN(self):
        """
        Fits the policy network (Q) for one epoch.
        """
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
        #   Currently works for CartPole-v1, but is maybe not general
        self.Q.train_one_epoch(s, a, target)

    def sample_random_batch(self):
        """
        Provides samples according to the batch size.

        Returns
        -------
        5 lists
            Lists with sampels for the states, actions, rewards, next states and termination flag.
        """
        idx = random.sample(range(len(self.replay_memory[0])), self.batch_size)
        s = [self.replay_memory[0][i] for i in idx]
        a = [self.replay_memory[1][i] for i in idx]
        r = [self.replay_memory[2][i] for i in idx]
        s_ = [self.replay_memory[3][i] for i in idx]
        done = [self.replay_memory[4][i] for i in idx]

        return s, a, r, s_, done
        
    def calculate_target(self, r, s_, done):
        """
        _summary_

        Parameters
        ----------
        r : list
            List of rewards.
        s_ : PyTorch tensor
            Tensor desribing next states.
        done : list
            List of boolean termination flags.

        Returns
        -------
        numpy array
            An array containing the target for which to fit the policy network (Q).
        """
        v, _ = self.Q_hat.predict_max(s_)
        target = np.add(r, np.multiply(self.discount, v.detach().numpy()))

        # Don't reward agent when ending episode
        target = np.where(done, 0, target)
        
        return target
