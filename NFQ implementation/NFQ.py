import torch
import numpy as np
from ann_designer import ANN


class NFQ:
    """
    An attempts at a Neural Fitted Q iteration implementation.

    Parameters
    ----------
    env : Gym environement
        OpenAI gym style environment
    discount : float
        The discount to use for future rewards
    epsilon : float
        The probability of taking a random action.
    """

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


    def train(self, no_episodes, no_epochs, batch_size, no_steps=1, random_action=False):
        """
        The main loop for performing episodes.

        Parameters
        ----------
        no_episodes : int
            The number of episodes to run.
        no_epochs : int
            The number of epochs to run when fitting the neural network.
        batch_size : int
            The batch size when fitting the neural network.
        no_steps : int, optional
            Max numbers of steps to take within each episode before 
            refitting the neural network, by default 1
        random_action : bool, optional
            Action selection based on a uniform random distribution (True)
            or based by an epislon-greedy policy (False).
            Setting this to True also disables fitting of the neural network, by default False
        """
        for ep in range(no_episodes):
            
            s = self.env.reset()
            done = False

            ep_len = 0
            while not done:
                # The number of steps to take before performing an update
                for cstep in range(no_steps):
                    if random_action:
                        a = np.random.choice(self.all_actions, size=1)[0]
                    else:
                        a = self.get_action(s=s)
                    s_, r, done, _ = self.env.step(a)
                    self.experience.append([s, a, r, s_])
                    s = s_

                    ep_len += 1

                    if done:
                        break

                if not random_action:
                    self.refit_Q(no_epochs, batch_size)

            print("Episode", ep, "\tLen", ep_len)

    def play_episodes(self, no_episodes, render=False):
        """
        Plays out a number of episodes by only selecting the greedy action.
        No updating or experiance recording performed.

        Parameters
        ----------
        no_episodes : int
            The number of episodes to play.
        render : bool, optional
            Whether or not to render the episodes for human watching.
        """
        for ep in range(no_episodes):
            s = self.env.reset()
            done = False
            if render:
                self.env.render()
            ep_len = 0
            while not done:
                a = self.get_greedy_action(s=s)
                s_, r, done, _ = self.env.step(a)
                if render:
                    self.env.render()
                s = s_
                ep_len +=1
            print("Episode", ep, "\tLen", ep_len)

    def get_action(self, s):
        """
        Gets the action according to an epsilon-greedy policy for
        the given state.

        Parameters
        ----------
        s : numpy array
            State representation.

        Returns
        -------
        int
            The action to take.
        """
        action_type = np.random.choice(
            a=["greedy", "random"], 
            size=1, 
            p=[1-self.epsilon, self.epsilon])[0]

        if action_type == "random":
            a = np.random.choice(self.all_actions, size=1)[0]
        else:
            a = self.get_greedy_action(s=s)
        
        return a

    def get_greedy_action(self, s):
        """
        Gets the greedy action for state s.

        Parameters
        ----------
        s : numpy array
            State representation.

        Returns
        -------
        int
            The greedy action.
        """
        max_action = 99
        max_action_v = -np.inf
        for action in self.all_actions:
            v = self.get_state_action_value(s=s, a=action)

            if v > max_action_v:
                max_action = action
                max_action_v = v
            # To break ties uniformly random
            elif v == max_action_v:
                rn_action = np.random.choice(a=[action, max_action], size=1)[0]
                if rn_action != max_action:
                    max_action = action
                    max_action_v = v
                
        a = max_action

        return a

    def get_state_action_value(self, s, a):
        """
        Calculates the state-action value for the provided state and action.

        Parameters
        ----------
        s : numpy array
            The state to evaluate.
        a : int
            The action evaluate.

        Returns
        -------
        float
            The state-action value.
        """
        x = self.create_one_prediction_sample(s, a)
        x = torch.as_tensor(x, dtype=torch.float)
        v = self.Q(x).item()

        return v

    def create_one_prediction_sample(self, s, a):
        """
        Creates a prediciton samlpe/training input in the correct format based on the given
        state and action. One-hot encodes the action.

        Parameters
        ----------
        s : numpy array
            State representation.
        a : int
            The action.

        Returns
        -------
        numpy array
            One training input.
        """
        actions = self.actions_oh.copy()
        actions[a] = 1
        sample = np.append(s, actions)

        return sample

    def calculate_target(self, r, s_):
        """
        Calculates the target value to use when fitting the neural network.

        Parameters
        ----------
        r : float
            The reward.
        s_ : numpy array
            The resulting state.

        Returns
        -------
        float
            The target value to fit the neural network on.
        """
        greedy_a = self.get_greedy_action(s=s_)
        greedy_v = self.get_state_action_value(s=s_, a=greedy_a)
        target = 1 + self.discount * greedy_v
        
        return target

    def create_training_data(self):
        """
        Creates a full set of training data based on all the experience collected so far.

        Returns
        -------
        pytorch tensors
            Two tensors: one for the input (x) and one for the target (y)
        """
        # Input: (s, a)
        x = [self.create_one_prediction_sample(s=item[0], a=item[1]) for item in self.experience]
        x = np.stack(x)
        x = torch.as_tensor(x, dtype=torch.float)

        # Target: r + discount * max_a(Q(s', a))
        y = [self.calculate_target(r=item[2], s_=item[3]) for item in self.experience]
        y = torch.as_tensor(y, dtype=torch.float)

        return x, y

    def refit_Q(self, no_epochs, batch_size=None):
        """
        Creates a new neural network and fit it to training data based 
        on the experience and the previous neural network's output.

        Parameters
        ----------
        no_epochs : int
            The number of epochs to train for.
        batch_size : int, optional
            The batch size to use in training. If set to None, then all
            training data will be used as one batch, by default None
        """
        x, y = self.create_training_data()
        self.Q = ANN().to("cpu")

        for e in range(no_epochs):
            self.Q.fit_one_epoch(x, y, batch_size)



    