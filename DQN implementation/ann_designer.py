import torch


class ANNBase(torch.nn.Module):
    """
    An base for Artificial Neural Network structures to be used for DQN.
    """

    def __init__(self):
        super().__init__()

        self.build_network()

    def build_network(self):
        """
        Needs to assign an archeture to self.seq as a torch.nn.Sequential() object
        with a loss function and optimizer assign.

        Raises
        ------
        NotImplementedError
            Needs to be implemented with the desired architecture.
        """
        raise NotImplementedError("Need to implement a arhcitecture for the ANN.")

    def forward(self, x):
        """
        Performs a forward pass through the network with the provided data.

        Parameters
        ----------
        x : PyTorch tensor
            The input data to predcit upon.

        Returns
        -------
        PyTorch tensor
            The output from the model.
        """
        pred = self.seq(x)

        return pred

    def predict_max(self, x):
        """
        Predict the maximum state-action value.

        Parameters
        ----------
        x : PyTorch tensor
            The input data to predict upon.

        Returns
        -------
        PyTorch tensors
            The value and the corresponding action.
        """
        pred = self(x)
        v, a = torch.max(pred, dim=1)

        return v, a

    def predict_given_actions(self, x, a):
        """
        Provides prediction for only the provided actions.

        Parameters
        ----------
        x : PyTorch tensor
            The input data to predict upon.
        a : list
            A list of action indicies to predict for.

        Returns
        -------
        PyTorch tensor
            The predictions.
        """
        pred = self(x)

        aux = torch.ones(len(a), dtype=torch.int).cumsum(0)
        a += aux * self.out_features - self.out_features
        pred = pred.flatten()
        pred = torch.index_select(pred, 0, a)

        return pred

    def train_one_epoch(self, x, a, y):
        """
        Trains the model based for the provided actions.

        Parameters
        ----------
        x : PyTorch tensor
            The input data to predict upon.
        a : list
            A list of action indicies to train for.
        y : PyTorch tensor
            The target values.
        """
        pred = self.predict_given_actions(x, a)

        loss = self.loss_fn(pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class SimpleMLP(ANNBase):
    """
    An Artificial Neural Network to be used for DQN.

    Parameters
    ----------
    in_features : int
        The number of input featurs to the network.
    out_features : int
        The number of output featurs to the network.
    lr : float, optional
        The learning rate, by default 0.0001
    """

    def __init__(self, in_features, out_features, lr):
        self.in_features = in_features
        self.out_features = out_features
        self.lr = lr

        super().__init__()

    def build_network(self):
        """
        Contructs the network.
        """
        self.seq = torch.nn.Sequential()

        no_neurons = 32

        self.seq.append(torch.nn.Linear(in_features=self.in_features, out_features=no_neurons))
        self.seq.append(torch.nn.ReLU())

        self.seq.append(torch.nn.Linear(in_features=no_neurons, out_features=no_neurons))
        self.seq.append(torch.nn.ReLU())

        self.seq.append(torch.nn.Linear(in_features=no_neurons, out_features=no_neurons))
        self.seq.append(torch.nn.ReLU())

        self.seq.append(torch.nn.Linear(in_features=no_neurons, out_features=self.out_features))

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.lr
        )


class ImageMLP(ANNBase):

    def __init__(self, out_features, lr):
        self.out_features = out_features
        self.lr = lr

        super().__init__()

    def build_network(self):
        self.seq = torch.nn.Sequential()

        self.seq.append(torch.nn.Flatten())

        # Lazy way of finding the right shape
        x = torch.randn(1, 1, 100, 150)
        in_features_fc = self.seq(x).shape[1]

        self.seq.append(torch.nn.Linear(in_features=in_features_fc, out_features=700))
        self.seq.append(torch.nn.ReLU())

        self.seq.append(torch.nn.Linear(in_features=700, out_features=125))
        self.seq.append(torch.nn.ReLU())

        self.seq.append(torch.nn.Linear(in_features=125, out_features=50))
        self.seq.append(torch.nn.ReLU())

        self.seq.append(torch.nn.Linear(in_features=50, out_features=self.out_features))

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.lr
        )


class CNN(ANNBase):

    def __init__(self, in_channels, out_features, lr):
        self.in_channels = in_channels
        self.out_features = out_features
        self.lr = lr

        super().__init__()

    def build_network(self):

        # Architecture inspired by: https://github.com/LM095/CartPole-CNN
        self.seq = torch.nn.Sequential()

        self.seq.append(torch.nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=5, stride=3, padding=2))
        self.seq.append(torch.nn.LeakyReLU())
        # TODO: leaky ReLU instead

        self.seq.append(torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=3, padding=2))
        self.seq.append(torch.nn.LeakyReLU())

        self.seq.append(torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=2))
        self.seq.append(torch.nn.LeakyReLU())

        self.seq.append(torch.nn.Flatten())

        # Lazy way of finding the right shape
        x = torch.randn(1, self.in_channels, 160, 240)
        in_features_fc = self.seq(x).shape[1]

        self.seq.append(torch.nn.Linear(in_features=in_features_fc, out_features=256))
        self.seq.append(torch.nn.LeakyReLU())

        self.seq.append(torch.nn.Linear(in_features=256, out_features=128))
        self.seq.append(torch.nn.LeakyReLU())

        self.seq.append(torch.nn.Linear(in_features=128, out_features=32))
        self.seq.append(torch.nn.LeakyReLU())

        self.seq.append(torch.nn.Linear(in_features=32, out_features=self.out_features))

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.lr
        )


if __name__ == "__main__":
    import torchvision
    import matplotlib.pyplot as plt
    import gym
    import numpy as np
    
    env = gym.make("CartPole-v1")
    env.reset()
    s = env.render(mode="rgb_array")

    # Retain colors
    s = env.render(mode="rgb_array")
    s = np.moveaxis(s, 2, 0)
    s = torch.tensor(s)
    transform = torchvision.transforms.Resize(100)
    resized_img = transform(s)
    resized_img = torch.moveaxis(resized_img, 0, 2)
    plt.imshow(resized_img)

    # Convert to grayscale
    s = env.render(mode="rgb_array")
    s = np.moveaxis(s, 2, 0)
    s = torch.tensor(s)
    gray = torchvision.transforms.functional.rgb_to_grayscale(s)
    transform = torchvision.transforms.Resize(100)
    resized_img = transform(gray)
    plt.imshow(torch.squeeze(resized_img), cmap="gray")

    # Convert all non-whites to black
    s = env.render(mode="rgb_array")
    s[s<255] = 0
    s = s/255
    s = np.moveaxis(s, 2, 0)
    s = torch.tensor(s)
    gray = torchvision.transforms.functional.rgb_to_grayscale(s)
    transform = torchvision.transforms.Resize(100)
    resized_img = transform(gray)
    plt.imshow(torch.squeeze(resized_img), cmap="gray")


