import torch


class ANN(torch.nn.Module):
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

    def __init__(self, in_features, out_features, lr=0.0001):
        
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.lr = lr

        self.seq = torch.nn.Sequential()

        no_neurons = 32

        self.seq.append(torch.nn.Linear(in_features=in_features, out_features=no_neurons))
        self.seq.append(torch.nn.ReLU())

        self.seq.append(torch.nn.Linear(in_features=no_neurons, out_features=no_neurons))
        self.seq.append(torch.nn.ReLU())

        self.seq.append(torch.nn.Linear(in_features=no_neurons, out_features=no_neurons))
        self.seq.append(torch.nn.ReLU())

        self.seq.append(torch.nn.Linear(in_features=no_neurons, out_features=out_features))

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=lr
        )

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

        a = torch.tensor(a)
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
