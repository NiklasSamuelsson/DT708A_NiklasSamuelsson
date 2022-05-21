from torch import nn
from torch import squeeze
from torch.optim import SGD, Rprop, Adam


class ANN(nn.Module):
    """
    An artificial neural network.
    Implemented as a multi layer perceptron.
    """

    def __init__(self):
        super().__init__()

        self.seq = nn.Sequential()

        no_neurons = 20

        self.seq.append(nn.Linear(in_features=6, out_features=no_neurons))
        self.seq.append(nn.Sigmoid())

        self.seq.append(nn.Linear(in_features=no_neurons, out_features=no_neurons))
        self.seq.append(nn.Sigmoid())

        self.seq.append(nn.Linear(in_features=no_neurons, out_features=no_neurons))
        self.seq.append(nn.Sigmoid())

        self.seq.append(nn.Linear(in_features=no_neurons, out_features=no_neurons))
        self.seq.append(nn.Sigmoid())

        self.seq.append(nn.Linear(in_features=no_neurons, out_features=1))
        #self.seq.append(nn.Sigmoid())

        self.loss_fn = nn.MSELoss()
        self.optimizer = Rprop(
            params=self.parameters(),
            lr=0.001
        )

    def forward(self, x):
        """
        Performs a forward pass through the network with the provided data.

        Parameters
        ----------
        x : pytorch tensor
            The input data to predcit upon.

        Returns
        -------
        pytorch tensor
            The output from the model.
        """
        output = squeeze(self.seq(x))

        return output

    def fit_one_epoch(self, x, y, batch_size=None):
        """
        Fits the neural network for one epoch.

        Parameters
        ----------
        x : pytorch tensor
            The input data to the neural network.
        y : pytorch tensor
            The target data.
        batch_size : int, optional
            The batch size to use in training. If set to None, then all
            training data will be used as one batch, by default None
        """
        stop = x.shape[0]
        if batch_size is None:
            batch_size = stop
            
        for b in range(0, stop, batch_size):
            x_ = x[b:b+batch_size]
            y_ = y[b:b+batch_size]

            pred = self(x_)
            loss = self.loss_fn(pred, y_)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()