from torch import nn
from torch import squeeze
from torch.optim import SGD, Rprop, Adam
import numpy as np


class ANN(nn.Module):

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

        self.seq.append(nn.Linear(in_features=no_neurons, out_features=1))
        self.seq.append(nn.Sigmoid())

        self.loss_fn = nn.MSELoss()
        self.optimizer = Adam(
            params=self.parameters(),
            lr=0.00001
        )

    def forward(self, x):

        output = squeeze(self.seq(x))

        return output

    def fit_one_epoch(self, x, y, batch_size=None):
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