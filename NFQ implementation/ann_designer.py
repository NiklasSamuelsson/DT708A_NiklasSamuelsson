from torch import nn
from torch import squeeze
from torch.optim import SGD, Rprop, Adam


class ANN(nn.Module):

    def __init__(self):
        super().__init__()

        self.seq = nn.Sequential()

        no_neurons = 50

        self.seq.append(nn.Linear(in_features=6, out_features=no_neurons))
        self.seq.append(nn.Sigmoid())

        self.seq.append(nn.Linear(in_features=no_neurons, out_features=no_neurons))
        self.seq.append(nn.Sigmoid())

        self.seq.append(nn.Linear(in_features=no_neurons, out_features=no_neurons))
        self.seq.append(nn.Sigmoid())

        self.seq.append(nn.Linear(in_features=no_neurons, out_features=no_neurons))
        self.seq.append(nn.Sigmoid())

        self.seq.append(nn.Linear(in_features=no_neurons, out_features=1))
        self.seq.append(nn.Sigmoid())

        self.loss_fn = nn.MSELoss()
        self.optimizer = Rprop(
            params=self.parameters(),
            lr=0.001
        )

    def forward(self, x):

        output = squeeze(self.seq(x))

        return output

    def fit_one_epoch(self, x, y):
        # Iterate over batches
        # TODO: implement batch size
        #for x_, y_ in zip(x, y):

        pred = self(x)
        loss = self.loss_fn(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()