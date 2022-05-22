import torch


class ANN(torch.nn.Module):

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
        pred = self.seq(x)

        return pred

    def train_one_epoch(self, x, y):
        pred = self(x)

        loss = self.loss_fn(pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        pred = self(x)
        loss = self.loss_fn(pred, y)

        return loss

