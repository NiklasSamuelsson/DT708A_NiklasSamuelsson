import torch


class ANN(torch.nn.Module):

    def __init__(self, in_features, out_features, lr=0.0001):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.lr = lr

        self.seq = torch.nn.Sequential()

        no_neurons = 20

        self.seq.append(torch.nn.Linear(in_features=in_features, out_features=no_neurons))
        self.seq.append(torch.nn.Sigmoid())

        self.seq.append(torch.nn.Linear(in_features=no_neurons, out_features=no_neurons))
        self.seq.append(torch.nn.Sigmoid())

        self.seq.append(torch.nn.Linear(in_features=no_neurons, out_features=no_neurons))
        self.seq.append(torch.nn.Sigmoid())

        self.seq.append(torch.nn.Linear(in_features=no_neurons, out_features=out_features))

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=lr
        )

    def forward(self, x):
        pred = self.seq(x)

        return pred

    def predict_max(self, x):
        pred = self(x)
        v, a = torch.max(pred, dim=1)

        return v, a

    def predict_given_actions(self, x, a):
        pred = self(x)

        a = torch.tensor(a)
        aux = torch.ones(len(a), dtype=torch.int).cumsum(0)
        a += aux * self.out_features - self.out_features
        pred = pred.flatten()
        pred = torch.index_select(pred, 0, a)

        return pred

    def train_one_epoch(self, x, a, y):
        pred = self.predict_given_actions(x, a)

        loss = self.loss_fn(pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        pred = self.predict_given_actions(x, a)
        loss = self.loss_fn(pred, y)

        return loss


if __name__ == "__main__":
    x = torch.randn(5, 4)
    y = torch.randn(5)

    a = [0, 1, 1, 0, 1]
    b = [1, 0, 0, 1, 1]

    model = ANN(4, 2)