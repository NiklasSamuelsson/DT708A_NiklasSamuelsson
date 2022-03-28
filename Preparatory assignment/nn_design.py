from torch import nn
from torch import randn
import numpy as np
from dataloaders import get_MNIST_dataloaders


class CNNMNIST(nn.Module):
    """
    Constructs a neural network model aimed to classify the MNIST
    dataset. Starts with convolutional 2D layers and ends with
    fully connected layers. 

    Almost all layers have ReLu as the activation function applied.
    """

    def __init__(self, conv_layers, fc_layers, optimizer, learning_rate,
     loss_function, batch_size):
        """
        Let's the user configure the network.

        Parameters
        ----------
        conv_layers : list of dicts
            A list which contains a number of dicts which is equal
            to the number of convolutional 2D layers to add.

            Each dict also contains kwargs for the Conv2d layers
            in PyTorch's module torch.nn. 
        fc_layers : list of dicts
            A list which contains a number of dicts which is equal
            to the number of fully connected (dense) layers to add.
            
            Each dict also contains kwargs for the Conv2d layers
            in PyTorch's module torch.nn. 
        optimizer : optimizer class from PyTorch
            Which optimizer class to use.
        learning_rate : float
            Constant learning rate.
        loss_function : loss function class from PyTorch
            Which loss function to use when optimizing and evaluating.
        batch_size : int
            Batch size.
        """
        super().__init__()

        # Placeholders for testing stats
        self.avgloss = []
        self.totacc = []

        self.train_dl, self.test_dl = get_MNIST_dataloaders(batch_size)

        self.seq = nn.Sequential()

        # Add convolutional layers first
        in_channels = 1
        for clayer_config in conv_layers:
            clayer_config["in_channels"] = in_channels
            self.seq.append(nn.Conv2d(**clayer_config))
            in_channels = clayer_config["out_channels"]
            self.seq.append(nn.ReLU())

        # Flatten so we can add dense layers
        self.seq.append(nn.Flatten())

        # Lazy way of finding the right shape
        x = randn(1, 1, 28, 28)
        in_features = self.seq(x).shape[1]

        for fclayer_config in fc_layers:
            fclayer_config["in_features"] = in_features
            self.seq.append(nn.Linear(**fclayer_config))
            in_features = fclayer_config["out_features"]
            self.seq.append(nn.ReLU())

        self.seq.append(nn.Linear(in_features=in_features, out_features=10))

        self.loss_fn = loss_function()
        self.optimizer = optimizer(
            params=self.parameters(),
            lr=learning_rate
        )

    def forward(self, x):
        """
        Lets the model perform a forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for prediction.

        Returns
        -------
        torch.Tensor
            The prediction in probabilities.
        """
        logits = self.seq(x)
        return logits

    def train_one_epoch(self):
        """ Performs one epoch of training """
        for x, y in self.train_dl:
            pred = self(x)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def test_one_epoch(self):
        """
        Performs one epoch of testing.

        Returns
        -------
        floats
            The average loss and the accuracy for the epoch.
        """
        no_batches = len(self.test_dl)
        loss = 0

        size = len(self.test_dl.dataset)
        accuracy = 0

        for x, y in self.test_dl:
            pred = self(x)
            loss += self.loss_fn(pred, y).item()
            accuracy += (pred.argmax(1) == y).sum().item()

        loss /= no_batches
        accuracy /= size

        return loss, accuracy

    def fit(self, convergence_treshold, window_1_size, window_2_size,
        max_no_epochs, warmup_periods):
        """
        Trains the model until it either reaches the max no. of epochs
        or until it converges.

        The convergance is based on two rolling windows of the average 
        test accuracy. If the difference between the windows is less than
        the convergance threshold, the training is stopped. 

        Parameters
        ----------
        convergence_treshold : float
            If the change is smaller than this, training is stopped.
        window_1_size : int
            The size of the larger of the two windows to test for
            convergence.
        window_2_size : int
            The size of the smaller of the two windows to test for
            convergence.
        max_no_epochs : int
            The maximum number of epochs allowed before training is stopped.
        warmup_periods : int
            The no. starting periods where the convergence threshold 
            is not considered.
        """

        self.avgloss = []
        self.totacc = []

        losschange = 1
        accchange = 1
        epoch = 1

        while (accchange > convergence_treshold) and (
            epoch <= max_no_epochs):
            self.train_one_epoch()

            closs, caccuracy = self.test_one_epoch()

            print("Epoch", epoch)
            print("Test accuracy:", caccuracy)

            self.avgloss.append(closs)
            self.totacc.append(caccuracy)

            avglossr1 = np.mean(self.avgloss[-window_1_size:])
            avglossr2 = np.mean(self.avgloss[-window_2_size:])
            losschange = (avglossr1-avglossr2)/avglossr1

            totaccr1 = np.mean(self.totacc[-window_1_size:])
            totaccr2 = np.mean(self.totacc[-window_2_size:])
            accchange = (totaccr2-totaccr1)/totaccr1

            print("Change in loss function:", losschange)
            print("Change in accuracy:", accchange)

            if epoch <= max(1, warmup_periods):
                losschange = 1
                accchange = 1
                
            epoch += 1

    def confusion_matrix(self):
        """
        Calculates a confusion matrix in percentages.

        Returns
        -------
        Numpy array
            A confusion matrix.
        """
        # TODO: just use the one from sklearn
        conf = np.zeros((10, 10))
        yfreq = np.zeros(10)

        for x, y in self.test_dl:
            pred = self(x)

            for l_, p_ in zip(y, pred):
                l = l_.item()
                p = p_.argmax().item()

                conf[l][p] += 1
                yfreq[l] += 1

        #conf = conf/yfreq

        return conf



                
