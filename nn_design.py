from torch import nn
from torch import randn


class CNNMNIST(nn.Module):
    """
    Constructs a neural network model aimed to classify the MNIST
    dataset. Starts with convolutional 2D layers and ends with
    fully connected layers. 

    Almost all layers have ReLu as the activation function applied.
    """

    def __init__(self, conv_layers, fc_layers):
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
        """
        super().__init__()

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

        # Add a final output layer
        self.seq.append(nn.Linear(in_features=in_features, out_features=10))

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