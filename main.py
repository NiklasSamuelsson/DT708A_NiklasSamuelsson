from nn_design import CNNMNIST
from dataloaders import get_MNIST_dataloaders
from training_testing import train, test
from torch import nn
from torch.optim import SGD, Adam
import numpy as np

if __name__ == "__main__":
    # Hardware parameter
    device = "cpu"

    # NN design parameters
    conv_layers = [
        {
            "out_channels": 24,
            "kernel_size": 5,
            "stride": 2
        }
    ]
    
    fc_layers = [
        {
            "out_features": 256
        }
    ]

    # Finialize network construction
    model = CNNMNIST(conv_layers, fc_layers).to(device)
    loss_fn = nn.CrossEntropyLoss()

    learning_rate = 1e-3
    optimizer = Adam(
        params=model.parameters(), 
        lr=learning_rate
    )

    batch_size = 64
    train_dl, test_dl = get_MNIST_dataloaders(batch_size)

    max_no_epochs = 40
    convergence_treshold = 0.0001
    avgloss = []
    totacc = []

    window_1_size = 6
    window_2_size = 3
    warmup_periods = 6
    losschange = 1
    accchange = 1
    epoch = 1

    # TODO: try out a learning rate scheduler?

    while accchange > convergence_treshold and epoch <= max_no_epochs:
        train(
            train_dl,
            model,
            loss_fn,
            optimizer
        )

        closs, caccuracy = test(
            test_dl,
            model,
            loss_fn
        )
        print("Epoch", epoch)
        print("Test accuracy:", caccuracy)

        avgloss.append(closs)
        totacc.append(caccuracy)

        avglossr1 = np.mean(avgloss[-window_1_size:])
        avglossr2 = np.mean(avgloss[-window_2_size:])
        losschange = (avglossr1-avglossr2)/avglossr1

        totaccr1 = np.mean(totacc[-window_1_size:])
        totaccr2 = np.mean(totacc[-window_2_size:])
        accchange = (totaccr2-totaccr1)/totaccr1

        print("Change in loss function:", losschange)
        print("Change in accuracy:", accchange)

        if epoch <= max(1, warmup_periods):
            losschange = 1
            accchange = 1
            
        epoch += 1



