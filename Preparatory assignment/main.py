import matplotlib
from nn_design import CNNMNIST
from torch import nn
from torch.optim import SGD, Adam
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from help_functions import plot_results, find_errornous_predictions

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
            "out_features": 200
        }
    ]

    # TODO: try out a learning rate scheduler?
    learning_rate = 1e-3
    batch_size = 64

    model = CNNMNIST(
        conv_layers, 
        fc_layers, 
        Adam, 
        learning_rate, 
        nn.CrossEntropyLoss,
        batch_size).to(device)

    convergence_treshold = 0.0001
    window_1_size = 6
    window_2_size = 3
    max_no_epochs = 70
    warmup_periods = 6

    model.fit(
        convergence_treshold,
        window_1_size,
        window_2_size,
        max_no_epochs,
        warmup_periods    
    )

    # Plot confusion matrix
    #%matplotlib widget
    #cm = model.confusion_matrix()
    #ax = plt.axes()
    #s = sns.heatmap(cm, annot=True, fmt=".0f")
    #s.set_xlabel("Predicted")
    #s.set_ylabel("Label")
    #ax.set_title("Adam")

    # Plot errornous predictions
    #label = 7
    #pred = 2
    #err_preds = find_errornous_predictions(model, label, pred)
    #%matplotlib widget
    #ax = plt.axes()
    #s = sns.heatmap(err_preds[0], cmap="magma")
    #ax.set_title(str(label) + " predicted as " + str(pred))

    





