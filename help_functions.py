import matplotlib.pyplot as plt
import numpy as np

def plot_results(results, title):
    """
    Plots results.

    Parameters
    ----------
    results : dict
        Dictionary with the name of the series to plot (str) as key
        and a list type object of with the value to plot.
    title : str
        Title of the plot.
    """
    plt.title(title)
    
    for name, values in results.items():
        plt.plot(values, label=name)
    
    plt.legend()
    plt.show()

def find_errornous_predictions(model, label, prediction):
    """
    Finds samples where the provided model predicted <prediction>
    where the labels were <label>.

    Parameters
    ----------
    model : PyTorch model of CNNMNIST type
        The model whose results to analyse. Must have a test dataloader
        as an attribute.
    label : int
        Which label to save the samples for.
    prediction : int
        Which prediction to save the samples for.

    Returns
    -------
    list of numpy arrays
        Samples in a list.
    """
    samples = []

    for x, y in model.test_dl:
        pred = model(x)

        i = 0
        for l_, p_ in zip(y, pred):
            l = l_.item()
            p = p_.argmax().item()

            if l==label and p==prediction:
                samples.append(x[i].numpy()[0])
            i += 1

    return samples
