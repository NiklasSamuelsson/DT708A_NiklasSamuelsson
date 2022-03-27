from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def get_MNIST_dataloaders(batch_size):
    """
    Loads the MNIST dataset and contructs to dataloaders
    using PyTorch's built in modules.

    Parameters
    ----------
    batch_size : int
        The size of the batches.

    Returns
    -------
    PyTorch DataLoaders
        One dataloader for training and one for testing
    """
    training_data = datasets.MNIST(
                        root="data",
                        train=True,
                        download=True,
                        transform=ToTensor()
    )

    testing_data = datasets.MNIST(
                        root="data",
                        train=False,
                        download=True,
                        transform=ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size)

    return train_dataloader, test_dataloader

