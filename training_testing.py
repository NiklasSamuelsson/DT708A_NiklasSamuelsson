

def train(dataloader, model, loss_fn, optimizer):
    """
    Performs one epoch of training.

    Parameters
    ----------
    dataloader : PyTorch DataLoader
        Needs to only generate training data.
    model : model object based on torch.nn.Module
        A properly implemented PyTroch model to train.
    loss_fn : PyTorch loss function
        Which loss function to optimize.
    optimizer : PyTorch optimizer
        Which optimizer to use in training.
    """
    for x, y in dataloader:
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(dataloader, model, loss_fn):
    """
    _summary_

    Parameters
    ----------
    dataloader : PyTorch DataLoader
        Needs to only generate training data.
    model : model object based on torch.nn.Module
        A properly implemented PyTroch model to test.
    loss_fn : PyTorch loss function
        Which loss function to use in order to measure progress.

    Returns
    -------
    _type_
        _description_
    """
    no_batches = len(dataloader)
    loss = 0

    size = len(dataloader.dataset)
    accuracy = 0

    for x, y in dataloader:
        pred = model(x)
        loss += loss_fn(pred, y).item()
        accuracy += (pred.argmax(1) == y).sum().item()

    loss /= no_batches
    accuracy /= size


    return loss, accuracy
