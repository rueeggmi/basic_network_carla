import torch.nn.functional as F


def nll_loss(output, target):
    # return F.nll_loss(output, target)
    return F.mse_loss(output, target)


def nll_loss_batch(outputs, targets, loss_weights):
    return_loss = 0
    for num, output in enumerate(outputs):
        return_loss += loss_weights[num] * F.nll_loss(output, targets[num])
    return return_loss
