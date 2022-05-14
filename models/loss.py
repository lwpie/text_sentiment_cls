import torch.nn as nn


class LOSS(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        if loss_type == 'bce':
            self.loss_crt = nn.BCELoss()
        else:
            raise NotImplementedError

    def forward(self, pred, label):
        # todo label不能是1，2
        return self.loss_crt(pred, label)
