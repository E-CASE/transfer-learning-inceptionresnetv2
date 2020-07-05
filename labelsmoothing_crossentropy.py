import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothLoss(nn.Module):

    def __init__(self, smoothing=0.0, num_classes=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        print(f'smoothing = {smoothing:.4f}')

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.smoothing / self.num_classes
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing) + self.smoothing / self.num_classes)
        # print(weight)
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss
