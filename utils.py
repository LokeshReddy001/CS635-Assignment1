# utils.py

import torch
from torch import nn

class ContrastiveLoss(nn.Module):
    """
        Given a list of scores s1,s2,..sn, calculates -log(e^s1/(e^s1+e^s2+...+e^sn))
    """
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, scores):
        scaled_scores = scores / 1.0
        max_score = torch.max(scaled_scores)
        stable_scaled_scores = scaled_scores - max_score
        log_sum_exp = max_score + torch.log(torch.sum(torch.exp(stable_scaled_scores)))
        loss = log_sum_exp - scaled_scores[0]

        return loss
