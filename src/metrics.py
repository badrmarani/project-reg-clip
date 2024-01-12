import torch
from torchmetrics import Metric


class WorstGroupAccuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", torch.zeros(4), dist_reduce_fx="sum")
        self.add_state("total", torch.zeros(4), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, y_pred, y_true, g):
        self.total += torch.arange(4, device=self.device).unsqueeze(1).eq(g).sum(dim=1)

        is_correct = y_true == y_pred
        for i in range(4):
            indices = torch.nonzero(g == i).squeeze(1)
            self.correct[i] += is_correct[indices].sum()

    @torch.no_grad()
    def compute(self):
        x = self.correct.float() / self.total
        x[x.isnan()] = 0.0
        wg_acc = x.min()
        return wg_acc.cpu().item(), x
