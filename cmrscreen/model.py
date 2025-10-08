from torch import Tensor
from chemprop import data, featurizers, models, nn


class FFN(nn.MulticlassClassificationFFN):
    def forward(self, Z: Tensor) -> Tensor:
        return self.train_step(Z)


def MPNN(n_classes=2):
    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    ffn = FFN(n_classes=n_classes)
    batch_norm = False
    metric_list = None
    model = models.MPNN(mp, agg, ffn, batch_norm, metric_list)
    return model