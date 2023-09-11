from typing import Optional, List
import torch
import torch.nn as nn

class ImportanceWeightModule(object):
    def __init__(self, discriminator: nn.Module, partial_classes_index: Optional[List[int]] = None):
        self.discriminator = discriminator
        self.partial_classes_index = partial_classes_index

    def get_importance_weight(self, feature: torch.Tensor, label=1):
        """
        Get importance weights for each instance.
        Args:
            feature (tensor): feature, in shape :math:`(N, F)`
            label: ground truth, integer
        Returns:
            instance weight in shape :math:`(N, 1)`
        """
        weight = torch.abs(label - self.discriminator(feature))
        if weight.mean(): weight = weight / weight.mean()
        weight = weight.detach()
        return weight

    def get_partial_classes_weight(self, weights: torch.Tensor, labels: torch.Tensor):
        assert self.partial_classes_index is not None

        weights = weights.squeeze()
        is_partial = torch.Tensor([label in self.partial_classes_index for label in labels]).to(weights.device)
        if is_partial.sum() > 0:
            partial_classes_weight = (weights * is_partial).sum() / is_partial.sum()
        else:
            partial_classes_weight = torch.tensor(0)

        not_partial = 1. - is_partial
        if not_partial.sum() > 0:
            not_partial_classes_weight = (weights * not_partial).sum() / not_partial.sum()
        else:
            not_partial_classes_weight = torch.tensor(0)
        return partial_classes_weight, not_partial_classes_weight