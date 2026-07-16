import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses, reducers


class UnpackReducer(reducers.BaseReducer):
    def element_reduction(self, losses, loss_indices, embeddings, labels):
        sorted_indices = torch.argsort(loss_indices)
        return losses[sorted_indices]


class NormalizeEmbeddingsWrapper(nn.Module):
    def __init__(self, loss: nn.Module):
        super().__init__()
        self.loss = loss

    def forward(self, embeddings, *args, **kwargs):
        return self.loss(F.normalize(embeddings, dim=1), *args, **kwargs)


class FeatLossFactory:
    @staticmethod
    def get(loss: str = None, **kwargs):
        if loss is None or loss == "None":
            return None
        elif loss == "sup_con_loss":
            # https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#supconloss
            params = {
                "temperature": 0.07,
            }
            params.update({k: v for k, v in kwargs.items() if k in inspect.signature(losses.SupConLoss).parameters})
            return NormalizeEmbeddingsWrapper(losses.SupConLoss(**params, reducer=UnpackReducer()))

        elif loss == "circle_loss":
            # https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#circleloss
            params = {
                "m": 0.40,
                "gamma": 32,
            }
            params.update({k: v for k, v in kwargs.items() if k in inspect.signature(losses.CircleLoss).parameters})
            return NormalizeEmbeddingsWrapper(losses.CircleLoss(**params, reducer=UnpackReducer()))

        elif loss == "multi_sim_loss":
            # https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#multisimilarityloss
            params = {
                "alpha": 2.0,
                "beta": 12.0,
                "base": 0.5,
            }
            params.update(
                {k: v for k, v in kwargs.items() if k in inspect.signature(losses.MultiSimilarityLoss).parameters}
            )
            return NormalizeEmbeddingsWrapper(losses.MultiSimilarityLoss(**params, reducer=UnpackReducer()))

        else:
            raise ValueError(f"Unknown feat loss type: '{loss}'")


class ClassConfWeight:
    def __init__(self, **kwargs):
        pass

    def __call__(self, target_scores, pred_scores, *args, **kwargs):
        return pred_scores.sigmoid().max(-1).values


class WeightFactory:
    @staticmethod
    def get(weight: str = None, **kwargs):
        if weight is None or weight == "None":
            return None
        elif weight == "class_conf":
            return ClassConfWeight()
        else:
            raise ValueError(f"Unknown weight type: '{weight}'")


class Masking:
    def __init__(self, mask_pct: float = 0.4, **kwargs):
        super().__init__(**kwargs)
        self.mask_pct = mask_pct

    def _masking(self, metric):
        k = max(1, int(len(metric) * (1 - self.mask_pct)))
        thresh = metric.topk(k).values[-1]
        return metric >= thresh


class ClassConfMask(Masking, ClassConfWeight):
    def __call__(self, target_scores, pred_scores, *args, **kwargs):
        class_conf = super().__call__(target_scores, pred_scores, *args, **kwargs)
        return self._masking(class_conf)


class RandMask:
    def __init__(self, mask_pct: float = 0.4, **kwargs):
        self.mask_pct = mask_pct

    def __call__(self, target_scores, pred_scores=None):
        k = max(1, int(target_scores.shape[0] * self.mask_pct))
        mask = torch.zeros(target_scores.shape[0], dtype=torch.bool)
        indices = torch.randperm(target_scores.shape[0])[:k]
        mask[indices] = True
        return ~mask


class RandMaskBalanced:
    def __init__(self, mask_pct: float = 0.4, min_per_class: int = 4, **kwargs):
        self.mask_pct = mask_pct
        self.min_per_class = min_per_class

    def __call__(self, target_scores, pred_scores=None):
        target_cls = target_scores.max(-1).indices
        n = target_scores.shape[0]
        k = max(1, int(n * self.mask_pct))
        mask = torch.zeros(n, dtype=torch.bool)

        # Guarantee at least min_per_class per unique class
        for cls in target_cls.unique():
            cls_indices = (target_cls == cls).nonzero(as_tuple=True)[0]
            k_cls = min(self.min_per_class, len(cls_indices))
            chosen = cls_indices[torch.randperm(len(cls_indices))[:k_cls]]
            mask[chosen] = True

        # Fill remaining budget with random unchosen indices
        remaining = k - mask.sum().item()
        if remaining > 0:
            unmasked = (~mask).nonzero(as_tuple=True)[0]
            extra = unmasked[torch.randperm(len(unmasked))[:remaining]]
            mask[extra] = True

        return ~mask


class MaskFactory:
    @staticmethod
    def get(mask: str = None, **kwargs):
        if mask is None or mask == "None":
            return None
        elif mask == "class_conf":
            return ClassConfMask(**kwargs)
        elif mask == "rand":
            return RandMask(**kwargs)
        elif mask == "rand_balanced":
            return RandMaskBalanced(**kwargs)
        else:
            raise ValueError(f"Unknown mask type: '{mask}'")


class ClsFeatLoss(nn.Module):
    def __init__(self, loss: str, mask: str = None, weight: str = None, **kwargs):
        super().__init__()
        assert mask is None or weight is None, "Only one of mask or weight can be specified, not both"
        self.loss = FeatLossFactory.get(loss, **kwargs)
        self.mask = MaskFactory.get(mask, **kwargs)
        self.weight = WeightFactory.get(weight, **kwargs)

    def forward(
            self,
            cls_feats: torch.Tensor,
            target_scores: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=cls_feats.device)
        target_cls = target_scores.max(-1).indices
        if self.mask is not None:
            mask = self.mask(cls_feats=cls_feats, target_scores=target_scores, *args, **kwargs)
            if not mask.sum():
                return loss
            cls_feats = cls_feats[mask]
            target_cls = target_cls[mask]

        loss_per_element = self.loss(cls_feats, target_cls).squeeze(-1)

        if self.weight is not None:
            weight = self.weight(target_scores=target_scores, *args, **kwargs)
            weight = weight / weight.sum()
            loss += (loss_per_element * weight).sum()
        else:
            loss += loss_per_element.mean()

        return loss
