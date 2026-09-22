import torch
import torch.nn as nn


class ClassLossWeighted(nn.Module):
    def __init__(self,
                 loss: nn.Module = nn.BCEWithLogitsLoss(reduction="none"),
                 class_weights: torch.Tensor = None,
                 class_weights_matrix: torch.Tensor = None
                 ) -> None:
        super().__init__()
        self.loss = loss

        self.register_buffer("class_weights", class_weights)
        self.register_buffer("class_weights_matrix", class_weights_matrix)

    def forward(self, pred_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        loss = self.loss(pred_scores, target_scores)
        if self.class_weights is not None:
            loss = loss * self.class_weights
        if self.class_weights_matrix is not None:
            gt_cls = target_scores.argmax(dim=-1)
            weight_per_sample = self.class_weights_matrix[gt_cls]
            loss = loss * weight_per_sample
        return loss
