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

    def forward(self, pred_scores: torch.Tensor, target_scores: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        loss = self.loss(pred_scores, target_scores, *args, **kwargs)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device=pred_scores.device)
            loss = loss * self.class_weights
        if self.class_weights_matrix is not None:
            self.class_weights_matrix = self.class_weights_matrix.to(device=pred_scores.device)
            labels = target_scores.argmax(dim=-1)
            weight_per_sample = self.class_weights_matrix[labels]
            weight_per_sample = torch.where(
                target_scores.any(dim=-1, keepdim=True),
                weight_per_sample,
                torch.ones_like(weight_per_sample),
            )
            loss = loss * weight_per_sample
        return loss
