import torch.nn as nn


class ClsFeatProjHeadFactory:
    @staticmethod
    def get(proj_head: str = None, dim: int = 256, **kwargs) -> list[nn.Module]:
        if proj_head == "n":
            return [
                nn.Linear(dim, 64)
            ]

        if proj_head == "s":
            return [
                nn.Linear(dim, 128)
            ]

        if proj_head == "m":
            return [
                nn.Linear(dim, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, 128)
            ]

        if proj_head == "l":
            return [
                nn.Linear(dim, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, 128)
            ]

        raise ValueError(f"Unknown proj head type: '{proj_head}'. Choose from: 'n', 's', 'm', 'l'")

class ClsFeatProjHead(nn.Sequential):
    def __init__(self, proj_head: str, dim: int = 256, **kwargs):
        super().__init__(*ClsFeatProjHeadFactory.get(proj_head=proj_head, dim=dim, **kwargs))
