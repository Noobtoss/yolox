import torch.nn as nn


class ClsFeatProjHeadFactory:
    @staticmethod
    def get(proj_head: str = None, dim: int = 256, **kwargs):
        if proj_head is None or proj_head == "None":
            return None

        if proj_head == "n":
            return nn.Sequential(
                nn.Linear(dim, 64)
            )

        if proj_head == "s":
            return nn.Sequential(
                nn.Linear(dim, 128)
            )

        if proj_head == "m":
            return nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, 128)
            )

        if proj_head == "l":
            return nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, 128)
            )

        raise ValueError(f"Unknown proj head type: '{proj_head}'. Choose from: 'n', 's', 'm', 'l'")
