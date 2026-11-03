from yolox.utils import WandbLogger as _WandbLogger


class WandbLogger(_WandbLogger):
    @classmethod
    def initialize_wandb_logger(cls, args, exp, val_dataset):
        wandb_params = dict()
        prefix = "wandb-"
        for k, v in zip(args.opts[0::2], args.opts[1::2]):
            if k.startswith("wandb-"):
                try:
                    wandb_params.update({k[len(prefix):]: v if k == "wandb-name" else int(v)})
                except ValueError:
                    wandb_params.update({k[len(prefix):]: v})

        return cls(config=vars(exp), val_dataset=val_dataset, **wandb_params)
