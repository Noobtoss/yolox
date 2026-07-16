import math


class ClsFeatScheduler:
    def __init__(self, name, cls_feat, total_epochs, **kwargs):
        self.name = name
        self.cls_feat = cls_feat
        self.total_epochs = total_epochs
        self.kwargs = kwargs

    def update_cls_feat(self, epoch):
        if self.name == "constant":
            return self.cls_feat
        elif self.name == "inverse_cos_decay":
            return self.cls_feat - cos_lr(self.cls_feat, self.total_epochs, epoch)
        elif self.name == "inverse_cos_wsd":
            return self.cls_feat - cos_lr_wsd(self.cls_feat, self.total_epochs, epoch, **self.kwargs)


def cos_lr(lr, total_iters, iters):
    """Cosine learning rate"""
    lr *= 0.5 * (1.0 + math.cos(math.pi * iters / total_iters))
    return lr


def cos_lr_wsd(lr, total_iters, iters, warmup_iters=0.0, stable_iters=0.2, zero_iters=0.2, **kwargs):
    """
    Four-phase learning rate schedule:
      1. Warmup  – linear ramp 0 → lr
      2. Stable  – flat at lr
      3. Decay   – cosine decay lr → 0
      4. Zero    – hard 0
    warmup_iters, stable_iters, zero_iters are fractions of total_iters (0.0 - 1.0)
    """
    warmup = round(warmup_iters * total_iters)
    stable = round(stable_iters * total_iters)
    zero   = round(zero_iters   * total_iters)
    decay  = total_iters - warmup - stable - zero

    if iters < warmup:
        return lr * (iters / max(1, warmup))
    elif iters < warmup + stable:
        return lr
    elif iters < warmup + stable + decay:
        progress = (iters - warmup - stable) / max(1, decay)
        return lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        return 0.0
