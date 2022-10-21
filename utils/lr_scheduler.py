"""
Noam Scheduler.
From: https://raw.githubusercontent.com/tugstugi/pytorch-saltnet/master/utils/lr_scheduler.py
"""

from torch.optim.lr_scheduler import _LRScheduler

# https://github.com/mozilla/TTS/blob/ff295c65242328a6bc23a9fd9b4e6d819342795a/utils/training.py
class NoamLR(_LRScheduler):
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    """

    def __init__(self, optimizer, warmup_steps=0.1, last_epoch=-1):
        self.warmup_steps = float(warmup_steps)
        super(NoamLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 1)
        return [
            base_lr * self.warmup_steps**0.5 *
            min(step * self.warmup_steps**-1.5, step**-0.5)
            for base_lr in self.base_lrs
        ]
