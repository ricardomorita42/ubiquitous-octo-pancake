from torch.utils.tensorboard import SummaryWriter

class TensorboardWriter(SummaryWriter):
    def __init__(self, logdir):
        super(TensorboardWriter, self).__init__(logdir)

    def log_train_loss(self, train_loss, epoch):
        self.add_scalar("Loss/train", train_loss, epoch)

    def log_val_loss_std(self, val_loss, val_std, epoch):
        self.add_scalar("Loss/val", val_loss, epoch)
        #self.add_scalar("Acc/val", val_acc, epoch)
        self.add_scalar("StdDev/val", val_std, epoch)
