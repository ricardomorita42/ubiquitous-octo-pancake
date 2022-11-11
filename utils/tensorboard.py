from torch.utils.tensorboard import SummaryWriter

class TensorboardWriter(SummaryWriter):
    def __init__(self, logdir):
        super(TensorboardWriter, self).__init__(logdir)

    def log_train_loss(self, train_loss, epoch):
        self.add_scalar("Train Loss", train_loss, epoch)

    def log_val_loss(self, val_loss, epoch):
        self.add_scalar("Valid. Loss", val_loss, epoch)

    def log_test_loss_std(self, test_loss, test_std, epoch):
        self.add_scalar("Test Loss", test_loss, epoch)
        self.add_scalar("Test StdDev", test_std, epoch)
