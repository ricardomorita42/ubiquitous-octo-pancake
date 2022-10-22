from torch.utils.tensorboard import SummaryWriter

class TensorboardWriter(SummaryWriter):
    def __init__(self, logdir):
        super(TensorboardWriter, self).__init__(logdir)

    def log_train_loss(self, train_loss, epoch):
        self.add_scalar("Loss/train", train_loss, epoch)

    def log_test_loss_acc_std(self, test_loss, test_acc, test_std, epoch):
        self.add_scalar("Loss/test", test_loss, epoch)
        self.add_scalar("Acc/test", test_acc, epoch)
        self.add_scalar("StdDev/test", test_std, epoch)
