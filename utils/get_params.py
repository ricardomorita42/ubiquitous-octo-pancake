from torch.nn import MSELoss, L1Loss
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import Adam, AdamW, RMSprop

from utils.lr_scheduler import NoamLR
from models.cnn import SpiraConvV2, CNN2
from math import sqrt

def get_loss(loss_name):
    if loss_name == "MSE" or loss_name == "RMSE":
        return MSELoss()
    elif loss_name == "MAE":
        return L1Loss()
    else:
        raise Exception("A loss '" + loss_name + "' não é suportada")


def get_model(model_name, model_config, audio_config):
    if model_name == "SpiraConvV2":
        return SpiraConvV2(model_config, audio_config)
    if model_name == "CNN2":
        return CNN2(model_config, audio_config)
    else:
        raise Exception("O modelo '" + model_name + "' não é suportado")


def get_optimizer(train_config, model):
    optimizers = {
        'Adam': Adam,
        'AdamW': AdamW,
        'RMS': RMSprop
    }
    optimizer = optimizers.get(train_config["optimizer"])

    if optimizer:
        return optimizer(model.parameters(),
                         lr=train_config["lr"],
                         weight_decay=train_config["weight_decay"])
    else:
        raise Exception("O otimizador '" + train_config["optimizer"] +
                        "' não é suportado")


def get_scheduler(train_config, optimizer, last_epoch):
    if train_config["scheduler"] == "Noam":
        scheduler = NoamLR(optimizer,
                           warmup_steps=train_config['warmup_steps'],
                           last_epoch=last_epoch - 1)
        return scheduler
    elif train_config["scheduler"] == "Exponential":
        scheduler = ExponentialLR(optimizer,
                                  gamma=0.96,
                                  last_epoch=last_epoch - 1)
        return scheduler
    else:
        return None
