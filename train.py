import argparse
import os
import torch
import torch.nn as nn

from utils.audio_processor import AudioProcessor
from utils.dataset import train_dataloader, test_dataloader
from utils.generic import load_config
from models.cnn import SpiraConvV2

# Não usados
from unittest.util import _MAX_LENGTH
from random import choice

def train(dataloader, model, loss_fn, optimizer, device):
    model.train()

    for batch_id, (features, targets) in enumerate(dataloader):
        features, targets = features.to(device), targets.to(device)
        pred = model(features)
        # print('Shape of features:', features.shape)
        # print('Shape of targets:', targets.shape, 'type:', targets.dtype)
        # print('Shape of pred:', pred.shape, 'type:', pred.dtype)

        loss = loss_fn(pred, targets)

        # Bakpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch_id % 5 == 0:
        #     loss, current = loss.item(), batch_id * len(features)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        #     # print("targets:", targets, "/ pred", pred)

def test(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for features, targets in dataloader:
            features, targets = features.to(device), targets.to(device)
            pred = torch.round(model(features))
            test_loss += loss_fn(pred, targets).item()
            correct += (pred == targets).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= len(dataloader.dataset)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def get_loss(loss_name):
    if loss_name == "MSE":
        return nn.MSELoss()
    else:
        raise Exception("A loss '" + loss_name + "' não é suportada")

def get_model(model_name, model_config):
    if model_name == "SpiraConvV2":
        model = SpiraConvV2(model_config)
        return model
    else:
        raise Exception("O modelo '" + model_name + "' não é suportado")

def get_optimizer(train_config, model):
    if train_config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                 lr=train_config["lr"],
                                 weight_decay=train_config["weight_decay"])
        return optimizer
    else:
        raise Exception("O otimizador '" + train_config["optimizer"] + "' não é suportado")

if __name__ == '__main__':
    '''
    Exemplo de uso: python train.py -c experiments/configs/exp-1.1.json
    '''

    # Converte e carrega arquivo json com dados do experimento
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str,
                        default="experiments/configs/exp-1.1.json",
                        help="json file with configurations")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file, for continue training")

    args = parser.parse_args()
    c = load_config(args.config_path)
    checkpoint_path = args.checkpoint_path
    logs_dir = c.train_config["logs_dir"]

    # Cria diretório dos logs se ele não existir
    os.makedirs(logs_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ap = AudioProcessor(**c.audio)

    # INICIALIZA MODELO E DATASETS =============================================

    trainloader = train_dataloader(c, ap)
    testloader = test_dataloader(c, ap)

    model = get_model(c.model_name, c.model_config)
    loss_fn = get_loss(c.train_config["loss_fn"])
    optimizer = get_optimizer(c.train_config, model)

    model.to(device)

    # INICIALIZA CHECKPOINT ====================================================

    if (checkpoint_path is not None):
        print("Lendo checkpoint", checkpoint_path)

        try:
            checkpoint = torch.load(checkpoint_path)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"]
            if device == "cuda":
                model.load_state_dict(checkpoint["model_state_dict"],
                                      map_location="cuda:0")
            else:
                model.load_state_dict(checkpoint["model_state_dict"])
        except:
            print("Aconteceu um erro na leitura do checkpoint", checkpoint_path)
            epoch = 0
    else:
        epoch = 0

    # TREINO / EPOCHS ==========================================================

    while epoch < c.train_config['epochs']:
        print('=================================================')
        print("Epoch %d" % epoch)
        print('=================================================')

        train(trainloader, model, loss_fn, optimizer, device)
        test(testloader, model, loss_fn, device)

        epoch += 1

        if epoch%c.train_config["checkpoint_interval"] == 0:
            save_path = os.path.join(logs_dir, "checkpoint_%d.pt" % epoch)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, save_path)
            print("Salvou checkpoint em", save_path)

    print("Done!")

    # d = Dataset(ap, c.dataset["train_csv"])

    # # Deixando esta parte executando para que se possa checar o funcionamento...
    # print("Imprimindo um exemplo")
    # # {audio_path: [[feature_window_1, feature_window_2, ...], sexo, idade, spO2]}
    # audio_path, [feature, sexo, idade, spO2] = choice(list(d.getWholeDataset().items()))
    # print("Exemplo:", audio_path)
    # ap.graphFeature(feature[0])

    # # Cria lista das features e dos targets
    # # features, targets = [], []

    # # for key, val in d.getWholeDataset().items():
    # #     for item in val[0]:
    # #         features.append(item)
    # #         targets.append(val[3])
