import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')

from utils.audio_processor import AudioProcessor
from utils.dataset import train_dataloader, val_dataloader, test_dataloader
from utils.generic import load_config
from utils.tensorboard import TensorboardWriter
from utils.get_params import get_loss, get_model, get_optimizer, get_scheduler

# Não usados
from unittest.util import _MAX_LENGTH
from random import choice

def train(dataloader, model, loss_fn, optimizer, device):
    train_loss = 0

    model.train()
    for features, targets in dataloader:
        features, targets = features.to(device), targets.to(device)
        pred = model(features)
        # print('Shape of features:', features.shape)
        # print('Shape of targets:', targets.shape, 'type:', targets.dtype)
        # print('Shape of pred:', pred.shape, 'type:', pred.dtype)

        loss = loss_fn(pred, targets)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss/len(dataloader)

def test(dataloader, model, loss_fn, device, acceptable_interval):
    test_loss, test_acc = 0, 0
    errors = []

    model.eval()
    with torch.no_grad():
        round = 0
        for features, targets in dataloader:
            round += 1
            features, targets = features.to(device), targets.to(device)
            pred = model(features)
            #print("pred = ", pred.view(pred.size(0)))
            #print("targets = ", targets.view(targets.size(0)))
            test_loss += loss_fn(pred, targets).item()
            correct_items = 0.0

            for x, y in zip(targets, pred):
                error = y.item() - x.item()
                errors.append(error)
                #print("({}, {})".format(x.item(),y.item()),end=" ")
                #print("c",(y + acceptable_interval <= x <= y - acceptable_interval))
                if abs(error) <= acceptable_interval:
                    #print("({}, {})".format(x.item(),y.item()),end=" ")
                    correct_items += 1

            test_acc += correct_items / len(pred)
            #print(f"\nround:{round}")
            #print("total items:", len(pred))
            #print("number of hits:", correct_items)
            #print("hits/total items:", correct_items/len(pred))

    test_loss /= round
    test_acc = 100*test_acc/round
    test_std = np.std(errors)

    print(f"Error: \n Accuracy: {test_acc:>0.1f}%, Avg loss: {test_loss:>8f}, Std dev: {test_std:>8f} \n")

    return test_loss, test_acc, test_std

def save_checkpoint(path, model, optimizer, epoch, val_loss):
    try:
        torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss
        }, path)
    except:
        print("Ocorreu um erro enquanto salvava o checkpoint", path)

def load_checkpoint(path, model, optimizer, device):
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location=device)
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except:
            print("Falhou ao inicializar modelo, talvez seja outro.")
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except:
            print("Falhou ao inicializar otimizador, talvez seja outro.")
        epoch = checkpoint["epoch"]
    else:
        print("Não localizou o arquivo ", path)
        epoch = 0

    return epoch

if __name__ == '__main__':
    '''
    Exemplo de uso: python train.py -c experiments/configs/exp-1.1.json \
                    --checkpoint_path checkpoints/exp-1.1/checkpoint_25.pt
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

    # Inicializa o writer para escrever logs compatíveis com o Tensorboard
    writer = TensorboardWriter(os.path.join(logs_dir, 'tensorboard'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ",device)
    ap = AudioProcessor(**c.audio)

    # Inicialização variáveis extras
    acceptable_interval = float(c.train_config["acceptable_interval"])

    # INICIALIZA MODELO E DATASETS =============================================

    trainloader = train_dataloader(c, ap)
    valloader = val_dataloader(c, ap)
    testloader = test_dataloader(c, ap)

    model = get_model(c.model_name, c.model_config, c.audio)
    loss_fn = get_loss(c.train_config["loss_fn"])
    optimizer = get_optimizer(c.train_config, model)

    if torch.cuda.device_count() > 1:
        print("Usando", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    # INICIALIZA CHECKPOINT ====================================================

    if (checkpoint_path is not None):
        print("Lendo checkpoint", checkpoint_path)
        epoch = load_checkpoint(os.path.abspath(checkpoint_path), model, optimizer, device)
    else:
        epoch = 0

    # LEARNING RATE SCHEDULER ==================================================

    scheduler = get_scheduler(c.train_config, optimizer, epoch)

    # TREINO / EPOCHS ==========================================================

    while epoch < c.train_config['epochs']:
        print('========================================================================')
        print("Epoch %d" % epoch)
        print('========================================================================')

        train_loss = train(trainloader, model, loss_fn, optimizer, device)
        val_loss, val_acc, val_std = test(valloader, model, loss_fn, device, acceptable_interval)

        if scheduler:
            scheduler.step()

        epoch += 1

        if epoch%c.train_config["summary_interval"] == 0:
            writer.log_train_loss(train_loss, epoch)
            writer.log_val_loss_acc_std(val_loss, val_acc, val_std, epoch)
            print("Write summary at epoch", epoch)
            print(f'Avg. Train Loss: {train_loss:>8f}')
            print(f'Avg. Val Loss: {val_loss:>8f} / Val Acc: {val_acc:>0.1f}% / Val std. dev: {val_std:>0.8}\n')

        if epoch%c.train_config["checkpoint_interval"] == 0:
            save_path = os.path.join(logs_dir, "checkpoint_%d.pt" % epoch)
            save_checkpoint(save_path, model, optimizer, epoch, val_loss)
            print("Salvou checkpoint em", save_path)

        best_checkpoint_path = os.path.join(logs_dir, "best_checkpoint.pt")

        try:
            checkpoint = torch.load(best_checkpoint_path, map_location=device)
            best_saved_loss = checkpoint["val_loss"]
        except:
            print("Não existe melhor checkpoint ou falhou na leitura")
            best_saved_loss = 9999999

        if val_loss < best_saved_loss:
            # Encontrei um modelo melhor que o salvo
            print("Saved loss: ", best_saved_loss, ", Actual loss:", val_loss)
            save_checkpoint(best_checkpoint_path, model, optimizer, epoch, val_loss)
            print("Salvou melhor checkpoint em", best_checkpoint_path)

    print("Done!\n")

    # Assegura de que todos tensorboard logs foram escritos e encerra ele
    writer.flush()
    writer.close()

    # save_path = os.path.join(logs_dir, "checkpoint_%d.pt" % epoch)
    # save_checkpoint(save_path, model, optimizer, epoch, val_loss)
    # print("Salvou checkpoint da última época em", save_path)

    print('========================================================================')
    print("TEST SUMMARY OF THE BEST CHECKPOINT")
    print('========================================================================')

    best_checkpoint_path = os.path.join(logs_dir, "best_checkpoint.pt")
    load_checkpoint(best_checkpoint_path, model, optimizer, device)

    test_loss, test_acc, test_std = test(testloader, model, loss_fn, device,
                                         acceptable_interval)

    # d = Dataset(ap, c.dataset["train_csv"])

    # # Deixando esta parte executando para que se possa checar o funcionamento...
    # print("Imprimindo um exemplo")
    # # {audio_path: [[feature_window_1, feature_window_2, ...], sexo, idade, spO2]}
    # audio_path, [feature, sexo, idade, spO2] = choice(list(d.getWholeDataset().items()))
    # print("Exemplo:", audio_path)
    # ap.graphFeature(feature[0])
