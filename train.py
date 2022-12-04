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

import math
import test as t

# Não usados
from unittest.util import _MAX_LENGTH
from random import choice

def train(dataloader, model, loss_fn, optimizer, device, rmse_flag):
    train_loss = 0

    model.train()
    for features, targets in dataloader:
        features, targets = features.to(device), targets.to(device)
        pred = model(features)
        # print('Shape of features:', features.shape)
        # print('Shape of targets:', targets.shape, 'type:', targets.dtype)
        # print('Shape of pred:', pred.shape, 'type:', pred.dtype)

        loss = loss_fn(pred, targets)
        
        if rmse_flag == True:
            loss = torch.sqrt(loss)

        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss / len(dataloader)

def validate(dataloader, model, loss_fn, device, rmse_flag):
    errors = []
    loss_list = []

    model.eval()
    with torch.no_grad():
        for features, targets in dataloader:
            features, targets = features.to(device), targets.to(device)
            pred = model(features)
            #print(f"round:{round}")
            #print("pred = ", pred.view(pred.size(0)))
            #print("targets = ", targets.view(targets.size(0)))

            loss = loss_fn(pred, targets).item() 
            if rmse_flag == True:
                loss = math.sqrt(loss)
            loss_list.append(loss)

            #print("total items:", len(pred))

    #print("loss list = ", loss_list)
    val_loss = np.mean(loss_list)

    #print("\nloss function = ", type(loss_fn).__name__)
    #print("tam dataloader = ", len(dataloader))
    #print(f"rounds:{round}")
    print(f"Error: \n Avg loss: {val_loss:>8f}\n")

    return val_loss

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

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print("device: ",device)
    ap = AudioProcessor(**c.audio)

    # Inicialização variáveis extras
    #acceptable_interval = float(c.train_config["acceptable_interval"])

    # INICIALIZA MODELO E DATASETS =============================================

    trainloader = train_dataloader(c, ap)
    valloader = val_dataloader(c, ap)
    testloader = test_dataloader(c, ap)

    model = get_model(c.model_name, c.model_config, c.audio)
    loss_fn = get_loss(c.train_config["loss_fn"])
    optimizer = get_optimizer(c.train_config, model)

    #if torch.cuda.device_count() > 1:
    #    print("Usando", torch.cuda.device_count(), "GPUs!")
    #    model = nn.DataParallel(model)

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
    # Tentei adicionar RMSE no util/get_params.py mas não estava indo, então coloquei aqui
    if c.train_config["loss_fn"] == "RMSE":
        rmse_flag = True
    else:
        rmse_flag = False

    while epoch < c.train_config['epochs']:
        print('========================================================================')
        print("Epoch %d" % epoch)
        print('========================================================================')

        train_loss = train(trainloader, model, loss_fn, optimizer, device, rmse_flag)
        val_loss = validate(valloader, model, loss_fn, device, rmse_flag)

        if scheduler:
            scheduler.step()

        epoch += 1

        if epoch%c.train_config["summary_interval"] == 0:
            writer.log_train_loss(train_loss, epoch)
            writer.log_val_loss(val_loss, epoch)
            print("Write summary at epoch", epoch)
            print(f'Avg. Train Loss: {train_loss:>8f}')
            print(f'Avg. Val Loss: {val_loss:>8f}\n')

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

    # save_path = os.path.join(logs_dir, "checkpoint_%d.pt" % epoch)
    # save_checkpoint(save_path, model, optimizer, epoch, val_loss)
    # print("Salvou checkpoint da última época em", save_path)

    print('========================================================================')
    print("TEST SUMMARY OF THE BEST CHECKPOINT")
    print('========================================================================')

    best_checkpoint_path = os.path.join(logs_dir, "best_checkpoint.pt")
    load_checkpoint(best_checkpoint_path, model, optimizer, device)

    test_avg_diff, test_std, test_conf, relative_diff_list = t.test(testloader, model, loss_fn, device)
    #print(f'Avg. Test Loss: {test_loss:>8f} / Test std: {test_std:>8f}\n')

    fig, ax = plt.subplots(tight_layout=True)
    relative_error = pd.DataFrame(relative_diff_list, columns=['erro_relativo'])
    relative_error.plot(kind='hist', density=True, bins=50, ax=ax)
    relative_error.plot(style={'erro_relativo': 'k-'}, kind='kde', ax=ax)
    ax.set_title(args.config_path.replace("experiments/configs/exp-","Exp ").replace(".json",""))
    ax.set_ylabel('Amostras')
    ax.set_xlabel('Erro relativo')
    ax.set_xlim([-0.01, 0.15])
    ax.get_legend().remove()
    img_path = 'images/' + args.config_path.replace("experiments/configs/exp-","saida-").replace(".json",".png")
    plt.savefig(img_path)

    test_conf_str = ' '.join(test_conf)
    writer.add_text("avg difference",str(test_avg_diff),0)
    writer.add_text("avg std",str(test_std),1)
    writer.add_text("confusion matrix (TP,TN,FP,FN)",str(test_conf_str),2)

    print(f'Avg. Test avg diff: {test_avg_diff:>8f}')
    print(f'Avg. Test std dev: {test_std:>8f}\n')
    print(f"(TP,TN,FP,FN) em %:{test_conf[0]:0.2f},{test_conf[1]:0.2f},{test_conf[2]:0.2f},{test_conf[3]:0.2f}")
    print("conf list:", test_conf_str)

    writer.flush()
    writer.close()

    # d = Dataset(ap, c.dataset["train_csv"])

    # # Deixando esta parte executando para que se possa checar o funcionamento...
    # print("Imprimindo um exemplo")
    # # {audio_path: [[feature_window_1, feature_window_2, ...], sexo, idade, spO2]}
    # audio_path, [feature, sexo, idade, spO2] = choice(list(d.getWholeDataset().items()))
    # print("Exemplo:", audio_path)
    # ap.graphFeature(feature[0])
