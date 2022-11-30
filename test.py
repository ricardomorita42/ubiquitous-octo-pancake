import argparse
import torch
import torch.nn as nn
import numpy as np
import os

from utils.audio_processor import AudioProcessor
from utils.dataset import test_dataloader
from utils.generic import load_config
from utils.tensorboard import TensorboardWriter
from utils.get_params import get_loss, get_model, get_optimizer, get_scheduler

def test(dataloader, model, loss_fn, device):
    errors = []
    loss_list = []

    model.eval()
    with torch.no_grad():
        #round = 0
        diff_list = []
        for features, targets in dataloader:
            features, targets = features.to(device), targets.to(device)
            pred = model(features)
            #print(f"round:{round}")
            #print("pred = ", pred.view(pred.size(0)))
            #print("targets = ", targets.view(targets.size(0)))
            
            for x,y in zip(targets,pred):
                difference = np.abs(x.item() - y.item())
                diff_list.append(difference)

            #loss = loss_fn(pred, targets).item() 
            #loss_list.append(loss)

            #print("total items:", len(pred))
            #round += 1

    #print("loss list = ", loss_list)
    #test_loss = np.mean(loss_list)
    #test_std = np.std(loss_list)

    diff_mean = np.mean(diff_list)
    diff_std = np.std(diff_list)
    #print("diff_list:", list(np.around(np.array(diff_list),2)))
    #print("diff mean;",diff_mean)
    #print("diff std;", diff_std)

    #print("\nloss function = ", type(loss_fn).__name__)
    #print("tam dataloader = ", len(dataloader))
    #print(f"rounds:{round}")
    #print(f"Error: \n Avg loss: {test_loss:>8f}, Std dev: {test_std:>8f} \n")
    print(f"Avg Difference: {diff_mean:>8f}, Std dev: {diff_std:>8f} \n")

    #return test_loss, test_std
    return np.mean(diff_list), np.std(diff_list)

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
    Exemplo de uso: python test.py -c experiments/configs/exp-1.1.json \
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


    # INICIALIZA MODELO E DATASETS =============================================
    testloader = test_dataloader(c, ap)

    model = get_model(c.model_name, c.model_config, c.audio)
    loss_fn = get_loss(c.train_config["loss_fn"])
    optimizer = get_optimizer(c.train_config, model)

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
    print('========================================================================')
    print("TEST SUMMARY OF THE BEST CHECKPOINT")
    print('========================================================================')

    best_checkpoint_path = os.path.join(logs_dir, "best_checkpoint.pt")
    load_checkpoint(best_checkpoint_path, model, optimizer, device)

    test_loss, test_std = test(testloader, model, loss_fn, device)
    #print(f'Avg. Test Loss: {test_loss:>8f} / Test std: {test_std:>8f}\n')
    #writer.log_test_loss_std(test_loss, test_std, epoch)
    writer.add_text("avg difference",str(test_loss),0)
    writer.add_text("avg std",str(test_std),1)
    print(f'Avg. Difference: {test_loss:>8f}')
    print(f'Avg. std dev: {test_std:>8f}\n')
    writer.flush()
    writer.close()
