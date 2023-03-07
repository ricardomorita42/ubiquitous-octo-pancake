import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from utils.audio_processor import AudioProcessor
from utils.dataset import test_dataloader
from utils.generic import load_config
from utils.tensorboard import TensorboardWriter
from utils.get_params import get_loss, get_model, get_optimizer, get_scheduler

def test(dataloader, model, loss_fn, device):
    errors = []
    loss_list = []
    relative_diff_list = []

    model.eval()
    with torch.no_grad():
        round = 0
        diff_list = []
        TP,TN,FP,FN = 0,0,0,0
        # numero de previsões corretas no indice 0, previsões erradas no indice 1
        above94 = [0,0]
        below90 = [0,0]
        between = [0,0]

        for features, targets in dataloader:
            features, targets = features.to(device), targets.to(device)
            pred = model(features)
            #print(f"round:{round}")
            #print("pred = ", pred.view(pred.size(0)))
            #print("targets = ", targets.view(targets.size(0)))
            
            for x,y in zip(targets,pred):
                difference = np.abs(x.item() - y.item())
                brute_difference = x.item() - y.item()
                diff_list.append(difference)

                writer.log_rel_diff(brute_difference/x.item(),round)
                relative_diff_list.append(brute_difference/x.item())

                #target é normal (x.item() >= 92)
                if (x.item() >= 92):        
                    #pred disse que é normal (y.item() >= 92)
                    if (y.item() >= 92):
                        TN += 1 
                    #pred disse que é doente (y.item() < 92)
                    else:
                        FP += 1

                #target é doente (x.item() < 92)
                else:
                    #pred disse que é normal (y.item() >= 92)
                    if (y.item() >= 92):
                        FN += 1
                    #pred disse que é doente (y.item() < 92)
                    else:
                        TP += 1

                #estimando acerto em faixas
                if (y.item() >= 94):      #pred maior que 94
                    if (x.item() >= 92):  # paciente é saudável
                        above94[0] += 1
                    else:
                        above94[1] += 1

                elif (y.item() < 90):      #pred menor que 90
                    if (x.item() <= 92):    # paciente é doente
                        below90[0] += 1
                    else:
                        below90[1] += 1
                
                else: 
                    if (y.item() >= 92):        
                        if (x.item() >= 92):
                            between[0] += 1 
                        else:
                            between[1] += 1

                    else:
                        if (y.item() < 92):
                            between[0] += 1 
                        else:
                            between[1] += 1



            #loss = loss_fn(pred, targets).item() 
            #loss_list.append(loss)

            #print("total items:", len(pred))
            round += 1

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
    #print(f"Avg Difference: {diff_mean:>8f}, Std dev: {diff_std:>8f} \n")
    #print("relative diff list: ", relative_diff_list)

    #print(f"(TP,TN,FP,FN):{TP},{TN},{FP},{FN}")
    comparisons = TP+TN+FP+FN
    TP_p = TP / comparisons * 100
    TN_p = TN / comparisons * 100
    FP_p = FP / comparisons * 100
    FN_p = FN / comparisons * 100
    #print(f"(TP,TN,FP,FN) em %:{TP_p:0.2f},{TN_p:0.2f},{FP_p:0.2f},{FN_p:0.2f}")

    confusion_list = [TP_p, TN_p, FP_p, FN_p]

    #return test_loss, test_std
    return diff_mean, diff_std, confusion_list, relative_diff_list, above94, below90, between

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

    test_loss, test_std, test_confusion_matrix, relative_diff_list, above94,below90,between = test(testloader, model, loss_fn, device)
    #print(f'Avg. Test Loss: {test_loss:>8f} / Test std: {test_std:>8f}\n')
    #writer.log_test_loss_std(test_loss, test_std, epoch)

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

    test_confusion_str = ''
    for number in test_confusion_matrix:
        test_confusion_str = test_confusion_str + str(number) + ' '
    test_confusion_str = test_confusion_str.rstrip()

    writer.add_text("avg difference",str(test_loss),0)
    writer.add_text("avg std",str(test_std),1)
    writer.add_text("confusion matrix (TP,TN,FP,FN)",str(test_confusion_str),2)
    writer.add_text("above94",str(above94),3)
    writer.add_text("abelow90",str(below90),4)
    writer.add_text("abetween",str(between),5)
    print(f'Avg. Difference: {test_loss:>8f}')
    print(f'Avg. std dev: {test_std:>8f}\n')
    print("conf list:", test_confusion_str)
    print("above94",str(above94))
    print("below90",str(below90))
    print("between",str(between))
    writer.flush()
    writer.close()

