import argparse
import os
import torch
import matplotlib
matplotlib.use('Agg')

from utils.audio_processor import AudioProcessor
from utils.dataset import train_dataloader, test_dataloader
from utils.generic import load_config
from utils.tensorboard import TensorboardWriter
from utils.get_params import get_loss, get_model, get_optimizer, get_scheduler

# Não usados
from unittest.util import _MAX_LENGTH
from random import choice

def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    train_loss = 0

    for batch_id, (features, targets) in enumerate(dataloader):
        features, targets = features.to(device), targets.to(device)
        pred = model(features)
        # print('Shape of features:', features.shape)
        # print('Shape of targets:', targets.shape, 'type:', targets.dtype)
        # print('Shape of pred:', pred.shape, 'type:', pred.dtype)

        loss = loss_fn(pred, targets)
        train_loss += loss.item()

        # Bakpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch_id % 5 == 0:
        #     loss, current = loss.item(), batch_id * len(features)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]")
        #     print("targets:", targets, "/ pred", pred)

    return train_loss/len(dataloader)

def test(dataloader, model, loss_fn, device, acceptable_interval):
    model.eval()
    num_batches = len(dataloader)
    test_loss, test_acc = 0, 0

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
                #print("({}, {})".format(x.item(),y.item()),end=" ")
                #print("c",(y + acceptable_interval <= x <= y - acceptable_interval))
                if ((y.item() - acceptable_interval) <= x.item() <= (y.item() + acceptable_interval)):
                    #print("({}, {})".format(x.item(),y.item()),end=" ")
                    correct_items += 1

            test_acc += correct_items / len(pred)
            #print(f"\nround:{round}")
            #print("total items:", len(pred))
            #print("number of hits:", correct_items)
            #print("hits/total items:", correct_items/len(pred))

    test_loss /= round
    test_acc = 100*test_acc/round
    print(f"Test Error: \n Accuracy: {test_acc:>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, test_acc

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
    testloader = test_dataloader(c, ap)

    model = get_model(c.model_name, c.model_config, c.audio)
    loss_fn = get_loss(c.train_config["loss_fn"])
    optimizer = get_optimizer(c.train_config, model)

    model.to(device)

    # INICIALIZA CHECKPOINT ====================================================

    if (checkpoint_path is not None):
        print("Lendo checkpoint", checkpoint_path)

        try:
            if device == 'cuda':
                checkpoint = torch.load(os.path.abspath(checkpoint_path), map_location="cuda:0")
            else:
                checkpoint = torch.load(os.path.abspath(checkpoint_path))
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"], )
            epoch = checkpoint["epoch"]
        except:
            print("Aconteceu um erro na leitura do checkpoint", checkpoint_path)
            epoch = 0
    else:
        epoch = 0

    # LEARNING RATE SCHEDULER ==================================================

    scheduler = get_scheduler(c.train_config, optimizer, epoch)

    # TREINO / EPOCHS ==========================================================

    while epoch < c.train_config['epochs']:
        print('=================================================')
        print("Epoch %d" % epoch)
        print('=================================================')

        train_loss = train(trainloader, model, loss_fn, optimizer, device)
        test_loss, test_acc = test(testloader, model, loss_fn, device, acceptable_interval)

        if scheduler:
            scheduler.step()

        epoch += 1

        if epoch%c.train_config["summary_interval"] == 0:
            writer.log_train_loss(train_loss, epoch)
            writer.log_test_loss_acc(test_loss, test_acc, epoch)
            print("Write summary at epoch", epoch)
            print(f'Avg. Train Loss: {train_loss:>8f}')
            print(f'Avg. Test Loss: {test_loss:>8f} / Test Acc: {test_acc:>0.1f}%\n')

        if epoch%c.train_config["checkpoint_interval"] == 0:
            save_path = os.path.join(logs_dir, "checkpoint_%d.pt" % epoch)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, save_path)
            print("Salvou checkpoint em", save_path)

    print("Done!")

    # Assegura de que todos tensorboard logs foram escritos e encerra ele
    writer.flush()
    writer.close()

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
