import argparse
from unittest.util import _MAX_LENGTH

from utils.audio_processor import AudioProcessor
from utils.dataset import Dataset, train_dataloader, test_dataloader
from utils.generic import load_config
from models.cnn import SpiraConvV2

from random import choice

import torch
import torch.nn as nn

from torchsummary import summary

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)

    model.train()

    for batch_id, (features, targets) in enumerate(dataloader):
        features, targets = features.to(device), targets.to(device)
        # Computa o erro da predição
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

def get_model(model_name, train_config, device):
    if model_name == "SpiraConvV2":
        model = SpiraConvV2(train_config).to(device)
        return model
    else:
        raise Exception("O modelo '" + model_name + "' não é suportado")

if __name__ == '__main__':
    '''
    Exemplo de uso: python train.py -c experiments/configs/exp-1.1.json
    '''

    # Converte e carrega arquivo json com dados do experimento
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str,
                        default="experiments/configs/exp-1.1.json",
                        help="json file with configurations")

    args = parser.parse_args()
    c = load_config(args.config_path)

    ap = AudioProcessor(**c.audio)

    # ==========================================================================
    trainloader = train_dataloader(c, ap)
    testloader = test_dataloader(c, ap)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(c.model_name, c.train_config, device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)

    for epoch in range(c.train_config['epochs']):
        train(trainloader, model, loss_fn, optimizer, device)
        test(testloader, model, loss_fn, device)

        print('=================================================')
        print("Epoch %d end" % epoch)
        print('=================================================')

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
