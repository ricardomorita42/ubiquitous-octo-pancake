import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
from pprint import pprint

from torch.utils.data import DataLoader

matplotlib.use('Agg')

from utils.audio_processor import AudioProcessor
from utils.dataset import Dataset, own_collate_fn
from utils.generic import load_config
from utils.get_params import get_model

def test(dataloader, model, device, loss_fn=None):
    feature_list, mse_loss_list, mae_loss_list = [], [], []
    mse_under_92, mse_above_92 = [], []
    mae_under_92, mae_above_92 = [], []
    mae_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    model.eval()

    with torch.no_grad():
        for features, targets in dataloader:
            features, targets = features.to(device), targets.to(device)
            pred = model(features)
            #print("pred = ", pred.view(pred.size(0)))
            #print("targets = ", targets.view(targets.size(0)))

            for x, y, feat in zip(targets, pred, features):
                feature_list.append(feat)
                temp_mse = mse_loss(y, x).item()
                temp_mae = mae_loss(y, x).item()
                mse_loss_list.append(temp_mse)
                mae_loss_list.append(temp_mae)

                if x >= 92:
                    mse_above_92.append(temp_mse)
                    mae_above_92.append(temp_mae)
                else:
                    mse_under_92.append(temp_mse)
                    mae_under_92.append(temp_mae)

    mse_loss_mean = np.mean(mse_loss_list)
    mae_loss_mean = np.mean(mae_loss_list)
    msea_loss_mean = np.mean(mse_above_92)
    mseu_loss_mean = np.mean(mse_under_92)
    maea_loss_mean = np.mean(mae_above_92)
    maeu_loss_mean = np.mean(mae_under_92)
    mse_loss_std = np.std(mse_loss_list)
    mae_loss_std = np.std(mae_loss_list)
    msea_loss_std = np.std(mse_above_92)
    mseu_loss_std = np.std(mse_under_92)
    maea_loss_std = np.std(mae_above_92)
    maeu_loss_std = np.std(mae_under_92)

    print(f"MAE - Error: \n \
        Geral: Avg loss: {mae_loss_mean:>8f}, Std dev: {mae_loss_std:>8f} \n \
        >= 92: Avg loss: {maea_loss_mean:>8f}, Std dev: {maea_loss_std:>8f} \n \
        <  92: Avg loss: {maeu_loss_mean:>8f}, Std dev: {maeu_loss_std:>8f} \n")

    print(f"MSE - Error: \n \
        Geral: Avg loss: {mse_loss_mean:>8f}, Std dev: {mse_loss_std:>8f} \n \
        >= 92: Avg loss: {msea_loss_mean:>8f}, Std dev: {msea_loss_std:>8f} \n \
        <  92: Avg loss: {mseu_loss_mean:>8f}, Std dev: {mseu_loss_std:>8f} \n")

    if type(loss_fn).__name__ == "L1Loss":
        return mae_loss_mean, mae_loss_std
    elif type(loss_fn).__name__ == "MSELoss":
        return mse_loss_mean, mse_loss_std

    return feature_list, mae_loss_list, mse_loss_list


def info(feat_list, mae_list, mse_list, d, device, num=10):
    sorted_indexes = np.argsort(mae_list)
    max_error_audio_paths = []
    max_error_features = [feat_list[i].transpose(0,1) for i in sorted_indexes[-num:]]

    dataset = d.getWholeDataset()

    for key, val in dataset.items():
        for feature in val[0]:
            for i in range(num):
                if torch.equal(max_error_features[i], torch.tensor(feature, device=device)):
                    max_error_audio_paths.append([key, i])

    max_error_audio_paths.sort(key=lambda x: x[1])

    print("Mediana MAE", mae_list[sorted_indexes[len(sorted_indexes) // 2]])
    print("Mediana MSE", mse_list[sorted_indexes[len(sorted_indexes) // 2]])
    print("\nÁudios com maiores erros: (path / MAE / MSE)")

    for n, i in enumerate(sorted_indexes[-num:]):
        print(max_error_audio_paths[n][0], "/", round(mae_list[i], 4), "/", round(mse_list[i], 4))

def load_checkpoint(path, model, device):
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location=device)
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except:
            print("Falhou ao inicializar modelo, talvez seja outro.")
            quit()
    else:
        print("Não localizou o arquivo ", path)
        quit()


if __name__ == '__main__':
    '''
    Exemplo de uso: python test.py -c experiments/configs/exp-1.1.json \
                    -m checkpoints/exp-1.1/checkpoint_25.pt
    '''

    # Converte e carrega arquivo json com dados do experimento
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config_path',
                        type=str,
                        default="experiments/configs/exp-1.1.json",
                        help="json file with configurations")
    parser.add_argument('-m',
                        '--model_path',
                        type=str,
                        default=None,
                        help="path of the model checkpoint .pt file")

    args = parser.parse_args()
    c = load_config(args.config_path)
    checkpoint_path = args.model_path

    if checkpoint_path is None:
        checkpoint_path = os.path.join(c.train_config["logs_dir"],
                                       'best_checkpoint.pt')

    ap = AudioProcessor(**c.audio)
    d = Dataset(ap, c.dataset["test_csv"])

    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    testloader = DataLoader(dataset=d,
                      num_workers=c.train_config["num_workers"],
                      batch_size=c.train_config["batch_size"],
                      collate_fn=own_collate_fn,
                      drop_last=True,
                      sampler=None)

    model = get_model(c.model_name, c.model_config, c.audio)

    model.to(device)

    load_checkpoint(checkpoint_path, model, device)

    print('========================================================================')
    print("TEST SUMMARY", args.config_path)
    print('========================================================================')

    # print("Model:", c.model_name)
    # print("Audio configuraiton:")
    # pprint(c.audio)
    # print("")

    features, mae, mse = test(testloader, model, device)
    info(features, mae, mse, d, device)
