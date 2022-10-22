import thop
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna

from utils.dataset import train_dataloader, test_dataloader
from utils.generic import AttrDict
from utils.audio_processor import AudioProcessor

DEVICE = device = "cuda" if torch.cuda.is_available() else "cpu"
C = AttrDict({
    "dataset": {
        "train_csv": "SPIRA_Dataset_V2/metadata_train.csv",
        "test_csv": "SPIRA_Dataset_V2/metadata_test.csv"
    },
    "train_config": {
        "num_workers": 4,
        "batch_size": 5
    },
    "audio": {
        "sr": 16000,
        "hop_length": 160,
        "win_length": 400,
        "n_fft": 1200,
        "n_mfcc": 40,
        "n_mels": 40,
        "mono": False,
        "window_length": 4,
        "step": 1
    }
})

AP = AudioProcessor(**C.audio)

TRAIN_LOADER = train_dataloader(C, AP)
VAL_LOADER = test_dataloader(C, AP)

class Transpose(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        inp = inp.unsqueeze(1)
        # inp = inp.transpose(1, 2).contiguous()
        return inp

def objective(trial):
    # Generate the model.
    model = define_model(trial).to(DEVICE)
    loss_fn = nn.MSELoss()

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    for epoch in range(10):
        train_model(model, optimizer, loss_fn, TRAIN_LOADER)
        loss = eval_model(model, VAL_LOADER, loss_fn)

        trial.report(loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    loss = eval_model(model, VAL_LOADER, loss_fn)

    return loss

def define_model(trial):
    """
    Define a estrutura do modelo a ser testado
    """
    n_layers = trial.suggest_int("n_layers", 2, 4, step=1)
    layers = []
    layers.append(Transpose())

    in_features = 1
    for i in range(n_layers):
        out_features = trial.suggest_categorical(f"n_out_features_{i}", [4, 8, 16, 32, 64])
        kernel_size_conv = trial.suggest_int(f"n_kernel_size_{i}", 1, 7, step=1)
        dilation = trial.suggest_int(f"n_dilation_{i}", 1, 2)
        layers.append(nn.Conv2d(in_features, out_features, kernel_size=(kernel_size_conv, 1), dilation=dilation))
        layers.append(nn.GroupNorm(out_features//2, out_features))
        layers.append(nn.Mish())
        layers.append(nn.MaxPool2d(kernel_size=(2, 1)))
        p = trial.suggest_float(f"dropout_{i}", 0.1, 0.4, step=0.1)
        layers.append(nn.Dropout(p))

        in_features = out_features

    layers.append(nn.Flatten())

    inp = torch.zeros(1, 401, 40)
    toy_activation_shape = nn.Sequential(*layers)(inp).shape

    layers.append(nn.Linear(toy_activation_shape[1], 1))

    return nn.Sequential(*layers)


# Defines training and evaluation.
def train_model(model, optimizer, loss_fn, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        loss_fn(model(data), target).backward()
        optimizer.step()


def eval_model(model, dataloader, loss_fn):
    '''
    É possível fazer o optuna ter multi objetivos (ex: minimizar a loss e
    maximizar a acurácia), mas daí o prune não suporta
    '''
    model.eval()
    loss = 0
    # correct = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            pred = torch.round(model(data))
            # correct += (pred == target).type(torch.float).sum().item()
            loss += loss_fn(pred, target).item()
    # accuracy = 100*correct / len(valid_loader)

    # flops, _ = thop.profile(model,
    #                         inputs=(torch.randn(1, 401, 40).to(DEVICE), ),
    #                         verbose=False)
    return loss/len(dataloader)

if __name__ == '__main__':
    '''
    Busca o melhor modelo e parâmetros usando o Optuna
    Exemplo de uso: python train_optuna.py
    '''

    study = optuna.create_study(study_name="CNN 2-4 layers test", direction="minimize")
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    fig = optuna.visualization.plot_pareto_front(study)
    fig.show()

    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.show()
