import torch
import torch.nn as nn
from utils.generic import Mish

class SpiraConvV2(nn.Module):
    def __init__(self, model_config, audio_config):
        super(SpiraConvV2, self).__init__()
        self.config = model_config
        self.audio = audio_config

        convs = [
            # CNN 1
            # Conv Mish, 32, K:7x1, D:2x1
            nn.Conv2d(1, 32, kernel_size=(7, 1), dilation=(2, 1)),
            nn.GroupNorm(16, 32),
            Mish(),
            # MaxPool 2x1
            nn.MaxPool2d(kernel_size=(2, 1)),
            # Dropout 70%
            nn.Dropout(p=0.7),

            # CNN 2
            # Conv Mish, 16, K:5x1, D:2x1
            nn.Conv2d(32, 16, kernel_size=(5, 1), dilation=(2, 1)),
            nn.GroupNorm(8, 16),
            Mish(),
            # MaxPool 2x1
            nn.MaxPool2d(kernel_size=(2, 1)),
            # Dropout 70%
            nn.Dropout(p=0.7),

            # CNN 3
            # Conv Mish, 8, K:3x1, D:2x1
            nn.Conv2d(16, 8, kernel_size=(3, 1), dilation=(2, 1)),
            nn.GroupNorm(4, 8),
            Mish(),
            # MaxPool 2x1
            nn.MaxPool2d(kernel_size=(2, 1)),
            # Dropout 70%
            nn.Dropout(p=0.7),

            # CNN 4
            # Conv Mish, 4, K:2x1, D:2x1
            nn.Conv2d(8, 4, kernel_size=(2, 1), dilation=(1, 1)),
            nn.GroupNorm(2, 4),
            Mish(),
            # MaxPool 2x1
            # nn.MaxPool2d(kernel_size=(2, 1)),
            # Droupout 70%
            nn.Dropout(p=0.7)
        ]

        self.conv = nn.Sequential(*convs)

        # FC1
        # # inp = torch.zeros(1, 1, 40, self.num_feature)
        # inp = torch.zeros(1, 1, 401, 40)
        # # get out shape
        # self.fc1 = nn.Linear(4*self.conv(inp).shape[-1], self.config['fc1_dim'])

        # FC1
        inp = torch.zeros(1, 1, 100*self.audio["window_length"]+1, self.audio["n_mfcc"])
        toy_activation_shape = self.conv(inp).shape
        fc1_input_dim = toy_activation_shape[1]*toy_activation_shape[2]*toy_activation_shape[3]
        self.fc1 = nn.Linear(fc1_input_dim, self.config['fc1_dim'])

        # FC2
        self.mish = Mish()
        self.fc2 = nn.Linear(self.config['fc1_dim'],
                            self.config['fc2_dim'])
        self.dropout = nn.Dropout(p=0.7)

        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        # print(x.shape)
        # N, H, W
        x = x.unsqueeze(1)
        # N, C, H, W
        # print(x.shape)
        x = self.conv(x)

        x = x.transpose(1, 2).contiguous()
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.mish(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # x = torch.sigmoid(x)
        x = self.linear(x)

        return x
