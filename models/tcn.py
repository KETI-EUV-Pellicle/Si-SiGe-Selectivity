import torch
import torch.nn as nn

class SiGe_TCN(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size=3, dropout=0.1):
        super(SiGe_TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size,
                            padding=(kernel_size-1) * dilation_size,
                            dilation=dilation_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        
        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = self.network(x)
        x = self.global_pool(x).squeeze(-1)
        return torch.sigmoid(self.fc(x))

class SiGe_X_Range_TCN(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size=3, dropout=0.1):
        super(SiGe_X_Range_TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size,
                          padding=(kernel_size-1) * dilation_size,
                          dilation=dilation_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        
        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], 1)
    
    def forward(self, x):
        x = self.network(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)