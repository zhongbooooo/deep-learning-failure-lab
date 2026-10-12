import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)

        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)

        return x


class LargeMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32 * 32 * 3, 1024)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(512, 256)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)

        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)

        return x


class DropoutMLP(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32 * 32 * 3, 1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(512, 256)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc4 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)

        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)

        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)

        x = self.relu3(self.fc3(x))
        x = self.dropout3(x)

        x = self.fc4(x)

        return x