import torch
import torch.nn as nn
import torchvision.models as models


class baselineLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_units = config['hidden_units']
        self.num_layers = config['num_layers']
        self.device = 0

        self.lstm_45 = nn.LSTM(
            input_size=2,
            hidden_size=self.hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )
        self.lstm_46 = nn.LSTM(
            input_size=2,
            hidden_size=self.hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )
        self.lstm_47 = nn.LSTM(
            input_size=2,
            hidden_size=self.hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )
        self.lstm_48 = nn.LSTM(
            input_size=2,
            hidden_size=self.hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )
        self.lstm_49 = nn.LSTM(
            input_size=2,
            hidden_size=self.hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear_46 = nn.Linear(in_features=self.hidden_units, out_features=2)
        self.linear_47 = nn.Linear(in_features=self.hidden_units, out_features=2)
        self.linear_48 = nn.Linear(in_features=self.hidden_units, out_features=2)
        self.linear_49 = nn.Linear(in_features=self.hidden_units, out_features=2)
        self.linear_50 = nn.Linear(in_features=self.hidden_units, out_features=2)



    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.randn(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(self.device)
        c0 = torch.randn(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(self.device)

        _,(last_hidden, last_cell) = self.lstm_45(x, (h0, c0))
        out_46 = torch.unsqueeze(self.linear_46(last_hidden[1]),1)

        _, (last_hidden, last_cell) = self.lstm_46(out_46, (last_hidden, last_cell))
        out_47 = torch.unsqueeze(self.linear_47(last_hidden[1]),1)

        _, (last_hidden, last_cell) = self.lstm_47(out_47, (last_hidden, last_cell))
        out_48 = torch.unsqueeze(self.linear_48(last_hidden[1]),1)

        _, (last_hidden, last_cell) = self.lstm_48(out_48, (last_hidden, last_cell))
        out_49 = torch.unsqueeze(self.linear_49(last_hidden[1]),1)

        _, (last_hidden, last_cell) = self.lstm_49(out_49, (last_hidden, last_cell))
        out_50 = torch.unsqueeze(self.linear_50(last_hidden[1]),1)

        output = torch.cat((out_46, out_47, out_48, out_49, out_50), dim=1)

        return output
