import torch
import torch.nn as nn
import torchvision.models as models

class linearModel(nn.Module): 
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(45*2, 5*2)
    
    def forward(self, x): 
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 45*2)
        output = self.linear(x)
        output = output.reshape(batch_size, 5, 2)
        return output

class baselineLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_units = config['hidden_units']
        self.num_layers = config['num_layers']
        self.device = 0

        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=self.hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )
        
        self.linear = nn.Conv1d(self.hidden_units, 2, 1)

    def forward(self, x):
        output,_ = self.lstm(x)
        output = output.transpose(1,2)
        output = self.linear(output)
        output = output.transpose(1,2)
        return output

    def forward_test(self, x, num_steps=5):
        result = []
        batch_size = x.shape[0]
        h = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(self.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(self.device)

        for i in range(num_steps): 
            output,(h, c) = self.lstm(x, (h, c))
            output = output[:,-1:,:]
            output = output.transpose(1,2)
            output = self.linear(output)
            output = output.transpose(1,2)
            result.append(output)

        result = torch.cat(result, 1)
        return result
