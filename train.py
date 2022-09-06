from tqdm import tqdm
import torch
from models import baselineLSTM
from models import linearModel
import torch.nn as nn
from utils import EarlyStopping
import os
import matplotlib.pyplot as plt
import sys

def prepare_model(config, device, model_type):
    # Print the type of the variable, sys.argv
    if model_type == 'LSTM': 
        model = baselineLSTM(config)
        model.load_state_dict(torch.load('lstm.pt'))
    elif model_type == 'linear': 
        model = linearModel(config)
        model.load_state_dict(torch.load('linear.pt'))
    model.eval()
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_function = nn.MSELoss()

    return model, optimizer, loss_function
    


def train_model(dataloader, config, device, model_type):

    train_loader, val_loader = dataloader

    model, optimizer, loss_function = prepare_model(config, device, model_type)

    train_loss = []
    vali_loss = []
    early_stop = EarlyStopping()

    for epoch_count in range(config['epoch']):
        model.train()
        if (epoch_count+1) % 10 == 0:
            print('epoch {}:'.format(epoch_count+1))
        train_loss_epoch = 0
        for i, (x,y) in enumerate(tqdm(train_loader)):
            data, target = x.to(device), y.to(device)
            if model_type == 'LSTM': 
                mixed = torch.cat([data, target], 1)
                output = model(mixed[:,:-1,:])[:,-5:,:]
            elif model_type == 'linear': 
                output = model(data)
            loss = loss_function(output, target)
            train_loss_epoch += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_loss_epoch = vali_model(model, loss_function, val_loader, device)

        train_loss.append(train_loss_epoch/len(train_loader.dataset))
        vali_loss.append(val_loss_epoch)

        print('train loss: ', train_loss_epoch)
        print('val loss: ', val_loss_epoch)

        # test for early stop
        early_stop(val_loss_epoch)
        if early_stop.early_stop:
            break

    return model


def vali_model(model, loss_function, vali_loader, device, model_type):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for data, target in vali_loader:
            data, target = data.to(device), target.to(device)
            if model_type == 'LSTM': 
                outputs = model.forward_test(data)
            elif model_type == 'linear': 
                outputs = model(data)
            loss = loss_function(outputs, target)
            val_loss += loss.item()
        val_loss = val_loss / len(vali_loader.dataset)

        return val_loss