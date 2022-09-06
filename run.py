from inspect import ArgInfo
import torch
import argparse
import os, sys, json
from dataloader import ArgoverseDataset, my_collate
from torch.utils.data import Dataset, DataLoader
from train import train_model

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default='linear', nargs='?',
                    help='Choose a model type for prediction')
args = vars(parser.parse_args())

def main(args):
    # read from json file
    print(args)
    f = open('config.json')
    config = json.load(f)
    batch_size = config['batch_size']

    print(f'CUDA availability: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f'GPU name: {torch.cuda.get_device_name(i)}')

    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    print("using cuda:{}".format(0))

    # initialize the training dataset
    train_data = ArgoverseDataset(data_path=config['train_path'])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, collate_fn=my_collate, num_workers=0)
    val_data = ArgoverseDataset(data_path=config['val_path'])
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=my_collate, num_workers=0)
    model = train_model((train_loader, val_loader), config, device, args['model_type'])
    torch.save(model.state_dict(), 'linear.pt')

if __name__ == "__main__":
    main(args)
