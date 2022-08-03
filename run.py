from inspect import ArgInfo
import torch
import argparse
import os, sys, json
from dataloader import ArgoverseDataset, my_collate
from torch.utils.data import Dataset, DataLoader
from train import train_model

parser = argparse.ArgumentParser()
parser.add_argument('--on_net', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='Whether to train locally or on google collab')


args = vars(parser.parse_args())

def main(args):
    if args['on_net']:
        config_path = 'config_net.json'
    else:
        config_path = 'config.json'

    # read from json file
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
    model = train_model((train_loader, val_loader), config, device)
    torch.save(model.state_dict(), 'first.pt')

if __name__ == "__main__":
    main(args)
