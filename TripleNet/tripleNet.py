import os

import yaml
from torch.utils.data import DataLoader

from TripleNet.model import *
from TripleNet.training import *


def get_pre_data_loaders(**config):
    pre_triplet_dataset = PreTripletDataset(os.path.join(config['dataset_params']['dataset_path'], 'labeled_data.csv'),
                                            os.path.join(config['dataset_params']['dataset_path'], 'pairs_train.csv'))
    pre_triplet_loader = DataLoader(pre_triplet_dataset, shuffle=True, num_workers=1,
                                    batch_size=config['exp_params']['batch_size'])
    return pre_triplet_loader


def get_cur_data_loaders(model, number_pos_neg, **config):
    triplet_dataset = TripletDataset(os.path.join(config['dataset_params']['dataset_path'], 'labeled_data.csv'),
                                     os.path.join(config['dataset_params']['dataset_path'], 'unlabeled_data.csv'),
                                     model,
                                     number_pos_neg
                                     )
    triplet_loader = DataLoader(triplet_dataset, shuffle=True, num_workers=0,
                                batch_size=config['exp_params']['batch_size'])
    return triplet_loader


def train(epochs, data_loader, model, optimizer, device, in_channels, data='', save_path=None):
    list_train_loss = []
    cur_min_val_error = float(1e8)

    for epoch in range(epochs):
        model.train()
        train_loss = train_triplet(data_loader, model, optimizer, device, in_channels)
        list_train_loss.append(train_loss)

        if (epoch + 1) % 10 == 0:
            print('Epoch: {} rank loss: {}'.format(epoch + 1, train_loss))

        if train_loss < cur_min_val_error:
            if save_path:
                torch.save(model.state_dict(), os.path.join(save_path, '{}_best_model.pth').format(data))
            cur_min_val_error = train_loss


def train_triple_net(data_name, in_channels, scale):
    with open('configs/config.yml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    config['model_params']['in_channels'] = int(in_channels)

    config['dataset_params']['dataset_path'] = os.path.join(config['dataset_params']['dataset_path_fixed'],
                                                            str(scale))
    config['dataset_params']['evaluation_path'] = os.path.join(config['dataset_params']['evaluation_path_fixed'],
                                                               str(scale))

    save_path = os.path.join(config['logging_params']['save_dir'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SiameseNetwork(**config['model_params']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['exp_params']['LR'])
    if config['exp_params']['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['exp_params']['LR'], momentum=0.9)
    elif config['exp_params']['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['exp_params']['LR'])

    pre_triplet_loader = get_pre_data_loaders(**config)
    train(config['exp_params']['epochs'], pre_triplet_loader, model, optimizer, device,
          config['model_params']['in_channels'], data_name, save_path)

    # triplet = SiameseNetwork(**config['model_params']).to(device)
    # optimizer = torch.optim.Adam(triplet.parameters(), lr=config['exp_params']['LR'])
    # triplet_loader = get_cur_data_loaders(model, config['model_params']['number_pos_neg'], **config)
    # train(config['exp_params']['epochs'], triplet_loader, triplet, optimizer, device,
    #       config['model_params']['in_channels'], data_name, save_path)
