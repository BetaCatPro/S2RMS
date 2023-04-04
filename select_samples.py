import os

import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as F
import yaml

from TripleNet.model import SiameseNetwork

np.set_printoptions(precision=3, suppress=True)


def load_data(data_name, in_channels, scale):
    with open('./configs/config.yml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

        unlabeled_path = os.path.join(config['dataset_params']['dataset_path'], str(scale), 'unlabeled_data.csv')
        unlabeled_data = pd.read_csv(unlabeled_path)
        unlabeled_data = unlabeled_data.drop(columns=[unlabeled_data.columns[0]])
        unlabeled_data_copy = unlabeled_data.drop(columns=[unlabeled_data.columns[-1]])

        load_path = os.path.join(config['logging_params']['save_dir'], '{}_best_model.pth'.format(data_name))
        config['model_params']['in_channels'] = int(in_channels)
        model = SiameseNetwork(**config['model_params'])
        model.load_state_dict(th.load(load_path, map_location=lambda storage, loc: storage.cuda(0)))
        model.eval()

        labeled_path = os.path.join(config['dataset_params']['dataset_path'], str(scale), 'labeled_data.csv')
        labeled_data = pd.read_csv(labeled_path)
        labeled_data = labeled_data.drop(columns=[labeled_data.columns[0], labeled_data.columns[-1]])

        return model, labeled_data, unlabeled_data, unlabeled_data_copy


def select_sim_samples(data_name, tau, in_channels, scale, data_size=0):
    model, labeled_data, unlabeled_data, unlabeled_data_copy = load_data(data_name, in_channels, scale)

    has_selected_unlabeled_list = []
    final_selected_list = pd.DataFrame()
    select_all = False

    # for labeled in labeled_data.iloc[data_size:, :].itertuples():
    for labeled in labeled_data.itertuples():
        input_tensor_1 = th.from_numpy(np.asarray(labeled[1::])).reshape(1, len(labeled[1::]))

        list_similarity = []
        if unlabeled_data.empty:
            break

        for unlabeled in unlabeled_data_copy.itertuples():
            input_tensor_2 = th.from_numpy(np.asarray(unlabeled[1::])).reshape(1, len(unlabeled[1::]))

            with th.no_grad():
                input_tensor_1 = th.reshape(input_tensor_1, (input_tensor_1.shape[0], -1, in_channels))
                input_tensor_2 = th.reshape(input_tensor_2, (input_tensor_2.shape[0], -1, in_channels))
                output, _ = model(input_tensor_1, input_tensor_2, input_tensor_2)
                list_similarity.append(float(output.item()))
        selected_index = np.argsort(list_similarity)[:1]
        sims_tau = [list_similarity[int(idx)] for idx in list(selected_index)]
        satisfy_tau = [t for t in sims_tau if t < tau]
        if satisfy_tau:
            selected_index = selected_index[0:len(satisfy_tau)]
            has_selected_unlabeled_list.append(unlabeled_data.iloc[selected_index])
            unlabeled_data = unlabeled_data.reset_index(drop=True).drop(selected_index)
            unlabeled_data_copy = unlabeled_data_copy.reset_index(drop=True).drop(selected_index)

    if len(has_selected_unlabeled_list) != 0:
        final_selected_list = pd.concat(has_selected_unlabeled_list)

    if unlabeled_data.empty:
        select_all = True
    return final_selected_list, unlabeled_data, select_all
