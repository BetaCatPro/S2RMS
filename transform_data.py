import os

import numpy as np
import torch as th
import yaml

from TripleNet.model import SiameseNetwork

np.set_printoptions(precision=3, suppress=True)


def transform(data_name, data, in_channels):
    with open('./configs/config.yml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    load_path = os.path.join(config['logging_params']['save_dir'], '{}_best_model.pth'.format(data_name))
    config['model_params']['in_channels'] = int(in_channels)
    model = SiameseNetwork(**config['model_params'])
    model.load_state_dict(th.load(load_path, map_location=lambda storage, loc: storage.cuda(0)))
    model.eval()

    input_tensor = th.from_numpy(data.iloc[:, :-1].values)
    transformed_data = model(input_tensor, input_tensor, None, True).detach().numpy()
    label = data.iloc[:, -1].values.reshape(-1, 1)

    return np.hstack([transformed_data, label])
