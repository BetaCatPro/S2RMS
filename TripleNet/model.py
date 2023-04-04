import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    def __init__(self, **config):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(config['in_channels'], config['number_of_neurons_1']),
            nn.LeakyReLU(inplace=True),
            nn.Linear(config['number_of_neurons_1'], config['number_of_neurons_2'])
        )

        self.fc2 = nn.Sequential(
            nn.Linear(config['number_of_neurons_2'] * 2, 1))

    def forward_once(self, x):
        output = self.fc1(x)
        return output

    def forward_second(self, x):
        return self.fc2(x)

    def forward(self, input1, input2, input3, get_trans_data=False):
        if not get_trans_data:
            output1 = self.forward_once(input1.float())
            output2 = self.forward_once(input2.float())
            output3 = self.forward_once(input3.float())
            concat_ap = th.cat((output1, output2), 2)
            concat_an = th.cat((output1, output3), 2)
            output_p = self.forward_second(concat_ap)
            output_n = self.forward_second(concat_an)
            return output_p.float(), output_n.float()
        else:
            output = self.forward_once(input1.float())
            return output


class PreTripletDataset:
    def __init__(self, training_labeled_path, training_pairs_path):
        self.labeled_data = pd.read_csv(training_labeled_path)
        self.pairs = pd.read_csv(training_pairs_path)

    def __getitem__(self, index):
        target_label = self.labeled_data.columns[-1]
        labeled_data = self.labeled_data.drop(columns=[target_label])
        labeled_sample = labeled_data.iloc[index]
        labeled_sample_idx = labeled_data['index'][index]

        element_list = self.pairs[
            (self.pairs['Sample1'] == labeled_sample_idx) | (self.pairs['Sample2'] == labeled_sample_idx)]
        sorted_element_list = element_list.sort_values(by='Difference', ascending=False)

        indices_pos = sorted_element_list.iloc[0]['Sample2'] if sorted_element_list.iloc[0][
                                                                    'Sample1'] == labeled_sample_idx else \
            sorted_element_list.iloc[0]['Sample1']
        indices_neg = sorted_element_list.iloc[-1]['Sample2'] if sorted_element_list.iloc[-1][
                                                                     'Sample1'] == labeled_sample_idx else \
            sorted_element_list.iloc[-1]['Sample1']

        anchor = np.delete(labeled_sample.to_numpy().astype(float), 0)
        positive = np.delete(labeled_data[labeled_data['index'] == indices_pos].to_numpy().astype(float), 0)
        negative = np.delete(labeled_data[labeled_data['index'] == indices_neg].to_numpy().astype(float), 0)

        return anchor, positive, negative

    def __len__(self):
        return len(self.labeled_data)


class TripletDataset:
    def __init__(self, training_labeled=None, training_unlabeled=None, model=None, number_pos_neg=5):
        self.labeled_data = pd.read_csv(training_labeled)
        self.unlabeled_data = pd.read_csv(training_unlabeled)
        self.model = model
        self.number_pos_neg = number_pos_neg

    def __getitem__(self, index):
        device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
        self.model.eval()

        labeled_data = self.labeled_data.iloc[:, 1:-1]
        unlabeled_data = self.unlabeled_data.iloc[:, 1:-1]

        cur_lab = labeled_data.loc[index]
        input_tensor_1 = th.from_numpy(np.asarray(cur_lab)).reshape(1, len(cur_lab)).to(device)

        list_of_distances = []
        for i in unlabeled_data.index.values.tolist():
            input_tensor_2 = th.from_numpy(np.asarray(unlabeled_data.loc[i])).reshape(1, len(
                unlabeled_data.loc[i])).to(device)
            with th.no_grad():
                output_1, output_2 = self.model(input_tensor_1, input_tensor_2, True)
                difference = F.pairwise_distance(output_1, output_2).cpu().detach().numpy()[0]
            list_of_distances.append(float(difference))

        sorted_res = np.argsort(list_of_distances)
        indices_pos, indices_neg = sorted_res[0: self.number_pos_neg], sorted_res[-self.number_pos_neg:]

        np.tile(cur_lab.to_numpy().astype(float), (5, 1))
        anchor = np.tile(cur_lab.to_numpy().astype(float), (self.number_pos_neg, 1))
        positive = unlabeled_data.iloc[indices_pos].to_numpy().astype(float)
        negative = unlabeled_data.iloc[indices_neg].to_numpy().astype(float)

        return anchor, positive, negative

    def __len__(self):
        return len(self.labeled_data)
