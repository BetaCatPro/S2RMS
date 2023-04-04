import copy
import math

import numpy as np
import pandas as pd
from m5py import M5Prime
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

from create_pairs import generate_file
from select_samples import select_sim_samples
from transform_data import transform


class S2RM:
    def __init__(self, co_learners, scale, in_channels, iteration=20, tau=1, gr=1, K=3):
        self.co_learners = co_learners
        self.scale = scale
        self.in_channels = in_channels
        self.iteration = iteration
        self.tau = tau
        self.gr = gr
        self.K = K

        self.labeled_data = None
        self.labeled_l_data = []
        self.labeled_v_data = []
        self.pi = []

    def _init_training_data_and_reg(self, data_name, labeled_data):
        self.labeled_data = labeled_data.reset_index(drop=True)

        for n in range(len(self.co_learners)):
            inner_data = self.labeled_data.sample(math.ceil(self.labeled_data.shape[0] * .8))
            outer_data = self.labeled_data.drop(inner_data.index)
            inner_data = transform(data_name, inner_data, in_channels=self.in_channels)
            outer_data = transform(data_name, outer_data, in_channels=self.in_channels)
            self.labeled_l_data.append(inner_data)
            self.labeled_v_data.append(outer_data)

            self.pi.append(None)

            self.co_learners[n].fit(inner_data[:, :-1], inner_data[:, -1])

    def _co_training(self, data_name, selected):
        origin_labeled_data_size = self.labeled_data.shape[0]

        while not selected.empty:
            self._inner_test(data_name, selected)
            for i in range(len(self.co_learners)):
                pool = shuffle(selected).reset_index(drop=True)
                _pi, _index = self._select_relevant_examples(i, pool, self.labeled_v_data[i], self.gr, data_name)
                self.pi[i] = _pi
                selected = pool.drop(_index)

                if selected.empty:
                    break

            if not any([p.size for p in self.pi]):
                break

            for i in range(len(self.co_learners)):
                if self.pi[i].size == 0:
                    continue

                self.labeled_l_data[i] = np.vstack([self.labeled_l_data[i], self.pi[i]])
                current_labeled_data = self.labeled_l_data[i]
                self.co_learners[i].fit(current_labeled_data[:, :-1], current_labeled_data[:, -1])

        now_labeled_data_size = self.labeled_data.shape[0]
        return origin_labeled_data_size, now_labeled_data_size - origin_labeled_data_size

    def _select_relevant_examples(self, j, unlabeled_data, labeled_v_data, gr, data_name):
        transform_unlabeled_data = transform(data_name, unlabeled_data, in_channels=self.in_channels)

        delta_x_u_result = []

        labeled_data_j = labeled_v_data[:, :-1]
        labeled_target_j = labeled_v_data[:, -1]

        epsilon_j = mean_squared_error(self.co_learners[j].predict(labeled_data_j), labeled_target_j,
                                       squared=False)

        others_learner_pred_list = []
        for k in range(self.K):
            for i in range(len(self.co_learners)):
                if i != j:
                    others_learner_pred_list.append(self.co_learners[i].predict(transform_unlabeled_data[:, :-1]))
        std_res = np.std(np.vstack(others_learner_pred_list).reshape(-1, len(others_learner_pred_list)), axis=1)
        stable_samples = [i for i in std_res if i < 0.4]
        stable_samples_idx = [list(std_res).index(i) for i in stable_samples]
        mean_prediction = sum([pred[stable_samples_idx] for pred in others_learner_pred_list]) / len(
            others_learner_pred_list)

        pred_unlabeled_data = np.hstack(
            [transform_unlabeled_data[stable_samples_idx][:, :-1], mean_prediction.reshape(-1, 1)])
        unlabeled_data = unlabeled_data.loc[stable_samples_idx].reset_index(drop=True)
        with_pseudo_label_unlabeled_data = pd.concat(
            [unlabeled_data.iloc[:, :-1], pd.DataFrame(mean_prediction)],
            axis=1)
        with_pseudo_label_unlabeled_data.columns = self.labeled_data.columns

        for x_u in pred_unlabeled_data:
            tmp_l_j = np.vstack([self.labeled_l_data[j], x_u])
            new_learner = copy.deepcopy(self.co_learners[j])
            new_learner.fit(tmp_l_j[:, :-1], tmp_l_j[:, -1])

            tmp_epsilon_j = mean_squared_error(new_learner.predict(labeled_data_j), labeled_target_j,
                                               squared=False)

            delta_x_u_result.append((epsilon_j - tmp_epsilon_j) / (epsilon_j + tmp_epsilon_j))

        x_u_index = np.argsort(delta_x_u_result)[::-1]
        i_counts = len([_ for _ in delta_x_u_result if _ > 0])
        i_counts = i_counts if i_counts <= gr else gr

        self.labeled_data = pd.concat([self.labeled_data, with_pseudo_label_unlabeled_data.loc[x_u_index[0:i_counts]]])

        return pred_unlabeled_data[x_u_index[0:i_counts]], [stable_samples_idx[i] for i in x_u_index[0:1]]

    def fit(self, data_name, labeled_data):
        self._init_training_data_and_reg(data_name, labeled_data)

        selected, unlabeled_data, is_select_all = select_sim_samples(data_name=data_name,
                                                                     tau=self.tau,
                                                                     in_channels=self.in_channels,
                                                                     scale=self.scale
                                                                     )

        for it in range(self.iteration):
            print('------- iter: {}/{} -------'.format(it + 1, self.iteration))
            if selected.empty:
                break

            labeled_data_size, selected_data_size = self._co_training(data_name, selected)

            if selected_data_size <= 3:
                break

            if is_select_all:
                break

            generate_file(
                self.labeled_data,
                unlabeled_data,
                None,
                is_first_split=False,
                scale=self.scale
            )

            self.tau = math.pow(.98, it + 1) * self.tau
            selected, unlabeled_data, is_select_all = select_sim_samples(
                data_name=data_name,
                tau=self.tau,
                in_channels=self.in_channels,
                scale=self.scale,
                data_size=labeled_data_size
            )

    def predict(self, data_name, data, methods=None):
        self.labeled_data = self.labeled_data.reset_index(drop=True)
        trans_unlabeled_data = transform(data_name, data, in_channels=self.in_channels)
        trans_unlabeled_data_x = trans_unlabeled_data[:, :-1]

        result = []
        if methods is None:
            methods = ['co_train']
        if 'co_train' in methods:
            pred = []
            weight = [1 / len(self.co_learners)] * len(self.co_learners)
            for learner, w in zip(self.co_learners, weight):
                pred.append(w * learner.predict(trans_unlabeled_data_x))
                result.append(learner.predict(trans_unlabeled_data_x))
            result.append(sum(pred))
        return result

    def _inner_test(self, data_name, selected):
        val_data = pd.read_csv('evaluation_dataset/{}/evaluation.csv'.format(self.scale)).drop(['index'], axis=1)
        val_data_y = val_data.iloc[:, -1]

        result = self.predict(data_name, val_data, methods=['co_train', 'm5p'])

        print('------- cur pool size: {}-------'.format(selected.shape[0]))
        print('RF1 : {}'.format(mean_squared_error(result[0], val_data_y, squared=False)))
        print('RF2 : {}'.format(mean_squared_error(result[1], val_data_y, squared=False)))
        print('RF3 : {}'.format(mean_squared_error(result[2], val_data_y, squared=False)))
        print('co-training : {}'.format(mean_squared_error(result[3], val_data_y, squared=False)))
