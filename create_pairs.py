import itertools
import os

import numpy as np
import pandas as pd
from scipy.spatial import distance

from utils.tools import to_csv

np.set_printoptions(precision=3, suppress=True)


def create_labeled_pairs(data):
    x_train = data.iloc[:, :-1]
    pairs_label = []
    lst = x_train.index.values.tolist()

    for rand1, rand2 in itertools.combinations_with_replacement(lst, 2):
        if rand1 == rand2:
            continue
        sim = distance.minkowski(x_train.loc[rand1], x_train.loc[rand2])
        pair = (rand1, rand2, sim)
        pairs_label.append(pair)

    return pairs_label


def generate_file(labeled_data, unlabeled_data, evaluation_data, scale, is_first_split=True):
    labeled_data = labeled_data.reset_index(drop=True)
    unlabeled_data = unlabeled_data.reset_index(drop=True)

    start_path = 'evaluation_dataset'
    if not is_first_split:
        start_path = 'training_dataset'

    directory_labeled_data_name = '{}'.format(scale)
    base_path = os.path.join(start_path, directory_labeled_data_name)
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    if is_first_split:
        to_csv(evaluation_data, os.path.join(base_path, 'evaluation.csv'), index=True)

        start_path = 'training_dataset'
        base_path = os.path.join(start_path, directory_labeled_data_name)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        print('------- Starting creating pairs -------')

        pairs_labeled = np.asarray(create_labeled_pairs(labeled_data))
        df_pairs_labeled = pd.DataFrame.from_records(
            pairs_labeled, columns=['Sample1', 'Sample2', 'Difference'])
        to_csv(df_pairs_labeled, os.path.join(base_path, 'pairs_train.csv'))

        print('------- Finished creating pairs -------')

    to_csv(labeled_data, os.path.join(base_path, 'labeled_data.csv'), index=True)
    to_csv(unlabeled_data, os.path.join(base_path, 'unlabeled_data.csv'), index=True)
