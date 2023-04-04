import math
import os

import datetime
import pandas as pd
import yaml
from scipy.io import arff

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from create_pairs import generate_file
from TripleNet.tripleNet import train_triple_net
from ssrm import S2RM

import warnings

warnings.filterwarnings("ignore")


def split_labeled_and_unlabeled_data(target_data, scale, rs):
    training_data = target_data.sample(2000, random_state=rs)
    labeled_data = training_data.sample(math.ceil(2000 * scale), random_state=rs)
    unlabeled_data = training_data.drop(labeled_data.index)
    test_data = target_data.drop(training_data.index)

    return labeled_data, unlabeled_data, test_data


if __name__ == '__main__':
    with open('configs/config.yml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    data_dir = os.listdir(config['experiment_params']['base_data_dir'])

    for run_iter in range(config['experiment_params']['run_iter']):
        for scale in config['dataset_params']['scale_labeled_samples']:
            for file_name in data_dir:
                start_time = datetime.datetime.now()

                file = file_name.split('.')[0]

                print('****************** Exp:{}, Scale: {}, DataSet: {} ******************'.format(run_iter, scale,
                                                                                                    file))

                data, meta = arff.loadarff(os.path.join(config['experiment_params']['base_data_dir'], file_name))
                data_labeled = pd.DataFrame(data)
                in_channels = data_labeled.shape[1] - 1
                labeled_data, unlabeled_data, test_data = split_labeled_and_unlabeled_data(data_labeled, scale,
                                                                                           rs=run_iter)

                generate_file(labeled_data, unlabeled_data, test_data, scale)
                train_triple_net(file, in_channels=in_channels, scale=scale)

                learner1 = RandomForestRegressor(n_estimators=100, min_samples_split=2, random_state=1)
                learner2 = RandomForestRegressor(n_estimators=120, min_samples_leaf=2, random_state=2)
                learner3 = RandomForestRegressor(n_estimators=150, min_samples_split=3, random_state=3)

                reg = S2RM(
                    co_learners=[learner1, learner2, learner3],
                    scale=scale,
                    in_channels=in_channels,
                    iteration=config['experiment_params']['iteration'],
                    tau=config['experiment_params']['tau'],
                    gr=config['experiment_params']['gr'],
                    K=config['experiment_params']['K']
                )
                reg.fit(file, labeled_data)
                methods = ['co_train', 'm5p']
                pred = reg.predict(file, test_data, methods=methods)

                r_mse = []
                pd_dict = {
                    'experiment_iter': run_iter,
                    'data': file,
                }
                for res, me in zip(pred[3:], ['co_train']):
                    cur_rmse = mean_squared_error(res, test_data.iloc[:, -1], squared=False)
                    r_mse.append(cur_rmse)
                    pd_dict['{}_rmse'.format(me)] = cur_rmse
                    print('****************** DataSet: {}, {} RMSE: {} ******************'.format(file, me,
                                                                                                  cur_rmse))
                save_dir = './docs/experiment/{}/{}'.format(file, scale)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                pd.DataFrame(pd_dict, index=[0]).to_csv(
                    '{}/{}-{}.csv'.format(save_dir, run_iter, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))

                end_time = datetime.datetime.now()
                print("耗时: {}秒".format(end_time - start_time))
