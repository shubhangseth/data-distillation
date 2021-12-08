import numpy as np
from numpy import linalg as LA
import os
import pandas as pd
from random import sample
import csv
from config import config

config_data = config.load_config_from_file()


def parse_data(datadir):
    img_list = []
    # ctr = 0
    for root, directories, filenames in os.walk(datadir):  # root: median/1
        # Randomly sample 100,000 files for training
        files = sample(filenames, 50000)
        for filename in files:
            if filename.endswith('.npy'):
                # ctr += 1
                # if ctr == 30:
                #     break
                filei = os.path.join(root, filename)
                img_list.append(filei)
    return img_list


def collinear(arr, cols):
    """
    This function is used to create the variance decomposition matrix for the features
    :param arr: Features array
    :param cols: Columns to process
    :return:
    """

    scaling_factor = LA.norm(arr)  # default is euclidean norm
    arr = arr / scaling_factor
    U, D, V = LA.svd(arr, full_matrices=False, compute_uv=True)
    di = np.diag(D)
    mu_max = np.max(D)
    mu_min = np.min(D)
    condition_number = mu_max / mu_min
    condition_index = np.reciprocal(D)
    condition_index = condition_index * mu_max
    inv_D = LA.inv(di)
    # print(inv_D.shape)
    # print(V.shape)
    VD_inv = np.matmul(V, inv_D)
    Q = np.multiply(VD_inv, VD_inv)
    Q_bar = np.diag(np.sum(Q, axis=1))  # diagonal matrix with the row sums of Q on the main diagonal and 0 elsewhere

    variance_propotion = np.matmul(Q.T, LA.inv(Q_bar))

    df1 = pd.DataFrame(variance_propotion, columns=cols)
    df2 = pd.DataFrame(condition_index)

    df1.insert(loc=0, column='condition_index', value=df2)
    print(df1.head(20))
    df1.to_csv('colinearity_statistics.csv', index=False)

    # df1 = df1.drop(columns=['condition_index'])

    variance_matrix = df1.to_numpy()
    return variance_matrix


def find_difficult_to_learn(variance_matrix, n=1, m=2):
    """
    This function is used to find the most difficult to learn features using the variance decomposition
    matrix. It also returns the correlated features for each hard to learn feature
    :param variance_matrix: Variance decomposition matrix
    :param n: Number of hard to learn features to select
    :param m: Number of correlated features to select per hard to learn feature
    :return: Concatenated list of hard to learn features and their corresponding correlated features
    """
    condition_indexes = variance_matrix[:, 0]
    most_difficult_to_learn = condition_indexes.argsort()[-n:]
    print(most_difficult_to_learn)

    relative_features = []
    for i in most_difficult_to_learn:
        feature_causing_learning_issue = variance_matrix[i, 1:].argsort()[-m:]
        print(i, feature_causing_learning_issue)
        relative_features = np.append(relative_features, feature_causing_learning_issue)

    print(relative_features)

    return np.append(most_difficult_to_learn, relative_features).astype(int)


if __name__ == '__main__':
    data_filepath = config_data['data_filepath']
    filenames = parse_data(data_filepath)
    print('filenames len is: ' + str(len(filenames)))
    # print(filenames)

    combined_data = np.array([np.load(fname) for fname in filenames])
    print('Combined data successfully')

    where_are_NaNs = np.isnan(combined_data)
    combined_data[where_are_NaNs] = 0

    colnames_file = pd.read_csv(config_data['data_preprocessing']['cols_filename'])
    colnames = list(colnames_file['0'])
    encoded_df = pd.DataFrame(combined_data, columns=colnames)

    file = open(config_data['data_preprocessing']['cols_to_drop_csv_file'], "r")
    csv_reader = csv.reader(file)
    dropped_cols = [row for row in csv_reader][0] + ['SSP', 'PINCP']
    # print(dropped_cols[0])
    encoded_df = encoded_df.drop(columns=dropped_cols)

    remaining_cols = [x for x in colnames if x not in dropped_cols]
    print('Remaining cols: ', len(remaining_cols))
    collinear(encoded_df.to_numpy(), remaining_cols)
