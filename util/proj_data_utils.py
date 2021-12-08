from config import config
from random import sample
import os
import csv

config_data = config.load_config_from_file()

def parse_data(datadir):
    """
    This function is used for randomly sampling 50,000 data points from a specified directory
    :param datadir: path to directory that contains the data files
    :return: randomly selected list of files to be loaded as datapoints
    """

    # Store the list of files to be returned
    file_list = []

    for root, directories, filenames in os.walk(datadir):  # root: median/1
        # Randomly sample 50,000 files for training
        files = sample(filenames, config_data['model_params']['data_size'])
        for filename in files:
            if filename.endswith('.npy'):
                filei = os.path.join(root, filename)
                file_list.append(filei)
    return file_list


def drop_columns(encoded_df, colnames):
    """
    This function reads a file called 'drop_cols' and drops the specified columns from the dataframe
    :param encoded_df: Original dataframe
    :return: Dataframe without columns, length of remaining columns, List of remaining columns
    """
    file = open("drop_cols.csv", "r")
    csv_reader = csv.reader(file)
    dropped_cols = [row for row in csv_reader][0]
    # print(dropped_cols[0])
    encoded_df = encoded_df.drop(columns=dropped_cols)

    remaining_cols = [x for x in colnames if x not in dropped_cols]
    print('Remaining cols: ', len(remaining_cols))
    return encoded_df, len(remaining_cols), remaining_cols



if __name__ == '__main__':
    config_data = config.load_config_from_file()
    cols_to_drop_file = config_data['data_preprocessing']['cols_to_drop_csv_file']
    print(cols_to_drop_file)
