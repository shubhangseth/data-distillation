from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from datetime import datetime
from random import sample

import pandas as pd
import time
import torch
import numpy as np
import torch.utils.data as data_utils
import os
import wandb
import csv
import math

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
        files = sample(filenames, 50000)
        for filename in files:
            if filename.endswith('.npy'):
                filei = os.path.join(root, filename)
                file_list.append(filei)
    return file_list


def test_model(model):
    """
    This function is used to test the model on a pre-split test dataset
    :param model: The trained model to be tested
    :return: Predictions from the model
    """

    # Load the test data
    test_tensor = data_utils.TensorDataset(test, test_target)
    test_loader = data_utils.DataLoader(dataset=test_tensor, batch_size=batch_size, shuffle=False)

    # Put the model in eval mode
    model.eval()
    predictions = []
    for batch_num, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        predictions += torch.flatten(outputs).tolist()

    # Compute R2 score for the predictions and write it to a file
    r2 = r2_score(test_target.tolist(), predictions)
    print('R2 Score: {}'.format(r2))
    with open("/home/ubuntu/proj/run/{}/r2.txt".format(runtime), "a") as fin:
        fin.write(str(r2))

    return predictions


def validate_model(model, epoch):
    """
    This function is used to validate a model while training
    :param model: Model to be validated
    :param epoch: Current epoch
    :return: Avg loss for the validation dataset
    """

    model.eval()
    avg_loss = 0.0
    total_loss = 0.0
    num_batches = len(val_loader)
    num_observations = 0.0
    for batch_num, (x, y) in enumerate(val_loader):
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        loss = criterion(outputs, y.view(y.shape[0], 1))
        avg_loss += loss.item()

        total_loss += loss.item()
        wandb.log({"batch_val_loss": loss,
                   'val_batch_num': batch_num})

        num_observations += len(x)

        if batch_num % 10 == 9:
            print('Epoch: {}\tBatch: {}\tAvg-Loss validation: {:.8f}'.format(epoch, batch_num + 1, avg_loss / 10))
            with open("/home/ubuntu/proj/run/{}/avg_val_loss.txt".format(runtime), "a") as fin:
                fin.write(str(batch_num) + ":" + str(
                    avg_loss) + '\n')

            val_avg_loss_list.append(avg_loss)
            avg_loss = 0.0
            
    loss_per_observation_validation = total_loss / num_observations
    wandb.log({'val_loss_per_observation': loss_per_observation_validation,
               'val_num_observations': num_observations})
    return total_loss / num_batches


def train_model(model, epoch):
    """
    This function is used to train the model in batches
    :param model: Model to be trained
    :param epoch: Current epoch
    :return:
    """

    avg_loss = 0.0
    total_loss = 0.0
    num_observations = 0.0
    num_batches = len(train_loader)
    model.train()
    train_batch_num = 0.0

    for batch_num, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)

        loss = criterion(outputs, y.view(y.shape[0], 1))
        loss.backward()

        train_batch_num += 1
        wandb.log({"batch_train_loss": loss,
                   'train_batch_num': train_batch_num})

        # If the output starts giving nan at any point, stop training
        if torch.isnan(outputs).any().item() == True:
            torch.save({
                "x": x
            }, "/home/ubuntu/proj/run/{}/bad_input_{}.txt".format(runtime, epoch))
            # with open("/home/ubuntu/proj/run/{}/bad_input_{}.txt".format(runtime, epoch), "a") as fin:
            #     fin.write(x)
            break

        optimizer.step()

        avg_loss += loss.item()

        num_observations += len(x)
        total_loss += loss.item()

        if batch_num % 10 == 9:
            print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.8f}'.format(epoch, batch_num + 1, avg_loss / 10))
            train_avg_loss_list.append(avg_loss)
            with open("/home/ubuntu/proj/run/{}/avg_train_loss.txt".format(runtime), "a") as fin:
                fin.write(str(batch_num) + ":" + str(
                    avg_loss) + '\n')
            avg_loss = 0.0

    loss_per_observation_training = total_loss / num_observations
    wandb.log({'train_loss_per_observation': loss_per_observation_training,
               'train_num_observations': num_observations})
    print('train loss decrease per observation: {}'.format(total_loss / num_observations))
    return total_loss / num_batches


class LinearRegression(torch.nn.Module):
    """
    This class represents the Neural network to be trained
    """

    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.layers = torch.nn.Sequential(*
                                          [torch.nn.Linear(input_size, input_size), torch.nn.BatchNorm1d(input_size),
                                           torch.nn.ReLU(), torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(input_size, input_size), torch.nn.BatchNorm1d(input_size),
                                           torch.nn.ReLU(), torch.nn.Dropout(p=0.2),
                                           torch.nn.Linear(input_size, 10000), torch.nn.BatchNorm1d(10000),
                                           torch.nn.Linear(10000, 1)])

        '''
        [torch.nn.Linear(input_size, 2048), torch.nn.BatchNorm1d(2048),
         torch.nn.ReLU(),
         torch.nn.Linear(2048, 2048), torch.nn.BatchNorm1d(2048), torch.nn.ReLU(),
         torch.nn.Linear(2048, 1024), torch.nn.BatchNorm1d(1024),
         torch.nn.Dropout(p=0.3), torch.nn.ReLU(),
         torch.nn.Linear(1024, 1024), torch.nn.BatchNorm1d(1024), torch.nn.ReLU(),
         torch.nn.Linear(1024, 1024), torch.nn.BatchNorm1d(1024),
         # torch.nn.Dropout(p=0.1), torch.nn.ReLU(),
         torch.nn.Linear(1024, 512), torch.nn.BatchNorm1d(512),
         # torch.nn.Dropout(p=0.05), torch.nn.ReLU(),
         torch.nn.Linear(512, 256), torch.nn.BatchNorm1d(256), torch.nn.ReLU(),
         torch.nn.Linear(256, 128), torch.nn.Linear(128, 1),
         torch.nn.Flatten()]
        '''

    def forward(self, x):
        # return self.layers(x)
        return self.linear(x)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)


def drop_columns(encoded_df):
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
    runtime = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
    os.mkdir('/home/ubuntu/proj/run/' + runtime)
    a = time.time()
    ##################################################################################################
    # Initialize Wandb
    ##################################################################################################
    wandb.init(project="11785-project", entity="shubhang")

    data_filepath = '/home/ubuntu/proj/psam_pusa/'
    filenames = parse_data(data_filepath)
    print('filenames len is: ' + str(len(filenames)))

    # Combine the data
    combined_data = np.array([np.load(fname) for fname in filenames])
    print('Combined data successfully')

    # Update nans in the data to 0
    where_are_NaNs = np.isnan(combined_data)
    combined_data[where_are_NaNs] = 0

    # Read the columns for the dataset
    colnames_file = pd.read_csv('/home/ubuntu/proj/psam_pusa_colnames.csv')
    colnames = list(colnames_file['0'])

    # Load the dataframe using np array and columns
    encoded_df = pd.DataFrame(combined_data, columns=colnames)

    # Drop columns which are not needed for the research
    encoded_df, len_remaining_cols, remaining_cols = drop_columns(encoded_df)

    encoded_df_memory = encoded_df.memory_usage(deep=True).sum()
    print('*' * 150)
    print(encoded_df_memory)
    print('*' * 150)
    wandb.log({'encoded_df_memory': encoded_df_memory})

    # # Split test cases before doing distillation to get real accuracy
    # train_target = torch.tensor(target_df['LOG_PINCP'].values.astype(np.float32))
    # train = torch.tensor(encoded_df.drop(columns=['PINCP']).values.astype(np.float32))
    # _, _, _, _ = train_test_split(train, train_target, test_size=0.1, random_state=1)

    # calculate variance decomposition matrix of the original data
    print(encoded_df.describe())

    # Replace encoded with new samples
    encoded_df_2 = encoded_df.__deepcopy__()

    # Create a new dataframe to store the target values
    target_df_2 = encoded_df_2.filter(['PINCP'], axis=1)
    print(target_df_2)

    # Update target values to 1 if they are 0, as we will be taking a log of the income
    target_df_2[target_df_2['PINCP'] < 1] = 1
    target_df_2['LOG_PINCP'] = np.log(target_df_2['PINCP'])

    # Drop the source column from both dataframes
    target_df_2 = target_df_2.drop(columns=['PINCP'])
    encoded_df_2 = encoded_df_2.drop(columns=['PINCP'])

    # Normalize columns
    # encoded_df = encoded_df.apply(lambda x: (x - x.mean()) / (1 + x.std()), axis=0)
    encoded_df_2_memory = encoded_df_2.memory_usage(deep=True).sum()
    print('*' * 150)
    print(encoded_df_2_memory)
    print('*' * 150)
    wandb.log({'encoded_df_2_memory': encoded_df_2_memory})
    """# Hyperparameters"""

    input_dims = encoded_df_2.shape[1]
    output_dims = 1
    learning_rate = 1e-3
    epochs = 50
    batch_size = 128

    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size
    }

    """# Load the GPU"""

    # Find the GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # !nvidia-smi

    """# Create model object"""

    model = LinearRegression(input_dims, output_dims)
    model.apply(LinearRegression.init_weights)
    model = model.to(device)
    print(model)
    """# Train the model"""

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True,
                                                           threshold=0.1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7, verbose=True)

    train_size = int(0.7 * len(encoded_df))
    validation_size = int(0.2 * len(encoded_df))

    print(target_df_2.describe())

    # Split to get the train data. Ignore the test data as we will be taking it from a fresh dataframe at the end
    # This will help avoid getting incorrect R2 scores on sampled data
    train_target_2 = torch.tensor(target_df_2['LOG_PINCP'].values.astype(np.float32))
    train_2 = torch.tensor(encoded_df_2.values.astype(np.float32))
    Xtrain_inter, _, ytrain_inter, _ = train_test_split(train_2, train_target_2, test_size=0.1, random_state=1)

    train_avg_loss_list = []  # initializing average training loss list

    # Now split train and validation data
    train_2, val, train_target_2, val_target = train_test_split(Xtrain_inter, ytrain_inter, test_size=0.22,
                                                                random_state=1)
    train_tensor = data_utils.TensorDataset(train_2, train_target_2)
    train_loader = data_utils.DataLoader(dataset=train_tensor, batch_size=batch_size, shuffle=True)

    val_tensor = data_utils.TensorDataset(val, val_target)
    val_loader = data_utils.DataLoader(dataset=val_tensor, batch_size=batch_size, shuffle=True)
    val_avg_loss_list = []  # initializing average validation loss list

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(val_loader),
    #                                                 epochs=epochs)

    ##################################################################################################
    # Train the model
    ##################################################################################################
    os.mkdir('/home/ubuntu/proj/run/' + runtime + '/models')
    wandb.define_metric('epoch')
    wandb.define_metric('total_interval')
    wandb.define_metric('val_loss_decrease_per_epoch', step_metric='epoch')
    wandb.define_metric('train_loss_decrease_per_epoch', step_metric='epoch')
    wandb.define_metric('val_loss_decrease_per_time', step_metric='total_interval')
    wandb.define_metric('train_loss_decrease_per_time', step_metric='total_interval')
    best_loss = math.inf
    train_loss_decrease_per_epoch = 0.0
    val_loss_epochs = 0.0
    train_loss_epochs = 0.0
    total_interval = 0.0
    for epoch in range(epochs):
        start = time.process_time()

        print('=' * 100)
        train_loss = train_model(model, epoch)
        train_loss_epochs += train_loss
        print('train loss: {}'.format(train_loss))
        val_loss = validate_model(model, epoch)
        print('val loss: {}'.format(val_loss))
        val_loss_epochs += val_loss
        if epoch > 20:
            scheduler.step(val_loss)
            # scheduler.step()

        if val_loss < best_loss:
            PATH = '/home/ubuntu/proj/run/{}/models/'.format(runtime) + 'model.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, PATH)
        interval = time.process_time() - start
        print('Time taken for Epoch: {} is {}'.format(epoch, interval))
        print('=' * 100)
        total_interval += interval
        val_loss_decrease_per_epoch = val_loss_epochs / (epoch + 1)
        val_loss_decrease_per_time = val_loss_epochs / total_interval
        train_loss_decrease_per_epoch = train_loss_epochs / (epoch + 1)
        train_loss_decrease_per_time = train_loss_epochs / total_interval
        wandb.log({'val_loss_decrease_per_epoch': val_loss_decrease_per_epoch,
                   'val_loss_decrease_per_time': val_loss_decrease_per_time,
                   'train_loss_decrease_per_epoch': train_loss_decrease_per_epoch,
                   'train_loss_decrease_per_time': train_loss_decrease_per_time,
                   'epoch': epoch,
                   'total_interval': total_interval})
        # print('val loss decrease per epoch {}'.format(val_loss_decrease_per_epoch))
        # print('val loss decrease per time {}'.format(val_loss_decrease_per_time))
        # print('train loss decrease per epoch {}'.format(train_loss_decrease_per_epoch))
        # print('train loss decrease per time {}'.format(train_loss_decrease_per_time))
    ##################################################################################################
    # Test the model on test dataset and get R2
    ##################################################################################################
    test_df = pd.DataFrame(combined_data, columns=colnames)
    test_df, len_remaining_cols, remaining_cols = drop_columns(test_df)

    # test_df[test_df['PINCP'] < 1]['PINCP'] = 1
    test_target_df = test_df.filter(['PINCP'], axis=1)
    test_target_df[test_target_df['PINCP'] < 1] = 1
    test_target_df['LOG_PINCP'] = np.log(test_target_df['PINCP'])
    test_target_df = test_target_df.drop(columns=['PINCP'])
    test_df = test_df.drop(columns=['PINCP'])

    test_labels = torch.tensor(test_target_df['LOG_PINCP'].values.astype(np.float32))
    test_input = torch.tensor(test_df.values.astype(np.float32))

    test, _, test_target, _ = train_test_split(test_input, test_labels, test_size=0.9, random_state=1)
    # test_tensor = data_utils.TensorDataset(test, test_target)
    # test_loader = data_utils.DataLoader(dataset=test_tensor, batch_size=batch_size, shuffle=True)

    predictions = test_model(model)
    b = time.time()
    print('Total time {}'.format(b - a))
