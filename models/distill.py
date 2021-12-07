from util import variance_decomposition


def sample_data(encoded_df, remaining_cols, config_data):
    '''
    This method is used to perform distillation on the dataset.
    First the condition index and the variance decomposition matrix are computed for the input dataframe.
    Using those computations, N most difficult to learn features and their corresponding M correlated features are extracted.
    Then stratified sampling techniques are applied to increase the proportions of data for the extracted features
    '''

    # calculate variance decomposition matrix of the original data
    print('Starting variance decomposition')
    variance_decomposition_matrix = variance_decomposition.collinear(encoded_df.drop(columns=['PINCP']).to_numpy(),
                                                                     remaining_cols.remove('PINCP'))
    print('Variance decomposition finished')

    # variance_decomposition_matrix = np.cov(encoded_df.to_numpy().T)
    print(variance_decomposition_matrix.shape)
    # variance_decomposition_matrix = pd.DataFrame.cov(encoded_df).to_numpy()

    print(encoded_df.describe())

    n = config_data['sampling_params']['n']
    m = config_data['sampling_params']['m']
    min_rows_per_strata = config_data['sampling_params']['min_rows_per_strata']

    # Find the most difficult to learn features and the features that are highly correlated with them
    most_difficult_to_learn = variance_decomposition.find_difficult_to_learn(variance_decomposition_matrix, n=n, m=m)

    # Get the names for the columns using their indices
    colname = encoded_df.columns[most_difficult_to_learn]
    print(list(colname))

    # Group the data by the difficult to learn columns and sample from the new dataset. This creates stratified sampling
    # For each group, min rows sampled is specified as 50 or else the len of the group
    encoded_df = encoded_df.groupby(list(colname), group_keys=False).apply(
        lambda x: x.sample(min(len(x), min_rows_per_strata)))

    print(encoded_df.describe())
    return encoded_df.__deepcopy__()
