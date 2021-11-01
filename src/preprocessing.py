import numpy as np

def divide_dataset_by_jet(data):
    """Divide the dataset in 4 to separate PRI_jetnum categorical values"""
    
    # Mask for 23rd column PRI_jet_num (categorical feature) 
    cols = list(range(0,24)) + list(range(25,32))
    
    data_0jet = np.copy(data[data[:,24] == '0'])
    data_0jet = np.asarray(data_0jet[:, cols], dtype=float)
    
    data_1jet = np.copy(data[data[:,24] == '1'])
    data_1jet = np.asarray(data_1jet[:, cols], dtype=float)
    
    data_2jet = np.copy(data[data[:,24] == '2'])
    data_2jet = np.asarray(data_2jet[:, cols], dtype=float)
    
    data_3jet = np.copy(data[data[:,24] == '3'])
    data_3jet = np.asarray(data_3jet[:, cols], dtype=float)
    
    return np.array([data_0jet, data_1jet, data_2jet, data_3jet], dtype=object)
    
    
def clean_features(test_data, train_data, undef):
    """ Apply a cleaning of the features """
    
    test_set = test_data
    train_set = train_data
    
    # Preprocessing of the four jet datasets
    for jet in range(4):
        
        # Remove columns full of undefined values (-999.0 in the datase)
        train_set[jet][train_set[jet] <= undef] = np.nan # replace undefined values -999 by NaN
        train_set[jet] = train_set[jet][:, ~np.all(np.isnan(train_set[jet]), axis=0)]
        test_set[jet][test_set[jet] <= undef] = np.nan # replace undefined values -999 by NaN
        test_set[jet] = test_set[jet][:, ~np.all(np.isnan(test_set[jet]), axis=0)]

        # Remove columns without standard deviation to remove column full of the same value
        train_id_pred = train_set[jet][:,0:2]
        train_features = train_set[jet][:,2:]
        train_features = train_features[:, np.nanstd(train_features, axis=0) != 0]
        train_set[jet] = np.concatenate((train_id_pred, train_features), axis=1)
        
        # Extract prediction column of test before because it has std=0, and re-insert it after at the begining of the table
        test_id_pred = test_set[jet][:,0:2]
        test_features = test_set[jet][:,2:]
        test_features = test_features[:, np.nanstd(test_features, axis=0) != 0]
        test_set[jet] = np.concatenate((test_id_pred, test_features), axis=1)
    
    return test_set, train_set


def replace_nan(data_jets):
    """ Replace remaining NaN values after cleaning by mean, median or zero"""
    
    data_mean = np.empty_like(data_jets)
    data_median = np.empty_like(data_jets)
    data_null = np.empty_like(data_jets)
    
    for jet in range(4):
    # Replace Remaining undefined values by Mean, median or zero
        data_mean[jet] = np.where(np.isnan(data_jets[jet]), np.nanmean(data_jets[jet], axis=0), data_jets[jet])
        data_median[jet] = np.where(np.isnan(data_jets[jet]), np.nanmedian(data_jets[jet], axis=0), data_jets[jet])
        data_null[jet] = np.where(np.isnan(data_jets[jet]), np.float64(0), data_jets[jet])
    
    return data_mean, data_median, data_null


def standardize(train_data_jets, test_data_jets):
    """ Standradize all datasets to get features with zero mean and standard deviation of 1"""
    
    nbr_jets = train_data_jets.shape[0]
    
    for jet in range(nbr_jets):
        # extract features for standardization
        train_data_features = train_data_jets[jet][:,2:] 
        test_data_features = test_data_jets[jet][:,2:] 
        # store train mean and std without considering nan values
        train_mean = np.nanmean(train_data_features, axis=0)
        train_std = np.nanstd(train_data_features, axis=0)
        # standardize train and test data with train mean and std
        train_data_features = (train_data_features - train_mean) / train_std
        test_data_features = (test_data_features - train_mean) / train_std
        # insert standardized features into original dataset with predictions
        train_data_jets[jet][:,2:] = train_data_features
        test_data_jets[jet][:,2:] = test_data_features
        
    return  train_data_jets, test_data_jets


def preprocessing(train_raw_, test_raw_):
    """ Preprocessing function applying all preprocessing steps to the datasets"""
    
    undef = np.float64(-999.0)
    pred_dict = {'s':'1','b':'0', '?':'-1'}
    # drop 1st column (Id) and also 1st row with column names ("[1:,") 
    train_raw = train_raw_[1:, :]
    test_raw = test_raw_[1:, :] 
    
    # Change s(signal) and b(background) for s:1 and b:0, and change '?' for -1
    train_raw[:,1] = np.vectorize(pred_dict.get)(train_raw[:,1].astype(str))
    test_raw[:,1] = np.vectorize(pred_dict.get)(test_raw[:,1].astype(str))
    
    # Divide the dataset in four according to PRI_jet_num feature and cast to float
    train_data_jets = divide_dataset_by_jet(train_raw)
    test_data_jets = divide_dataset_by_jet(test_raw)
    
    # Remove columns with nan values or with standard deviation of 0
    test_data_jets, train_data_jets = clean_features(test_data_jets, train_data_jets, undef)
    
    # Standardize train and test sets to have mean=0 and std=1
    train_data_jets, test_data_jets = standardize(train_data_jets, test_data_jets)
    
    # Replace remaining undefined values by mean, median or zero
    train_data_mean, train_data_median, train_data_null = replace_nan(train_data_jets)
    test_data_mean, test_data_median, test_data_null = replace_nan(test_data_jets)
    
    return train_data_mean, train_data_median, train_data_null, test_data_mean, test_data_median, test_data_null

    
    