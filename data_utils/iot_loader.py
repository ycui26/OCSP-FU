import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import warnings

# Suppress warnings for clean console output
warnings.filterwarnings('ignore')

def allocate_dataset(args):
    """
    Load CICIoT tabular datasets, perform stratified subsampling, and 
    allocate them to clients in an IID (label-balanced) manner.
    
    Args:
        args: Argument parser object (contains num_users, seed, etc.)
        
    Returns:
        train_set: Pandas DataFrame containing the training data.
        test_set: Pandas DataFrame containing the testing data.
        dict_users: Dictionary mapping client_idx to their allocated positional indices.
    """
   
    
    df_train = pd.read_csv('./processed_train.csv')
    df_test = pd.read_csv('./processed_test.csv')
    
    # Use args.seed for reproducibility, fallback to 42
    seed = 42
    
    # ==========================================
    # 1. Stratified Subsampling
    # ==========================================
    max_train_samples = 10000
    max_test_samples = 2000
    
    # Chain operations cleanly: Shuffle -> Group by Label -> Take top K
    train_set = df_train.sample(frac=1, random_state=seed).groupby('label').head(max_train_samples)
    test_set = df_test.sample(frac=1, random_state=seed).groupby('label').head(max_test_samples)
    
    train_set = train_set.reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)
    
    # ==========================================
    # 2. Data Allocation
    # ==========================================
    
    skf = StratifiedKFold(n_splits=args.num_users, shuffle=True, random_state=seed)
    dict_users = {}
    
    # skf.split yields (train_idx, test_idx) for each fold. 
    # We use 'test_idx' as the exclusive chunk of data allocated to a specific client.
    for client_idx, (_, chunk_indices) in enumerate(skf.split(train_set, train_set['label'])):
        dict_users[client_idx] = chunk_indices
        

    return train_set, test_set, dict_users