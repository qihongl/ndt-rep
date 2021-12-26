'''
To re-used the hyperparameters (Config, Lorenz_CFG, DATA_MODES) and the data loading function from [1]. 

References: 

- [1] Ye, J., & Pandarinath, C. (2021). Representation learning for neural population activity with Neural Data Transformers. In arXiv [q-bio.NC]. arXiv. http://arxiv.org/abs/2108.01210
    - Code: https://github.com/snel-repo/neural-data-transformers  
'''


import h5py
import numpy as np

LOG_EPSILON = 1e-7

# @dataclass
class Config:
    MAX_LEN = 50
    BATCH_SIZE = 1
    LR = 0.001
    EMBED_DIM = 32
    NUM_HEAD = 2  # used in bert model
    FF_DIM = 16  # used in bert model
    NUM_LAYERS = 1
    DROPOUT_RATE = .5 
    EPSILON = LOG_EPSILON


class LORENZ_CFG: 
    EMBED_DIM = 2 # We embed to 2 here so transformer can use 2 heads. Perf diff is minimal.
    LOGRATE = True
    MASK_MAX_SPAN = 1
    USE_ZERO_MASK = True
    MASK_MODE = "timestep" # ["full", "timestep"]
    MASK_RATIO = 0.25    
    MASK_TOKEN_RATIO = 1.0 # We don't need this if we use zero mask
    MASK_RANDOM_RATIO = 0.5 # Of the non-replaced, what percentage should be random?
    MAX_GRAD_NORM = 200.0
    
    NUM_UPDATES = 10000 # Max updates
    
class DATASET_MODES:
    train = "train"
    val = "val"
    test = "test"
    trainval = "trainval"
    

def get_data_from_h5(mode, filepath):
    r"""
        returns:
            spikes
            rates (None if not available)
            held out spikes (for cosmoothing, None if not available)
        * Note, rates and held out spikes codepaths conflict
    """

    with h5py.File(filepath, 'r') as h5file:
        h5dict = {key: h5file[key][()] for key in h5file.keys()}
        if 'eval_data_heldin' in h5dict: # NLB data
            get_key = lambda key: h5dict[key].astype(np.float32)
            train_data = get_key('train_data_heldin')
            train_data_fp = get_key('train_data_heldin_forward')
            train_data_heldout_fp = get_key('train_data_heldout_forward')
            train_data_all_fp = np.concatenate([train_data_fp, train_data_heldout_fp], -1)
            valid_data = get_key('eval_data_heldin')
            train_data_heldout = get_key('train_data_heldout')
            if 'eval_data_heldout' in h5dict:
                valid_data_heldout = get_key('eval_data_heldout')
            else:
                valid_data_heldout = np.zeros((valid_data.shape[0], valid_data.shape[1], train_data_heldout.shape[2]), dtype=np.float32)
            if 'eval_data_heldin_forward' in h5dict:
                valid_data_fp = get_key('eval_data_heldin_forward')
                valid_data_heldout_fp = get_key('eval_data_heldout_forward')
                valid_data_all_fp = np.concatenate([valid_data_fp, valid_data_heldout_fp], -1)
            else:
                valid_data_all_fp = np.zeros(
                    (valid_data.shape[0], train_data_fp.shape[1], valid_data.shape[2] + valid_data_heldout.shape[2]), dtype=np.float32
                )

            # NLB data does not have ground truth rates
            if mode == DATASET_MODES.train:
                return train_data, None, train_data_heldout, train_data_all_fp
            elif mode == DATASET_MODES.val:
                return valid_data, None, valid_data_heldout, valid_data_all_fp
        train_data = h5dict['train_data'].astype(np.float32).squeeze()
        valid_data = h5dict['valid_data'].astype(np.float32).squeeze()
        train_rates = None
        valid_rates = None
        if "train_truth" and "valid_truth" in h5dict: # original LFADS-type datasets
            has_rates = True
            train_rates = h5dict['train_truth'].astype(np.float32)
            valid_rates = h5dict['valid_truth'].astype(np.float32)
            train_rates = train_rates / h5dict['conversion_factor']
            valid_rates = valid_rates / h5dict['conversion_factor']
            train_rates = np.log(train_rates + LOG_EPSILON)
            valid_rates = np.log(valid_rates + LOG_EPSILON)
    if mode == DATASET_MODES.train:
        return train_data, train_rates, None, None
    elif mode == DATASET_MODES.val:
        return valid_data, valid_rates, None, None
    elif mode == DATASET_MODES.trainval:
        # merge training and validation data
        if 'train_inds' in h5dict and 'valid_inds' in h5dict:
            # if there are index labels, use them to reassemble full data
            train_inds = h5dict['train_inds'].squeeze()
            valid_inds = h5dict['valid_inds'].squeeze()
            file_data = merge_train_valid(
                train_data, valid_data, train_inds, valid_inds)
            if has_rates:
                merged_rates = merge_train_valid(
                    train_rates, valid_rates, train_inds, valid_inds
                )
        else:
            # if self.logger is not None:
            #     self.logger.info("No indices found for merge. "
            #     "Concatenating training and validation samples.")
            # file_data = np.concatenate([train_data, valid_data], axis=0)
            if has_rates:
                merged_rates = np.concatenate([train_rates, valid_rates], axis=0)
        return file_data, merged_rates if has_rates else None, None, None
    else: # test unsupported
        return None, None, None, None
