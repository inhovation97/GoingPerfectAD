from .custom_dataset import Custom_datasets
from torch.utils.data import DataLoader
import json
import os 

def load_data_paths_lst(data_path, data_name: str, mode: str=None):
    data_dict_lst = []
    with open( os.path.join(data_path, f'{data_name}_dict.json'), 'r' ) as f:
        data_dict = json.load(f)
        data_dict_lst.append(data_dict)
    
    trainset, validset, testset = [], [], []
    data_kind = ['train', 'valid', 'test']
    
    for d_kind, dataset in zip(data_kind, [trainset, validset, testset]):
        for data_paths in data_dict[d_kind].values():
            dataset += data_paths

    if mode == 'test':
        return testset
    else: 
        return trainset, validset

def make_datasets(data_path, data_name: str, mode: str=None):
    
    # test mode
    if mode == 'test':
        testset = load_data_paths_lst(data_path, data_name, mode)
        
        testset = Custom_datasets(
        file_path_lst = testset,
        mode = 'test'
        )

        return testset
    
    # training mode
    else: 
        trainset, validset = load_data_paths_lst(data_path, data_name)

        trainset = Custom_datasets(
            file_path_lst = trainset,
            mode = 'train'
        )
        
        validset = Custom_datasets(
            file_path_lst = validset,
            mode = 'valid'
        )
        return trainset, validset

def create_dataloader(dataset, batch_size: int = 16, shuffle: bool = False):

    return DataLoader(
        dataset     = dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = 16
    )