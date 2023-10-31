from .custom_dataset import Custom_datasets
from torch.utils.data import DataLoader
import json
import os 
import glob

def make_datasets(data_path: str, data_name: str, mode: str):
    
    # train mode
    if mode == 'train':
        data_path_lst = glob.glob(os.path.join(data_path,data_name,'*'))
        trainset = Custom_datasets(
        file_path_lst = data_path_lst,
        mode = 'train'
        )
        return trainset

    if mode == 'df_real':
        data_path = '/media/data1/FaceForensics_Dec2020/FaceForensics++/face_data/data_c23/original_sequences/youtube/c23/faces'
        video_lst = sorted(glob.glob(os.path.join(data_path,'*')))
        data_path_lst = []
        for v in video_lst:
            v_len = len(glob.glob(os.path.join(v,'*')))
            stride = int(v_len/100)
            for i, f in enumerate(glob.glob(os.path.join(v,'*'))):
                if v_len % stride == 0:
                    data_path_lst.append(f)

        trainset = Custom_datasets(
        file_path_lst = data_path_lst,
        mode = 'train'
        )
        return trainset

    if mode == 'test':
        data_path = '/home/inho/df_detection/experiments_xception/data/'

        testset = load_data_paths_lst(data_path=data_path, data_name=data_name, mode=mode)
        testset = Custom_datasets(
        file_path_lst = testset,
        mode = 'test'
        )
        return testset

def create_dataloader(dataset, batch_size: int = 16, shuffle: bool = False):

    return DataLoader(
        dataset     = dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = 16
    )

####### DF dataset TEST를 위한 함수
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