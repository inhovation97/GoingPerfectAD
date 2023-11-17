from torch.utils.data import Dataset, DataLoader # 데이터 커스터마이징
from PIL import Image # PIL = Python Image Library
import cv2 # albumentation transform을 쓰려면 꼭 이 라이브러리를 이용
from typing import List, Tuple
import glob
import os
from torchvision import transforms
import torch

"""숙지 사항"""
########################### Datasets 클래스는 경로를 개별 이미지 경로의 직전상위의 상위 폴더로 받는다.
########################### 경로 내에서 real이나 original이라는 문자열이 있으면, real_label: 0 부여
class Custom_datasets(Dataset):

  def __init__(self, 
               file_path_lst: List[str], 
               mode = str,
               ):

    self.all_data = file_path_lst
    self.resize = (299, 299)
    self.mean = (0.5, 0.5, 0.5)
    self.std  = (0.5, 0.5, 0.5)
    self.mode = mode
        
    # 데이터 증강: aug -> to tensor -> normalize 순
    if self.mode == 'train':
        self.transform = transforms.Compose([
                                transforms.ToPILImage(), # transform 함수는 PIL이미지만 받기때문에 추가함.
                                transforms.RandomHorizontalFlip(p = 0.5), # p확률로 이미지 좌우반전
                                transforms.RandomVerticalFlip(p = 0.5), # p확률로 상하반전
                                transforms.ToTensor(), 
                                transforms.Normalize(mean=self.mean, std=self.std) # 텐서 후, normalize
                            ])
    else:
        self.transform = transforms.Compose([
                                transforms.ToPILImage(), # transform 함수는 PIL이미지만 받기때문에 추가함.
                                transforms.ToTensor(),
                                transforms.Normalize(mean=self.mean, std=self.std)
                            ])
            

  def __getitem__(self, index):

    if torch.is_tensor(index):        # 인덱스가 tensor 형태일 수 있으니 리스트 형태로 바꿔준다.
        index = index.tolist()

    data_path = self.all_data[index]
    #img = np.array(Image.open(data_path).convert("RGB")) # albumenatation transform을 쓰려면 cv2 라이브러리로 이미지를 읽어야 함
    image=cv2.imread(data_path)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR -> RGB 변환
    image=cv2.resize(image, dsize=(self.resize[1], self.resize[0]))

    # transform 적용
#     if self.transform is not None:    
    image = self.transform(image)

    # 이미지 이름을 활용해 label 부여
    if 'real' in data_path or 'original' in data_path: # real 이미지는 레이블을 1로 받는다.
        label = 0
    else :
        label = 1

    return image, label

  def __len__(self):
    length = len(self.all_data)
    return length
        