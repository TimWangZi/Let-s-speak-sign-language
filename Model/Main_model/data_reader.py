import json
import csv
import torch
from torch.utils.data import DataLoader, Dataset

DATA_LABEL_TRAIN = '.\\train_data\\label\\train.csv'
DATA_LABEL_TEST = '.\\train_data\\label\\test.csv'

DATA_TRAIN = '.\\train_data\\test\\'
DATA_TEST = '.\\train_data\\train\\'

class TrainDataLoader(DataLoader):
    def __init__(self, device:torch.device, label_file_name = DATA_LABEL_TRAIN ,folder_name = DATA_TRAIN):
        print('现在开始加载标签:')
        # 用迭代器加载CSV文件
        with open(label_file_name, 'r') as f_data:
            data_reader = csv.DictReader(f_data)
            self._data_ = list()
            for item in data_reader:
                self._data_.append((item['Number'] ,item['Translator'] ,item['Chinese']))
        print('标签加载完成')
        self.data_folder = folder_name
        self._device_ = device
    
    def __len__(self):
        return len(self._data_)
    
    def __getitem__(self ,idx):
        result = list()
        with open(self.data_folder + self._data_[idx][1] + self._data_[idx][0]) as f:
            json_object = json.loads(f.read())
            # 遍历每一帧
            for frame in json_object:
                frame_data = list()
                # 分别遍历每一帧下的字典数据
                for key,value in frame['face'].items():
                    frame_data.append(value)
                for key,value in frame['upper_body'].items():
                    frame_data.append(value)
                for key,value in frame['hands'].items():
                    frame_data.append(value)
                result.append(frame_data)
        result = torch.Tensor(result).to(self._device_)
        return (self._data_[idx][2] ,result)




