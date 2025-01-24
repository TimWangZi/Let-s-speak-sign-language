# 特征提取网络(1D卷积)
# 提取原始数据中的重要部分，减少处理的数据量
import torch
import torch.nn as nn
class FeatureExtConv(nn.Module):
    def __init__(self ,feature_dim:int):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(feature_dim ,128 ,3) ,nn.LeakyReLU())
        self.pool1 = nn.AdaptiveMaxPool1d(output_size=24)
        self.conv2 = nn.Sequential(nn.Conv1d(128 ,128 ,3) ,nn.LeakyReLU())
        self.pool2 = nn.AdaptiveMaxPool1d(output_size=11)
        self.conv3 = nn.Sequential(nn.Conv1d(128 ,128 ,3),nn.LeakyReLU())
    
    def forward(self ,x:torch.Tensor):
        x = self.conv1(x)
        self.pool1.output_size = x.shape[2] // 2
        x = self.pool1(x)
        x = self.conv2(x)
        self.pool2.output_size = x.shape[2] // 2
        x = self.pool2(x)
        x = self.conv3(x)
        return x
