# Define your model
import matplotlib.pyplot as plt
import pandas as pd
import torch

from cam_tools import *
from d2l import *
from dp.model import *

#加载模型
model_path = r"D:\all_code\cam_code\model\3.pt"
model = ResNet(in_channels=6, classes=3)
model.load_state_dict(torch.load(model_path))
model.eval()


#加载数据
IMU_data=get_test_data(r"D:\all_code\cam_code\data\test1","1x_0")
print(IMU_data.shape)
start=3
IMU_data=IMU_data[:,int(start*500):int(start*500+1024)]
input_tensor=torch.Tensor(IMU_data)

#获取最后一个全连接层的参数

    # 获取模型的所有参数
params = model.parameters()
    # 将参数列表转换为一个可迭代的列表
param_list = list(params)
    # 获取倒数第二层的参数
last_layer_params = param_list[-2]
print(last_layer_params.shape)

#获取网路最后一个卷积层的输出
    # 定义一个列表来保存最后一层卷积层的输出结果
last_conv_output = []
    # 定义一个钩子函数来获取最后一层卷积层的输出结果
def get_last_conv_output(module, input, output):
    last_conv_output.append(output)
    # 获取最后一层卷积层
last_conv_layer=model.features[-2].layer[6]
    # 将钩子函数附加到最后一层卷积层上
hook = last_conv_layer.register_forward_hook(get_last_conv_output)

#获取cam
out = model(input_tensor.unsqueeze(0))
print(out)
print(last_conv_output[-1].shape)
result=get_cam_1D(last_layer_params,last_conv_output[-1],out)
print(result.shape)
print(result)
print(type(result))
plt.plot(IMU_data[0])
plt.show()

import seaborn as sns
result0=pd.DataFrame(result[0].reshape(1,-1))
print(result0.shape)
sns.heatmap(result0, cmap='coolwarm',cbar=False)
plt.show()
# 绘制第二个子图
result1=pd.DataFrame(result[1].reshape(1,-1))
sns.heatmap(result1, cmap='coolwarm',cbar=False)
plt.show()
# 绘制第二个子图
result2=pd.DataFrame(result[2].reshape(1,-1))
sns.heatmap(result2, cmap='coolwarm',cbar=False)
plt.show()
result=pd.DataFrame(result)
sns.heatmap(result, cmap='coolwarm',cbar=False)
plt.show()

print(result0,result1,result0+result1)
