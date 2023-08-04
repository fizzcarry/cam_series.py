import matplotlib.pyplot as plt
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
IMU_data=get_test_data(r"D:\all_code\cam_code\data\original\test1\1","2")
# print(IMU_data.shape)
# print(IMU_data[:,0:50])
# plt.plot(IMU_data[0,2000:2000+512])
# plt.show()
# plt.plot(IMU_data[1,2000:2000+512])
# plt.show()
# plt.plot(IMU_data[2,2000:2000+512])
# plt.show()
# plt.plot(IMU_data[3,2000:2000+512])
# plt.show()
# plt.plot(IMU_data[4,2000:2000+512])
# plt.show()
# plt.plot(IMU_data[5,2000:2000+512])
# plt.show()

print(IMU_data.shape)
for i in range(100):
    start=i
    IMU_data_temp=IMU_data[:,start*211:start*211+1024]
    input_tensor=torch.Tensor(IMU_data_temp)

    #获取预测结果
    out = model(input_tensor.unsqueeze(0))
    print(i)
    print(input_tensor.shape)
    print(torch.softmax(out,dim=1))
    print(out)