# Define your model
import torch

from cam_tools import *
from d2l import *
from dp.model import *

#加载模型
model_path = r"D:\all_code\cam_code\model\5.pt"
model = ResNet(in_channels=6, classes=3)
model.load_state_dict(torch.load(model_path))
model.eval()
# Set your CAM extractor
from torchcam.methods import SmoothGradCAMpp
cam_extractor = SmoothGradCAMpp(model)

from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp


from torchvision.io.image import read_image
IMU_data=get_test_data(r"D:\all_code\cam_code\data\test1")
IMU_data=IMU_data[:,:,:2024]
# img = Image.open("data/example.png").convert('RGB')
input_tensor=torch.Tensor(IMU_data)
print(IMU_data.shape)


# 实例化 SmoothGradCAMpp 类
cam_extractor = SmoothGradCAMpp(model)

# Preprocess your data and feed it to the model
out = model(input_tensor.unsqueeze(0))
a=out.squeeze(0).argmax()
# Retrieve the CAM by passing the class index and the model output
activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
import matplotlib.pyplot as plt
plt.imshow(activation_map[0].squeeze(0).numpy()); plt.axis('off'); plt.tight_layout(); plt.show()
