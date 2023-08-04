import torch
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from cam_tools import *
#加载模型
model = resnet18(pretrained=True).eval()

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
last_conv_layer = model.layer4[-1].conv2


# 将钩子函数附加到最后一层卷积层上
hook = last_conv_layer.register_forward_hook(get_last_conv_output)

# 加载输入
from torchvision.io.image import read_image
img = read_image("../data/image/example.png")
img=img[:3,:,:]
input_tensor = normalize(resize(img, [224, 224]) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

out = model(input_tensor.unsqueeze(0))
print(out.shape)
print(last_conv_output[0].shape)
result=get_cam_2D(last_layer_params,last_conv_output[-1],out)
print(result.shape)

plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask

# Resize the CAM and overlay it
result = overlay_mask(to_pil_image(img), to_pil_image(torch.Tensor(result), mode='F'), alpha=0.5)
# Display it
plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()