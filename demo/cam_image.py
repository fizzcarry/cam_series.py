# Define your model
from torchvision.models import resnet18
from PIL import Image
model = resnet18(pretrained=True).eval()
import numpy as np
# Set your CAM extractor
from torchcam.methods import SmoothGradCAMpp
cam_extractor = SmoothGradCAMpp(model)

from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp

model = resnet18(pretrained=True).eval()
# Get your input
from torchvision.io.image import read_image
img = read_image("../data/image/2.png")
img=img[:3,:,:]
# img = Image.open("data/example.png").convert('RGB')

# Preprocess it for your chosen model
input_tensor = normalize(resize(img, [224, 224]) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# with SmoothGradCAMpp(model) as cam_extractor:
#   # Preprocess your data and feed it to the model
#   out = model(input_tensor.unsqueeze(0))
#   # Retrieve the CAM by passing the class index and the model output
#   activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

# 实例化 SmoothGradCAMpp 类
cam_extractor = SmoothGradCAMpp(model)

# Preprocess your data and feed it to the model
out = model(input_tensor.unsqueeze(0))
# 使用 np.argsort 得到数组中元素按降序排序后的索引
arr=(out.detach().numpy()).flatten()
sorted_indices = np.argsort(-arr)

# 获取前 10 个最大的数及其对应的索引
top_10_values = arr[sorted_indices[:10]]
top_10_indices = sorted_indices[:10]

# 打印结果
print("最大的10个数：", top_10_values)
print("对应的索引：", top_10_indices)
a=out.squeeze(0).argmax()
# Retrieve the CAM by passing the class index and the model output
activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
print(type(activation_map))
print(a)
print(activation_map[0].shape)
import matplotlib.pyplot as plt
plt.imshow(activation_map[0].squeeze(0).numpy()); plt.axis('off'); plt.tight_layout(); plt.show()
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask

# Resize the CAM and overlay it
result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
# Display it
plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()