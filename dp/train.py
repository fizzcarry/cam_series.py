import os

from cam_config import *
from cam_tools import *
from d2l import *
from model import *
lr, num_epochs, batch_size = 0.00005, 3, 20
X_all,y_all=get_spilt_data()
x_train, x_test, y_train, y_test = train_test_split(X_all,y_all,test_size=0.25)
train_dataset=MyDataset(x_train,y_train)
test_dataset=MyDataset(x_test ,y_test)
train_iter = data.DataLoader(train_dataset, batch_size, shuffle=True,
                            num_workers=d2l.get_dataloader_workers())
test_iter = data.DataLoader(test_dataset, batch_size, shuffle=True,
                            num_workers=d2l.get_dataloader_workers())

print(x_train.shape)
print(y_train)

net=ResNet(in_channels=6,classes=3)

d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

#模型保存
model_path=r"D:\all_code\cam_code\model"
model_path = os.path.join(model_path , str(num_epochs) + ".pt")

torch.save(net.state_dict(), model_path)


# model.load_state_dict(torch.load(PATH))
# model.eval()

plt.show()