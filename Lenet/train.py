import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 有顺序
transforms = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ]
)

trainpath = r"../frames/target/train"  # 路径
testpath = r"../frames/target/test"  # 路径

trainloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(trainpath, transform=transforms),
    batch_size=36,
    shuffle=True,
    num_workers=0
)

testloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(testpath, transform=transforms),
    batch_size=36,
    shuffle=False,
    num_workers=0
)

test_data_iter = iter(testloader)
test_image, test_label = test_data_iter.next()

classes = ('front', 'reverse', 'others')

def imshow(img):
    # 标准化 y=(x-0.5)/0.5   反标准化 y*0.5 +0.5
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# imshow(torchvision.utils.make_grid(test_image))
# print(" ".join('%5s ' % classes[test_label[j]] for j in range(4)))

net = LeNet().cuda()
loss_function = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(net.parameters(), lr=0.002)

for epoch in range(30):
    running_loss = 0.0
    # print("Epoch:", epoch)
    for step, data in enumerate(trainloader, start=0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # print(inputs.shape, labels.shape)
        inputs, labels = inputs.cuda(), labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        # print("Step:", step, "Loss:", loss)
        loss.backward()
        optimizer.step()    # 参数更新
        running_loss += loss.item()
        if step % 10 == 9:
            # 不去计算梯度损失
            with torch.no_grad():
                outputs = net(test_image.cuda())
                predict_y = torch.max(outputs, dim=1)[1]
                predict_y = predict_y.cpu()
                accuracy = (predict_y == test_label).sum().item() / test_label.size(0)

                print('[%d %5d] tran_loss: %.3f test_accuracy: %.3f' %
                      (epoch+1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0

print("Finish Training")
save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)