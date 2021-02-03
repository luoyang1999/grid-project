import torch
import os
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet

transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('front','reverse')

net = LeNet()
net.load_state_dict(torch.load('Lenet1.pth'))

front_path = "../frames/target3/val/front/"
reverse_path = "../frames/target3/test/reverse/"

front_file = os.listdir(front_path)
reverse_file = os.listdir(reverse_path)

for file in reverse_file:

    im = Image.open(reverse_path+file)
    print(type(im))
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = net(im)
        print(outputs)
        predict = torch.max(outputs, dim=1)[1].data.numpy()
    print(classes[int(predict)])


