from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = ImageFolder(root='D:\\test', transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)

    classes = ('cocacola_tin', 'icetea_lemon', 'icetea_peach' \
                   , 'nescafe_tin', 'nesfit', 'pepsi', 'pepsi_max' \
                   , 'pepsi_twist', 'redbull', 'redbull_sugar_free' \
                   , 'sprite', 'tadelle', 'tropicana_apricot' \
                   , 'tropicana_mixed', 'zuber')

    import matplotlib.pyplot as plt
    import numpy as np
    import torchvision


    # functions to show an image

    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # get some random training images
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    print(labels)
    print(' '.join('%5s' % classes[labels[j]] for j in range(15)))

    '''
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            # 3 input image channel, 15 output channel, 5x5 square convolution
            # kernel
            self.conv1 = nn.Conv2d(3, 15, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(15, 16, 5)
            self.fc1 = nn.Linear(13456, 128)
            self.fc2 = nn.Linear(128, 84)
            self.fc3 = nn.Linear(84, 15)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), 13456)

            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Net()
    net.to(device)

    net.load_state_dict(torch.load("model/resnet101.pt"))
    net.eval()

    outputs = net(images.to(device))

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(5)))
    '''

    device = torch.device("cuda")

    model = torchvision.models.alexnet()
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(
            in_features=2048,
            out_features=15
        ),
        torch.nn.Sigmoid()
    )
    model.load_state_dict(torch.load("model/alexnet.pth"))
    model.eval()
    model = model.cuda(device=device)

    outputs = model(images.to(device))

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(15)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on test images: %d %%' % (
            100 * correct / total))

    class_correct = list(0. for i in range(15))
    class_total = list(0. for i in range(15))

    with torch.no_grad():
        for data in testloader:

            images, labels = data[0].to(device), data[1].to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(15):
                label = labels[i]

                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(15):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))