from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils,datasets
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
# Ignore warnings
import warnings
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = ImageFolder(root='D:\\train', transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    # =-========================22222222==========================================
    testset = ImageFolder(root='D:\\test', transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)

    classes = ('cocacola_tin', 'icetea_lemon', 'icetea_peach'\
                    ,'nescafe_tin', 'nesfit', 'pepsi', 'pepsi_max'\
                   ,'pepsi_twist', 'redbull', 'redbull_sugar_free'\
                   , 'sprite', 'tadelle', 'tropicana_apricot'\
                   , 'tropicana_mixed', 'zuber')

    def get_num_correct(preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    import torchvision
    import torch.nn as nn

    torch.cuda.current_device()

    model = torchvision.models.resnet50()
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(
            in_features=2048,
            out_features=15
        ),
        torch.nn.Sigmoid()
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize optimizer
    import torch.optim as optim

    model = model.cuda(device=device)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    writer = SummaryWriter()

    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    for epoch in range(25):  # loop over the dataset multiple times

        total_correct = 0
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]

            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            total_correct += get_num_correct(outputs, labels)

            writer.add_scalar('Loss', running_loss, epoch)
            writer.add_scalar('Number Correct', total_correct, epoch)
            writer.add_scalar('Accuracy', total_correct / len(trainset), epoch)
            '''
            for name, param in model.named_parameters():
                writer.add_histogram(name, param.clone(), epoch)
            '''
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
            running_loss = 0.0

    writer.close()
    print('Finished Training')

    torch.save(model.state_dict(), "model/resnet50.pth")