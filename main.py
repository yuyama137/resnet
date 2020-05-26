import PIL

PIL.PILLOW_VERSION = PIL.__version__# これをやらないと、torchvisionでエラーを吐く。。。

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from model import *

import os

"""
test with cifar10(10 class).
original paper was tested with imagenet2012 wich is classification over 1000 classes.

ref : https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
"""

class args():
    lr = 0.1# learning rate
    maxepoch = 10
    batchnum = 8

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # 前処理(ネット上のをそのまま利用した)
    transform_train = transforms.Compose([# 学習データの前処理
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([# テストデータの前処理
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=1)# バッチサイズをcpuテスト用に極端に小さくしてある。

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = ResNet_18(10)
    net = net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)# パラメータはネットのもの

    # train
    for epoch in range(args.maxepoch):
        print("epoch : {}".format(epoch))
        net.train()
        total = 0
        correct = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            out = net(inputs)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

            # target -> onehot
            class_num = 10
            targets_onehot = torch.eye(class_num)[targets]

            _, predict = out.max(0)
            #import pdb; pdb.set_trace()
            total += targets.size(0)
            correct += predict.eq(targets_onehot).sum().item()

            # acc = correct/total
            # print("epochnum : {}, accuracy : {}".format(epoch,acc))

            if batch_idx%100 == 0:
                print("batch_idx : {}, loss : {}".format(batch_idx, loss))
        

    acc = correct/total
    print("epochnum : {}, accuracy : {}".format(epoch,acc))


if __name__=="__main__":
    args = args()
    main(args)