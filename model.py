import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models
from torchvision.transforms.functional import to_pil_image
from torchinfo import summary

# Device configuration
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "CPU")
print("Using torch %s %s"
    % (torch.__version__, torch.cuda.get_device_properties(0)
        if cuda else "CPU"))

# 데이터 전처리
'''
0.5는 임의의 값이므로, 데이터셋의 최적 평균, 표준편차를 구하면 더 좋은 결과가 나올 수 있음
'''
trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# root 경로 내에 있는 이미지 데이터를 tensor 형태로 변환함
train_set = datasets.ImageFolder(root="Datasets/", transform=trans)

print("18번째 데이터 : ")
print(train_set.__getitem__(18)) # 18번째 데이터

# train_set 데이터 개수
print("train_set data : {}".format(len(train_set)))

# Classes of trainSet
classes = train_set.classes
print("train_set classes : {}".format(classes))

# DataLoader
batch_size = 81 # batch_size
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)
print("train_loader data : {}".format(len(train_loader)))

# Iteration
dataiter = iter(train_loader)
images, labels = dataiter.next()
print("train_loader data label : {}".format(labels))
print("total of train_loader data : ")
print(images)
print("train_loader data = torch.Size([batch size, channel size, width of img, height of img])")
print("train_loader data = {}".format(images.shape))

# ResBlock
class ResBlock(nn.Module): # BottleNeck Structure
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels * ResBlock.expansion,
                               kernel_size=1, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * ResBlock.expansion)
        self.shortcut = nn.Sequential()

        # Projection mapping
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels,
                                                    out_channels * ResBlock.expansion,
                                                    kernel_size=1,
                                                    stride=stride,
                                                    bias=False),
                            nn.BatchNorm2d(out_channels * ResBlock.expansion)
            )

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        output = output + self.shortcut(output)
        output = F.relu(output)
        return output

# LFSRNet
class LFSRNet(nn.Module):
    def __init__(self):
        super(LFSRNet, self).__init__()
        self.in_channels = 3
        self.layer1 = self._make_layer(3, 1, stride=1)
        self.conv4 = nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(243, 3, kernel_size=3, stride=1, padding=1, bias=False)

    def _make_layer(self, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResBlock(self.in_channels, channels, stride))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, input):
        # Feature extraction
        output_1 = self.layer1(input)
        print("output_1 :")
        print(output_1.shape)
        # print(output_1)

        # Feature fusion_1
        output_2 = F.relu(self.conv4(output_1))
        print("output_2 :")
        print(output_2.shape)
        # print(output_2)

        # Feature fusion_2
        output_2 = output_2.view(1, 243, 512, 512) # Reshape the tensor [81, 3, 512, 512] -> [1, 243, 512, 512]
        output_3 = F.relu(self.conv5(output_2))
        print("output_3 :")
        print(output_3.shape)
        # print(output_3)

# Train function
def train(model, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        # loss = F.cross_entropy(output, target)

# Model visualization
model = LFSRNet().to(device)
print(model)

print("< Training >")
epochs = 1
for epoch in range(epochs):
    train(model, train_loader)

'''
< Model Visualization >
Test code by using random tensor
'''
print("< Model Visualization >")
# x = torch.randn(81, 3, 512, 512).to(device)
# output = model(x)
# print("----------")
summary(model, (81, 3, 512, 512), device=device.type)

'''
< Data Visualization >
'''
def IMshow(img):
    img = img / 2 + 0.5  # unnormalize
    np_img = img.numpy()
    # plt.imshow(np_img)
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    print(np_img.shape)
    print((np.transpose(np_img, (1, 2, 0))).shape)

# print(images.shape)
# IMshow(torchvision.utils.make_grid(images))
# print(images.shape)
# print((torchvision.utils.make_grid(images)).shape)
# print("".join("%5s "%classes[labels[j]] for j in range(1)))
# # torchvision.utils.make_grid(images)
# # plt.imshow(images.numpy())
# plt.show()

'''
< TODO >
'''
# dataset class별로 train_set(imagefolder 이용), train_loader(dataloader 이용) 만듬

# Solved
# 각 class에서 81개의 tensor가 resblock을 통과