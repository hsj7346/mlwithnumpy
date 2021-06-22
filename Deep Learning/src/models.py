import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class Digit_Classifier(nn.Module):
    def __init__(self):
        super(Digit_Classifier, self).__init__()
        self.f_h = nn.Linear(784,128)
        self.s_h = nn.Linear(128,64)
        self.output = nn.Linear(64,10)


    def forward(self, inputs):
        result = F.relu(self.f_h(inputs))
        result = F.relu(self.s_h(result))
        return self.output(result)



class Dog_Classifier_FC(nn.Module):

    def __init__(self):
        super(Dog_Classifier_FC, self).__init__()
        self.f_h = nn.Linear(12288,128)
        self.s_h = nn.Linear(128,64)
        self.output = nn.Linear(64,10)
        

    def forward(self, inputs):
        inputs = inputs.flatten(start_dim=1)
        result = F.relu(self.f_h(inputs))
        result = F.relu(self.s_h(result))
        return self.output(result)
        

class Dog_Classifier_Conv(nn.Module):

    def __init__(self, kernel_size, stride):
        super(Dog_Classifier_Conv, self).__init__()
        self.f_h = nn.Conv2d(3,16,kernel_size = kernel_size[0], stride = stride[0])
        self.s_h = nn.Conv2d(16,32,kernel_size = kernel_size[1], stride = stride[1])
        self.output = nn.Linear(5408,10)
        
        

    def forward(self, inputs):
        result = inputs.permute((0, 3, 1, 2))
        result = F.relu(self.f_h(result))
        result = F.max_pool2d(result,2)
        result = F.relu(self.s_h(result))
        result = F.max_pool2d(result,2)
        result = result.flatten(start_dim=1)
        return self.output(result)

class Large_Dog_Classifier(nn.Module):

    def __init__(self):
        super(Large_Dog_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(4, 6, kernel_size=(3,3), padding=1)
        self.conv3 = nn.Conv2d(6, 8, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(8, 10, kernel_size=(3, 3))
        self.conv5 = nn.Conv2d(10, 12, kernel_size=(3, 3))
        self.conv6 = nn.Conv2d(12, 14, kernel_size=(3, 3))
        self.conv7 = nn.Conv2d(14, 16, kernel_size=(3, 3), stride=(2,2))
        self.fc1 = nn.Linear(11664, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, input):
        input = input.permute((0, 3, 1, 2))
        input = F.relu(self.conv1(input))
        input = F.relu(self.conv2(input))
        input = F.relu(self.conv3(input))
        input = F.relu(self.conv4(input))
        input = F.relu(self.conv5(input))
        input = F.relu(self.conv6(input))
        input = F.relu(self.conv7(input))
        input = F.relu(self.fc1(input.view(-1, 11664)))
        input = self.fc2(input)

        return input
