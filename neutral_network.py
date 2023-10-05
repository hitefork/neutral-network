# coding=gbk
'''import torch as t
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable


class Linear(nn.Module):
       def __int__(self,in_feature,out_feature):
              super(Linear,self).__init__()
              self.w=nn.Parameter(t.randn(in_feature,out_feature))
              self.b=nn.Parameter(t.randn(out_feature))
       def forward(self,x):
              x=x.mm(self.w)
              return x+self.b.expand_as(x)
layer=Linear(4,3)
input=Variable(t.randn(2,4))

#��֪��
class Perceptron(nn.Module):
    def __int__(self,in_features,hidden_features,out_features):
        nn.module.__init__(self)
        self.layer1=Linear(in_features,hidden_features)
        ##�˴���Linear��ǰ���Ѿ������
        self.layer2=Linear(hidden_features,out_features)
    def forward(self,x):
        x=self.layer1(x)
        x=t.sigmoid(x)
        return self.layer2(x)
    '''
    
#��򵥵�������Ԥ����д����

import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt
import random
import os
from PIL import Image,ImageOps
import cv2 as cv
import numpy as np
from os import path 











# ����ToTensor��Normalize��transform
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize((0.5,), (0.5,))

# ����Compose��transform
transform = transforms.Compose([
    to_tensor,  # ת��Ϊ����
    normalize  # ��׼��
])

# �������ݼ�
data_train = datasets.MNIST(root="..//data//",
                            transform=transform,
                            train=True,
                            download=True)

data_test = datasets.MNIST(root="..//data//",
                           transform=transform,
                           train=False,
                           download=True)
# װ������
data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=64,
                                                shuffle=True)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=64,
                                               shuffle=True)


class MLP(torch.nn.Module):

    def __init__(self, num_i, num_h, num_o):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(num_i, num_h)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h, num_h)  # 2������
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(num_h, num_o)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x


def train(model):
    # ��ʧ��������������������Ŀ���ǩ���бȽϣ�����������֮��Ĳ��졣��ѵ���ڼ䣬���ǳ�����С����ʧ��������ʹ������ǩ���ӽ�
    cost = torch.nn.CrossEntropyLoss()
    # �Ż�����һ��ʵ�������ڵ���ģ�Ͳ�������С����ʧ������
    # ʹ�÷��򴫲��㷨�����ݶȲ�����ģ�͵�Ȩ�ء����������ʹ��Adam�Ż������Ż�ģ�͡�model.parameters()�ṩ��Ҫ�Ż��Ĳ�����
    optimizer = torch.optim.Adam(model.parameters())
    # ���õ�������
    epochs = 20
    for epoch in range(epochs):
        sum_loss = 0
        train_correct = 0
        for data in data_loader_train:
            inputs, labels = data  # inputs ά�ȣ�[64,1,28,28]
            #     print(inputs.shape)
            inputs = torch.flatten(inputs, start_dim=1)  # չƽ���ݣ�ת��Ϊ[64,784]
            #     print(inputs.shape)
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = cost(outputs, labels)
            loss.backward()
            optimizer.step()

            _, id = torch.max(outputs.data, 1)
            sum_loss += loss.data
            train_correct += torch.sum(id == labels.data)
        print('[%d/%d] loss:%.3f, correct:%.3f%%, time:%s' %
              (epoch + 1, epochs, sum_loss / len(data_loader_train),
               100 * train_correct / len(data_train),
               time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
    model.train()


# ����ģ��
def test(model, test_loader):
    model.eval()
    test_correct = 0
    for data in test_loader:
        inputs, lables = data
        inputs, lables = Variable(inputs).cpu(), Variable(lables).cpu()
        inputs = torch.flatten(inputs, start_dim=1)  # չ������
        outputs = model(inputs)
        _, id = torch.max(outputs.data, 1)
        test_correct += torch.sum(id == lables.data)
    print(f'Accuracy on test set: {100 * test_correct / len(data_test):.3f}%')



# Ԥ��ģ��
def predict(model, data):
    model.eval()
    with torch.no_grad():
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
    return pred

#����ģ��
def save_network(model):
      torch.save(model, 'model.pkl')





batch_size = 64

#model = MLP(num_i, num_h, num_o)

model = torch.load('model.pkl')
#train(model)
test(model, data_loader_test)
#save_network(model)
'''
chosen = random.randint(1,60)
# Ԥ��ͼƬ������ȡ���Լ�ǰ10��ͼƬ
for i in range(chosen,chosen+10):
    # ��ȡ���������еĵ�һ��ͼƬ
    test_image = data_test[i][0]
    image_label=data_test[i][1]
    # չƽͼƬ
    test_image = test_image.flatten()
    # ����һά��Ϊ batch ά��
    test_image = test_image.unsqueeze(0)
    # ��ʾͼƬ
    plt.imshow(test_image.view(28, 28), cmap='gray')
    plt.show()
    pred = predict(model, test_image)
    print('Prediction:', pred.item())
    #print('Result:', image_label)
    print('\n')

'''
file= "./picture"
for im in os.listdir(file):
 
#�Լ���ͼ�Ĵ�ŵ�ַ
 picture = Image.open(file + "/" + im)
# print(picture)
# print(picture.mode)
# print(picture.getpixel((15,10)))
 picture_L=picture.convert('L',)
 img_array = np.array(picture_L)
#ת��Ϊnp���飬�Ա�ת��Ϊtensorģ��
#ת��Ϊfloattensor���飬28*28->1*784�����Ӧ
 a=torch.FloatTensor(img_array.reshape(1,784))
#[0,255]->[0,1]��ת��
 a=a/255
#��һ��������ֵ�������ǰ�����ݼ��Ĵ����Ӧ��
 a=(a-0.1307)/0.3081
 pred = predict(model, a)
# print(picture_L)
# print(picture_L.size)
# print(picture_L.getpixel((15,10)))
# print(picture_L.mode)
 inverted_image=ImageOps.invert(picture_L)#��Ϊ��ͼ����ǰ׵׺��֣���MNIST�෴������Ҫ��תһ��
# plt.figure(figsize=(15,15))
# plt.tick_params(colors='white')#����ͼƬ����̶���ɫ
######���ɵ�ͼ���Ѿ���һ�������ص���ֵ��0-1֮��
#inverted_image.save(r'D:\LearnPython\number3_gray.png')#����ת�����ͼ��
#plt.figure(figsize=(15,10))
 plt.tick_params(colors='white')  #��������̶���ɫ
 plt.imshow(inverted_image,cmap="gray")  #��ʾҪԤ�������ͼ��Ĭ����ʾ��
 plt.show()
 print('Prediction:', pred.item())



