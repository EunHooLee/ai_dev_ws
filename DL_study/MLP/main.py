import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

import numpy as np
import matplotlib.pyplot as plt

import torch.nn.init as init


DEVICE = th.device('cuda' if th.cuda.is_available() else 'cpu')
print('Using Pytorch version: ',th.__version__,' Device: ',DEVICE)

BATCH_SIZE = 32
EPOCHS = 10

train_dataset = datasets.MNIST(root= "./data/MNIST",
train=True,
download=True,
transform=transforms.ToTensor())

test_dataset = datasets.MNIST(root="./data/MNIST",
train=False,
transform=transforms.ToTensor())

train_loader = th.utils.data.DataLoader(dataset=train_dataset,
batch_size=BATCH_SIZE,
shuffle=True)

test_loader = th.utils.data.DataLoader(dataset=test_dataset,
batch_size=BATCH_SIZE,
shuffle=False)

for X_train, Y_train in train_loader:
    print('X_train: ', X_train.size(),' type: ',X_train.type())
    print('Y_train: ', Y_train.size(),' type: ',Y_train.type())
    break

pltsize = 1
plt.figure(figsize=(10 * pltsize, pltsize))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(X_train[i,:,:,:].numpy().reshape(28,28), cmap="gray_r")
    plt.title('Class: ' + str(Y_train[i].item()))

# MLP
# class Net(nn.Module):
#     def __init__(self):
#         super(Net,self).__init__()
#         self.fc1 = nn.Linear(28 * 28, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256,10)
    
#     def forward(self,x):
#         x = x.view(-1,28*28) # Flatten (1*1*28*28 tensor를 row vector로 변환)
#         x = self.fc1(x)
#         x = F.sigmoid(x)
#         x = self.fc2(x)
#         x = F.sigmoid(x)
#         x = self.fc3(x)
#         x = F.log_softmax(x,dim =1)
#         return x

# MLP with Dropout
# class Net(nn.Module):
#     def __init__(self):
#         super(Net,self).__init__()
#         self.fc1 = nn.Linear(28 * 28, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256,10)
#         self.dropout_prob = 0.5
    
#     def forward(self,x):
#         x = x.view(-1,28*28) # Flatten (1*1*28*28 tensor를 row vector로 변환)
#         x = self.fc1(x)
#         x = F.sigmoid(x)
#         x = F.dropout(x, training= self.training, p = self.dropout_prob)
#         x = self.fc2(x)
#         x = F.sigmoid(x)
#         x = F.dropout(x, training= self.training, p = self.dropout_prob)
#         x = self.fc3(x)
#         x = F.log_softmax(x,dim =1)
#         return x
    # self.traning 은 아래 코드에서 model.train() 을 하면 True, model.eval() 하면 False 가 된다.
    # True 일 경우만 dropout을 적용한다. 즉, evaluation(평가) 과정에는 적용하지 않는다.

# MLP with ReLU + Dropout + batch normalization
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256,10)
        self.dropout_prob = 0.5
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(256)

    def forward(self,x):
        x = x.view(-1,28*28) # Flatten (1*1*28*28 tensor를 row vector로 변환)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.dropout(x, training= self.training, p = self.dropout_prob)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.dropout(x, training= self.training, p = self.dropout_prob)
        x = self.fc3(x)
        x = F.log_softmax(x,dim =1)
        return x

def weight_init(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight.data)

model = Net().to(DEVICE)
model.apply(weight_init)
optimizer  = th.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
optimizer = th.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

def train(model, train_loader, optimizer, log_interval):
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print("Train Epoch: ,{} [{}/{}({:.0f}%)]\tTrain Loss: {:.6f}".format(
                Epoch, batch_idx * len(image), len(train_loader.dataset),100. * batch_idx / len(train_loader),loss.item()
            ))

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with th.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim=True)[1]
            correct +=prediction.eq(label.view_as(prediction)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

for Epoch in range(0,EPOCHS+1):
    train(model, train_loader,optimizer, log_interval= 200)
    test_loss , test_accuracy = evaluate(model, test_loader)
    print("\n[EPOCH: {} ], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} %\n".format(Epoch, test_loss, test_accuracy))
    