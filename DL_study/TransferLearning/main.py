import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

import torchvision.models as models
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

DEVICE = th.device('cuda' if th.cuda.is_available() else 'cpu')
print('Using PyTorch Version: {} , Device : {}'.format(th.__version__, DEVICE))

BATCH_SIZE = 32
EPOCHS = 10

# Data pre-processing
data_transform = {
    'train' : transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
}

image_datasets = {x : datasets.ImageFolder("./data/hymenoptera_data", data_transform[x]) for x in ['train','val']}

dataloaders = {x : th.utils.data.DataLoader(image_datasets[x],batch_size=BATCH_SIZE,num_workers=0,shuffle=True) for x in ['train', 'val']}

for (X_train, Y_train) in dataloaders['train']:
    print('X_train : {}, type : {}'.format(X_train.size(), X_train.type()))
    print('Y_train : {}, type : {}'.format(Y_train.size(), Y_train.type()))
    break

# pltsize = 1
# plt.figure(figsize=(10 * pltsize, pltsize))
# for i in range(10):
#     plt.subplot(1,10,i+1)
#     plt.axis('off')
#     image_ = np.transpose(X_train[i],(1,2,0)).numpy()
#     # image = np.array([0.5,0.5,0.5]) * image + np.array([0.5,0.5,0.5])
#     image_ = np.clip(image_,0,1)
#     plt.title('Class: ' + str(Y_train[i].item()))
#     plt.imsave('./data/image_{}.png'.format(i),image_)
#     # plt.imshow(image)

# Model
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,2) # 이거 왜 2 로하면 안되냐?  나머지는 다되는데 ###########################################
model = model.to(DEVICE)

optimizer = th.optim.Adam(model.parameters(), lr = 0.0001)
criterion = nn.CrossEntropyLoss()
print(model)


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
            print('Train Epoch : {} [{}/{} ({:.0f}%)\tTrain Loss : {:.6f}'.format(Epoch, batch_idx * len(image),len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with th.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss +=criterion(output, label).item()
            prediction = output.max(1, keepdim=True)[1]
            correct +=prediction.eq(label.view_as(prediction)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy



if __name__ == '__main__':
    for Epoch in range(1,EPOCHS+1):
        train(model, dataloaders['train'],optimizer,log_interval=5)
        test_loss, test_accuracy = evaluate(model, dataloaders['val'])
        print("\n[EPOCH : {}],\tTest Loss: {:.4f},\tTest Accuracy: {:.2f} % \n".format(Epoch, test_loss, test_accuracy))
