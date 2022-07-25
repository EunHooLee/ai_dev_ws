from pickletools import optimize
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt


DEVICE = th.device('cuda' if th.cuda.is_available() else 'cpu')
print("Using Pytorch version : ", th.__version__, "Device : ",DEVICE)

BATCH_SIZE = 32
EPOCKS = 10

train_dataset = datasets.FashionMNIST(root='./data/FashionMNIST',
train=True,
download=True,
transform=transforms.ToTensor())

test_dataset = datasets.FashionMNIST(root='./data/FashionMNIST',
train=False,
transform=transforms.ToTensor())

train_loader = th.utils.data.DataLoader(dataset=train_dataset,
batch_size=BATCH_SIZE,
shuffle=True)

test_loader = th.utils.data.DataLoader(dataset=test_dataset,
batch_size=BATCH_SIZE,
shuffle=False)

for (X_train, Y_train) in train_loader:
    print("X_train : ", X_train.size(), ' type : ', X_train.type())
    print("Y_train : ", Y_train.size(), ' type : ', Y_train.type())
    break

pltsize = 1
plt.figure(figsize=(10*pltsize, pltsize))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(X_train[i,:,:,:].numpy().reshape(28,28), cmap= 'gray_r')
    plt.title('Class : ' + str(Y_train[i].item))

# AE Model

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,32),
        )

        self.decoder = nn.Sequential(
            nn.Linear(32,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
        )
    
    def forward(self,x):
        encoded = self.encoder(x) # Latent Variable Vector
        decoded = self.decoder(encoded)
        return encoded, decoded

model = AE().to(DEVICE)
optimizer = th.optim.Adam(params=model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print(model)

def train(model, train_loader, optimizer, log_interval):
    model.train()
    for batch_idx, (image,_) in enumerate(train_loader): #Iteration 정의, label은 안쓴다. 데이터를 배치만큼 가져와서 gradient update, for 1 당 iter 1.
        image = image.view(-1,28*28).to(DEVICE)
        target = image.view(-1,28*28).to(DEVICE)
        optimizer.zero_grad()
        encoded, decoded = model(image) # 왜 model.forward(image) 라고 안해도 되지?
        loss = criterion(decoded, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{}({:.0f}%)]\tTrain Loss : {:.6f}".format(Epoch, batch_idx * len(image),len(train_loader.dataset),
            100 * batch_idx/len(train_loader),loss.item()))

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    real_image = []
    gen_image = []
    with th.no_grad():
        for image, _ in test_loader:
            image = image.view(-1,28*28).to(DEVICE)
            target = image.view(-1,28*28).to(DEVICE)
            encoded, decoded = model(image)

            test_loss +=criterion(decoded, image).item()
            real_image.append(image.to("cpu"))
            gen_image.append(decoded.to("cpu"))

    test_loss /= len(test_loader.dataset)
    return test_loss, real_image, gen_image

for Epoch in range(1,EPOCKS+1):
    train(model,train_loader,optimizer,log_interval=200)
    test_loss, real_image, gen_image = evaluate(model,test_loader)
    print("\n[EPOCH: {}],\tTest Loss: {:.4f}".format(Epoch, test_loss))

    f, a = plt.subplots(2,10,figsize=(10,4))

    for i in range(10):
        img = np.reshape(real_image[0][i],(28,28))
        a[0][i].imshow(img,cmap="gray_r")
        a[0][i].set_xticks(())
        a[0][i].set_yticks(())

    for i in range(10):
        img = np.reshape(gen_image[0][i],(28,28))
        a[0][i].imshow(img,cmap="gray_r")
        a[0][i].set_xticks(())
        a[0][i].set_yticks(())
    plt.show() 

        