import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,num_classes):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*4*4,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,num_classes)
    
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def train(net,trainloader,optimizer,epochs,device:str ):
    criterion = nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()

def test(net,testloader,device:str):
    criterion = nn.CrossEntropyLoss()
    correct , loss = 0,0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            output = net(images)
            loss += criterion(output,labels).item()
            _, predicted = torch.max(output,1)
            correct += (predicted == labels).sum().item()
    accuracy = correct/len(testloader.dataset)
    return loss, accuracy