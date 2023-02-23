# In[]
#libary
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
# In[]
class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        
        self.conv1_1=nn.Conv2d(in_channels=3,out_channels=16, kernel_size=3, stride=1, padding=1)  
        self.conv1_2=nn.Conv2d(in_channels=16,out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_1=nn.Conv2d(in_channels=32, out_channels=32,kernel_size=3,stride=1,padding=1)
        self.conv3_1=nn.Conv2d(in_channels=32, out_channels=64,kernel_size=3,stride=1,padding=1)
 
        self.pool=nn.MaxPool2d(2,2)
        self.bn1=nn.BatchNorm2d(16)
        self.bn2=nn.BatchNorm2d(32)
        self.bn3=nn.BatchNorm2d(64)
        self.fc1=nn.Linear(64*8*8,10)

    def forward(self, input):
        output=F.relu(self.conv1_1(input))
        output=self.pool(output)
        output=self.bn1(output)
        
        output=F.relu(self.conv1_2(output))
        output=self.pool(output)
        output=self.bn2(output)
        
        output=F.relu(self.conv2_1(output))
        output=self.pool(output)
        output=self.bn2(output)
        
        output=F.relu(self.conv3_1(output))
        output=self.pool(output)
        output=self.bn3(output)
        
        output=output.view(-1,64*8*8)#攤平
        output=F.relu(self.fc1(output))
       
        return output
# In[]
class SportLoader(data.Dataset):
    def __init__(self,mode,transform=None):
       
        self.mode=mode
        self.sport=pd.read_csv(mode+'.csv')
        self.img_name=np.asarray(self.sport.iloc[:,0])#取cvs第一行資料
        self.label=np.asarray(self.sport.iloc[:,1])#取csv第二行資料
        self.transform=transform #image transform
    
    def __len__(self):
        return len(self.img_name)
    
    def __getitem__(self, index):
        image_path=self.mode+"/"+self.img_name[index]
        self.img =Image.open(image_path)#Image.open(image_path)#io.imread(image_path)#
        self.target=self.label[index]
        
        if self.transform:
            self.img=self.transform(self.img)
        
        return self.img,self.target
# In[]
transform_train=transforms.Compose([
    #transforms.RandomRotation(15),
    #transforms.RandomRotation([90,180]),
    transforms.Resize([128,128]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
transform_valid=transforms.Compose([
    transforms.Resize([128,128]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
train_dataset=SportLoader("train",transform_train)
valid_dataset=SportLoader("val",transform_valid)

# In[]
train_loader=DataLoader(train_dataset, batch_size=10,shuffle=True)
valid_loader=DataLoader(valid_dataset, batch_size=10,shuffle=True)
# In[]
model=Network()
optimizer=torch.optim.SGD(model.parameters(), lr=0.001,momentum=0.9)
loss_func=nn.CrossEntropyLoss()
model=model.cuda()
loss_func=loss_func.cuda()
input_shape=(-1,3,128,128)
num_epochs=30
# In[]
def fit_model(model, loss_func, optimizer, input_shape, num_epochs, train_loader, valid_loader):
    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []
    for epoch in range(num_epochs):
        correct_train = 0
        total_train = 0
        model.train()
        for i,(images, labels) in enumerate(train_loader):#enumerate
           # 1.Define variables
           train =images.view(input_shape).float().cuda()
           labels = labels.cuda()
           # 3.Forward propagation
           outputs = model(train).cuda()
           # 2.Clear gradients
           optimizer.zero_grad()
           # 4.Calculate softmax and cross entropy loss
           train_loss = loss_func(outputs, labels).cuda()
           # 5.Calculate gradients
           train_loss.backward()
           # 6.Update parameters
           optimizer.step()
           # 7.Get predictions from the maximum value
           predicted = torch.max(outputs.data, 1)[1]
           # 8.Total number of labels
           total_train += len(labels)
           # 9.Total correct predictions
           correct_train += (predicted == labels).float().sum()
           torch.cuda.empty_cache()
        #10.store val_acc / epoch
        train_accuracy = 100 * correct_train / float(total_train)
        training_accuracy.append(train_accuracy.cpu())
        # 11.store loss / epoch
        training_loss.append(train_loss.data.cpu())
        correct_test = 0
        total_test = 0
        model.eval()
        with torch.no_grad():
            for images, labels in valid_loader:
                # 1.Define variables
                valid = images.view(input_shape).float().cuda()
                labels =labels.cuda()
                # 2.Forward propagation
                outputs = model(valid).cuda()
                # 3.Calculate softmax and cross entropy loss
                val_loss = loss_func(outputs, labels).cuda()
                # 4.Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                # 5.Total number of labels
                total_test += len(labels)
                # 6.Total correct predictions
                correct_test += (predicted == labels).float().sum()
                torch.cuda.empty_cache()
            # 7.store val_acc / epoch
            val_accuracy = 100 * correct_test / float(total_test)
            validation_accuracy.append(val_accuracy.cpu())
            # 8.store val_loss / epoch
            validation_loss.append(val_loss.data.cpu())
        print('Train Epoch: {}/{} Traing_Loss: {} Traing_acc: {:.6f}% Val_Loss: {} Val_accuracy: {:.6f}%'.format(epoch+1, num_epochs, train_loss.data, train_accuracy, val_loss.data, val_accuracy))
    return training_loss, training_accuracy, validation_loss, validation_accuracy  
# In[]
training_loss, training_accuracy, validation_loss, validation_accuracy = fit_model(model, loss_func, optimizer, input_shape, num_epochs, train_loader, valid_loader)
# visualization
plt.plot(range(num_epochs), training_loss, 'b-', label='Training_loss')
plt.plot(range(num_epochs), validation_loss, 'g-', label='validation_loss')
plt.title('Training & Validation loss')
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.plot(range(num_epochs), training_accuracy, 'b-', label='Training_accuracy')
plt.plot(range(num_epochs), validation_accuracy, 'g-', label='Validation_accuracy')
plt.title('Training & Validation accuracy')
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# In[]
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
# In[]
torch.save(model.state_dict(),'HW1_311553046.pt')
