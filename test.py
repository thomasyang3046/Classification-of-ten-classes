import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import os
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
def fit_model(model, input_shape, valid_loader):
    #validation_accuracy = []
    label_result=[]
    model.eval()
    with torch.no_grad():
        for images in valid_loader:
            # 1.Define variables
            valid = images.view(input_shape).cuda()
            outputs = model(valid).cuda()
            predicted = outputs.data.max(1)[1]
            label_result.append(predicted.item())
        return label_result

# In[]

class SportLoader(data.Dataset):
    def __init__(self,mode,transform=None):
        self.mode=mode
        self.transform=transform #image transform
    def __len__(self):
        return len(self.img)
    def __getitem__(self, index):
        imagelist=os.listdir('test')
        imagelist.sort(key=lambda x:int(x.split('.')[0]))
        image_path=self.mode+"/"+imagelist[index]
        self.img =Image.open(image_path)#io.imread(image_path)#
        if self.transform:
            self.img=self.transform(self.img)
        return self.img#,self.target#,self.img_name[index]
# In[]
load_model=Network()
load_model.load_state_dict(torch.load('HW1_311553046.pt'))
load_model.cuda()
# In[]
transform_valid=transforms.Compose([
    transforms.Resize([128,128]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
# In[]
valid_dataset=SportLoader("test",transform_valid)
input_shape=(-1,3,128,128)
label=fit_model(load_model,input_shape,valid_dataset)
print(label)
# In[]
'''imagelist=os.listdir('test')
imagelist.sort(key=lambda x:int(x.split('.')[0]))
filename="HW1_311553046.csv"
submission=pd.DataFrame({'names':imagelist,'label':label})
submission.to_csv(filename,index=False)'''
