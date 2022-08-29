import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image
import glob

from haze import haze
from snow import snow
from low_light import low_light

import glob, os, shutil

from torchvision.models import vgg16
from torchvision import models

class CNN_model(nn.Module):
  def __init__(self):
    super(CNN_model, self).__init__()
    self.convlayer_1 = nn.Conv2d(3, 16, 3, 2)
    self.convlayer_2 = nn.Conv2d(16, 32, 3, 2)
    self.convlayer_3 = nn.Conv2d(32, 32, 3, 2)
    self.activate = nn.LeakyReLU(0.1)
    self.fc1 = nn.Linear(1568, 64)
    self.fc2 = nn.Linear(64, 3)
    self.sigmoid = nn.Sigmoid()
  def forward(self,img):
    l1 = self.convlayer_1(img)
    l1_1 = self.activate(l1)
    l2 = self.convlayer_2(l1_1)
    l2_1 = self.activate(l2)
    l3 = self.convlayer_3(l2_1)
    l3_1 = self.activate(l3)
    l4 = self.convlayer_3(l3_1)
    l4_1 = self.activate(l4)
    l5 = self.convlayer_3(l4_1)
    l5_1 = self.activate(l5)
    x = torch.reshape(l5_1,(-1,1568))   #256*256*32
    x1= self.fc1(x)
    x1_1 = self.activate(x1)
    x2 = self.fc2(x1_1)
    # out = self.sigmoid(x2)
    # return out
    return x2

class dataset(Dataset):
    
    def __init__(self, data_root, data_root_clean, transforms):
        self.data_root = data_root
        self.data_root_clean = data_root_clean
        self.transforms = transforms
        
        self.image_path_list = sorted(glob.glob(data_root + '/*.jpg'))
        self.image_path_list_clean = sorted(glob.glob(data_root_clean + '/*.jpg'))
        
    def __len__(self):
        return len(self.image_path_list)
        
    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        img = Image.open(image_path)
        img_tensor = self.transforms(img)
        
        image_path_clean = self.image_path_list_clean[idx]
        img_clean = Image.open(image_path_clean)
        img_tensor_clean = self.transforms(img_clean)
        
        return img_tensor, img_tensor_clean, image_path, image_path_clean    

lowvis_train_path = '/images/train'  #path of low visibility training images
clean_train_path = 'original/train'  #path of original train images captured in favourable ambience
lowvis_test_path = '/images/test'  #path of low visibility testing images
clean_test_path = 'original/test'  #path of original test images captured in favourable ambience

train_dataset = dataset(lowvis_train_path, clean_train_path , transforms=transforms.Compose([
                                      transforms.Resize((256, 256)),
                                      transforms.ToTensor()
                                   #   transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))
test_dataset = dataset(lowvis_test_path, clean_test_path, transforms=transforms.Compose([
                                      transforms.Resize((256, 256)),
                                      transforms.ToTensor()
                                #      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))
train_loader = DataLoader(train_dataset, batch_size=1, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, pin_memory=True)

model = CNN_model().to('cuda')
resume= False
if resume:
  model.load_state_dict(torch.load('path'))
  model.to('cuda')

num_epochs= 200
lr = 1e-4

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) #check optimizer used in iayolo
model.train()

criterion = nn.BCEWithLogitsLoss().to('cuda')
best_loss = 10000000000
print('---starting training---')
for epoch in range(num_epochs):
    # some initialization stuff
  losses = []
  for i, (img, img_clean, img_path, img_path_clean) in enumerate(train_loader):
    print("iter: "+str(i))
    optimizer.zero_grad()
    params = model(img.to('cuda'))
    # copy image to intermediate 
    img_path = img_path[0]
    img_path_clean = img_path_clean[0]
    # shutil.copy2(img_path, './intermediate_files')
    # intermediate_file_path = './intermediate_files/'+img_path.split('/')[-1]
    print(params[0][0], params[0][1], params[0][2])
    img_name =  img_path.split('/')[-1]
    img_type = img_name.split('_')[-1].split('.')[0]
    print('img type: '+str(img_type))
    if img_type == '1':
      #normal
      print('normal')
      target = torch.tensor([[0,0,0]]).float().to('cuda')
    elif img_type == '2':
      #haze
      print('haze')
      target = torch.tensor([[0,1,0]]).float().to('cuda')
    elif img_type == '3':
      #lowlight
      print('lowlight')
      target = torch.tensor([[1,0,0]]).float().to('cuda')
    elif img_type == '4':
      #haze+lowlight
      print('haze+lowlight')
      target = torch.tensor([[1,1,0]]).float().to('cuda')
    elif img_type == '5':
      #haze+snow
      print('haze+snow')
      target = torch.tensor([[0,1,1]]).float().to('cuda')
    elif img_type == '6':
      #snow
      print('snow')
      target = torch.tensor([[0,0,1]]).float().to('cuda')

    loss = criterion(params,target)
    loss.backward()
    optimizer.step()
    losses.append(loss.item)
    print(f'\loss iter: {loss.item():.5f}', flush=True)
  mean_loss = sum(losses)/len(losses)
  print('epoch '+str(epoch)+' : loss'+str(mean_loss))
  if mean_loss < best_loss:
    save_path = 'saved_models/'+str(epoch)+'.pt'
    torch.save(model.state_dict(), save_path)
    best_loss = mean_loss
