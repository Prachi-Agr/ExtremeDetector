'''
Generate enhanced images using trained CNN that identifies which enhancement models the images need to be passed through.
Train YOLOv5 using enhanced images
'''
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

from haze import haze   # import haze removal model
from snow import snow   # import snow removal model
from low_light import low_light  #import low light removal model

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
    
    return x2

#load model
path = 'saved_models/113.pt'
model = CNN_model().to('cuda')
model.load_state_dict(torch.load(path))

# generate enhanced images for test set
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
        
        # print(image_path_clean, image_path, type(image_path))
        return img_tensor, img_tensor_clean, image_path, image_path_clean        


test_dataset = dataset('DatasetCompiled/images/test/', 'DatasetCompiled/clean/test/', transforms=transforms.Compose([
                                      transforms.Resize((256, 256)),
                                      transforms.ToTensor()
                                #      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))
# train_loader = DataLoader(train_dataset, batch_size=1, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, pin_memory=True)

correct = 0
total = 0

print('generate test images')
import time

start = time.time()

for i, (img, img_clean, img_path, img_path_clean) in enumerate(test_loader):
    params = model(img.to('cuda'))
    params = torch.sigmoid(params)
    print('iter: '+str(i))
    print(params)
    img_path = img_path[0]
    img_path_clean = img_path_clean[0]
    shutil.copy2(img_path, './enhanced_mock/test')
    enhanced_file_path = './enhanced_mock/test/'+img_path.split('/')[-1]

    # classification accuracy based on img type
    img_name =  img_path.split('/')[-1]
    img_type = img_name.split('_')[-1].split('.')[0]
    print('img type: '+str(img_type), flush=True)
    if img_type == '1':
      #normal
      target = torch.tensor([[0,0,0]]).float().to('cuda')
    elif img_type == '2':
      #haze
      target = torch.tensor([[0,1,0]]).float().to('cuda')
    elif img_type == '3':
      #lowlight
      target = torch.tensor([[1,0,0]]).float().to('cuda')
    elif img_type == '4':
      #haze+lowlight
      target = torch.tensor([[1,1,0]]).float().to('cuda')
    elif img_type == '5':
      #haze+snow
      target = torch.tensor([[0,1,1]]).float().to('cuda')
    elif img_type == '6':
      #snow
      target = torch.tensor([[0,0,1]]).float().to('cuda')
    
    ll_val = 0
    haze_val = 0
    snow_val = 0
    # pass through suggested models and save 
    if params[0][0]>0.5:
      # pass into low viz removal model
        print('low light')
        low_light(enhanced_file_path)
        ll_val = 1

    if params[0][1]>0.5:
      #pass into dehaze model
        print('haze')
        haze(enhanced_file_path)
        haze_val = 1

    if params[0][2]>0.5:
      #pass into desnow model
        print('snow')
        snow(enhanced_file_path)
        snow_val = 1


end = time.time()
print(end - start)