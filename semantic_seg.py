#!pip install unrar
#!unrar x "file_path"
import numpy as np
import cv2
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchmetrics
from torchmetrics import Dice, JaccardIndex
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2 # convert np.array -> torch.tensor
import os
from tqdm import tqdm # slicing annimation 
from glob import glob #read img folder

class DogCatDatasets(Dataset):
  def __init__(self, root_dir, text_file, transform = None):
    # transform = augmentation + norm(convert pixel: 0-255 to 0-1) + np.array -> torch.tensor
    super().__init__()
    self.root_dir = root_dir #root directory
    self.txt = text_file
    self.transform = transform
    self.img_paths_list = [] # read img name in trainvl.txt file
    with open(self.txt) as file:
      for line in file:
        self.img_paths_list.append(line.split(" ")[0]) #split text by blank space. take 0 argument(dog/cat's name)
 
  def __len__(self): # training set/ test set length
    return len(self.img_paths_list)

  def __getitem__(self, idx): # -> label, image
    img_path = os.path.join(self.root_dir, "/content/images", "{}.jpg".format(self.img_paths_list[idx]))
    mask_path = os.path.join(self.root_dir, "annotations", "trimaps", "{}.png".format(self.img_paths_list[idx]))
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # -> [1 2 3]
    # Convert to binary classification
    # in annotations/ README
    # Pixel Annotations: 
    # 1: Foreground -> 1
    mask[mask == 2] = 0 # 2: Background -> 0
    mask[mask == 3] = 1 # 3: Not classified -> 1
    # img = (RGB), mask = 2D matrix
    if self.transform is not None:
      # transform will return a dictionary with two keys: 
      # image will contain the augmented image
      # mask will contain the augmented mask.
      transformed = self.transform(image=img, mask=mask)
      transformed_image = transformed['image']
      transformed_mask = transformed['mask']
    return transformed_image, transformed_mask

train_size = 384
train_transform = A.Compose([
    A.Resize(width = train_size, height = train_size),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Blur(),
    A.Sharpen(),
    A.RGBShift(), # each pixel will be moved randomly in color channel from 0 to r/g/b_shift_limit
    A.Cutout (num_holes=5, max_h_size=25, max_w_size=25, fill_value=0), #https://www.google.com/url?sa=i&url=https%3A%2F%2Fgithub.com%2Fxkumiyu%2Fnumpy-data-augmentation&psig=AOvVaw3syGbKWTWFkKP-WDLdDeEF&ust=1683729892902000&source=images&cd=vfe&ved=0CBEQjRxqFwoTCICB-eO86P4CFQAAAAAdAAAAABAE

    # Normalized (mean, std: imagenet)
    # Normalization is applied by the formula: img = (img - mean * max_pixel_value) / (std * max_pixel_value)
    A.Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    #Convert np.array -> torch.tensor (B, 3, H, W)
    ToTensorV2(),
]) # p: probability 

test_transform = A.Compose([
    A.Resize(width = train_size, height = train_size),
    A.Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
     ToTensorV2(),  
])


class UnNormalize(object):
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std
  
  def __call__(self, tensor):
    for t, m, s in zip(tensor, self.mean, self.std):
      t.mul_(s).add_(m)
    return tensor

unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


training_set = DogCatDatasets("/content", "/content/annotations/trainval.txt", train_transform)
test_set = DogCatDatasets("/content", "/content/annotations/test.txt", test_transform)
img, mask = training_set.__getitem__(100) #torch.Size([3, 384, 384]) torch.Size([384, 384])
# print(img.shape, mask.shape)
# print(mask.unique()) # ->tensor([0, 1], dtype=torch.uint8)
plt.subplot(1,2,1)
plt.imshow(unorm(img).permute(1, 2, 0)) #permute (H, W, 3)
plt.subplot(1,2,2)
plt.imshow(mask)
plt.show()


# import torch.nn as nn
# import torch.nn.functional as F
# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5)
#         self.conv2 = nn.Conv2d(20, 20, 5)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         return F.relu(self.conv2(x))

# OR
# UNet model: 4 block down, 1 neck, 4 block up
def unet_block(in_channel, out_channel):
  return nn.Sequential(
      nn.Conv2d(in_channel, out_channel, 3, 1, 1), #(in_channel, out_channel, kernel_size, stride, padding)
      nn.ReLU(),
      nn.Conv2d(out_channel, out_channel, 3, 1, 1), 
      nn.ReLU()
  )

class UNet(nn.Module):
  def __init__(self, n_class):
    super().__init__()
    self.n_class = n_class
    self.downsample = nn.MaxPool2d(2) # maxpool 2x2
    self.upsample = nn.Upsample(scale_factor=2, mode="bilinear") #Interpolation Search up-conv 2x2
    self.block_down1 = unet_block(3, 64)
    self.block_down2 = unet_block(64, 128)
    self.block_down3 = unet_block(128, 256)
    self.block_down4 = unet_block(256, 512)
    self.block_neck = unet_block(512, 1024)
    self.block_up1 = unet_block(1024+512, 512)
    self.block_up2 = unet_block(256+512, 256)
    self.block_up3 = unet_block(128+256, 128)
    self.block_up4 = unet_block(128+64, 64)
    # check if n_class has the maximum value in its channel label
    self.conv_cls = nn.Conv2d(64, self.n_class, 1) # the last convolutional operation(kernel 1x1) -> (B, n_class, H, W)

  def forward(self, x):
    x1 = self.block_down1(x)
    x = self.downsample(x1)
    x2 = self.block_down2(x)
    x = self.downsample(x2)
    x3 = self.block_down3(x)
    x = self.downsample(x3)
    x4 = self.block_down4(x)
    x = self.downsample(x4)
    # block neck
    x = self.block_neck(x)
   
    # concatenate output of down4 with up1
    x = torch.cat([x4, self.upsample(x)], dim=1) #dim = 1: concatenate channel (B, C, H, W) 
    x = self.block_up1(x) #convolutional operation and ReLU func
    #concat down3, up2
    x = torch.cat([x3, self.upsample(x)], dim=1) 
    x = self.block_up2(x) #convolutional operation and ReLU func
    #concat down2, up3
    x = torch.cat([x2, self.upsample(x)], dim=1) 
    x = self.block_up3(x) #convolutional operation and ReLU func
    #concat down1, up4
    x = torch.cat([x1, self.upsample(x)], dim=1) 
    x = self.block_up4(x) #convolutional operation and ReLU func
    x = self.conv_cls(x)
    return x


# Save the arguments after training model (loss, accuracy, ...)
class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count 


# accuracy calculate
def accuracy(actual, target):
  actual = actual.flatten() # -> to 1D vector
  target = target.flatten()
  acc = torch.sum(actual == target) 
  return acc/target.shape[0]



from numpy.ma.extras import average
# select gpu 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load data
batch_size = 8
n_workers = os.cpu_count() # use maximum cpu threads
trainloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size,
                                          shuffle=True, num_workers=n_workers) 

testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=False, num_workers=n_workers)

# Model 
model = UNet(1).to(device) # training with gpu

# loss function
criterion = nn.BCEWithLogitsLoss() # raw output: transfer through sigmoid/ softmax

# optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr=0.1e-4)
n_eps = 15 # epochs

#metrics
dice_calc = torchmetrics.Dice(num_classes=2, average="macro").to(device)
iou_calc = torchmetrics.JaccardIndex(num_classes=2, task="binary", average="macro").to(device)

#meter
accuracy_meter = AverageMeter()
train_loss_meter = AverageMeter()
dice_meter = AverageMeter()
iou_meter = AverageMeter()


# Model training
for ep in range(1, 1+n_eps):
  # Each loop will reset these args 
  accuracy_meter.reset()
  train_loss_meter.reset()
  dice_meter.reset()
  iou_meter.reset()
  model.train() 
  for batch_id, (x, y) in enumerate(tqdm(trainloader), start=1):
    optimizer.zero_grad() # delete derevative val, if not, it will be accumulated in loss.backward()
    n = x.shape[0]
    x = x.to(device).float()
    y = y.to(device).float()
    y_predict = model(x)
    y_predict = y_predict.squeeze() # target (4, 384, 384); y_predict (4,1,384,384); y_predict.squeeze() -> (4, 384, 384) to compare
    # y_predict: logit form 
    loss = criterion(y_predict, y)
    loss.backward() 
    optimizer.step() 

    with torch.no_grad(): 
      y_predict_mask = y_predict.sigmoid().round().long() 
      dice_score = dice_calc(y_predict_mask, y.long()) # dice score 1 batch training
      iou_score = iou_calc(y_predict_mask, y.long()) # -> tensor
      accuracy_score = accuracy(y_predict_mask, y.long())
    
      #update params
      train_loss_meter.update(loss.item(), n) # .item(): to float because loss must be in tensor form
      iou_meter.update(iou_score.item(), n) 
      dice_meter.update(dice_score.item(), n)
      accuracy_meter.update(accuracy_score.item(), n)
  print("Epoch: {}, Loss = {}, IoU = {}, Dice = {}, Accuracy = {}".format(
      ep, train_loss_meter.avg, iou_meter.avg, dice_meter.avg, accuracy_meter.avg
  ))

  # Save model 
  if ep >= 13:
    torch.save(model.state_dict(), "/content/model_ep_{}".format(ep))


model.eval()
test_iou_meter = AverageMeter()
test_dice_meter = AverageMeter()
with torch.no_grad():
    for batch_id, (x, y) in enumerate(tqdm(testloader), start=1):
        n = x.shape[0]
        x = x.to(device).float()
        y = y.to(device).float()
        y_predict = model(x)
        y_predict = y_predict.squeeze()
        y_predict_mask = y_predict.sigmoid().round().long() 
        dice_score = dice_calc(y_predict_mask, y.long()) # dice score 1 batch training
        iou_score = iou_calc(y_predict_mask, y.long())
        test_iou_meter.update(iou_score.item(), n) 
        test_dice_meter.update(dice_score.item(), n)
print("TEST: IoU = {}, dice = {}".format(test_iou_meter.avg, test_dice_meter.avg))


import random
model.eval()
idx = random.randint(0,100)
with torch.no_grad():
  x,y = test_set[idx]
  x = x.to(device).float().unsqueeze(0) # add B=1 in the first place
  y_predict = model(x).squeeze() #(1,1,H,W) -> (H,W)
  y_predict_mask = y_predict.sigmoid().round().long()
  #plot x, y, y_predict_mask
  plt.subplot(1,3,1)
  plt.imshow(unorm(x.squeeze().cpu()).permute(1, 2, 0)) #permute (H, W, 3) # gpu -> cpu
  plt.subplot(1,3,2)
  plt.imshow(y) # target 
  plt.subplot(1,3,3)
  plt.imshow(y_predict_mask.cpu()) # actual output
  plt.show()