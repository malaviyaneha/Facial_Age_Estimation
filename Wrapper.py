import torch
import numpy as np
import torchvision.transforms as transforms
import cv2
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io import read_image
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()


data_age = np.load('ages.npy')
data_face = np.load('faces.npy')
X_train, X_val, y_train, y_val = train_test_split(data_face, data_age, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

data_face = np.concatenate((X_train , X_val , X_test),axis=0)
data_age = np.concatenate((y_train , y_val ,y_test),axis=0)

# Create an empty array to hold the resized images
resized_images = np.empty((data_face.shape[0], 3, 224, 224))

# Loop over the images and resize them
for i in range(data_face.shape[0]):
    # Resize the image to 224x224 using numpy
    image = np.resize(data_face[i], (224,224))
    
    # Replicate the grayscale values to create three channels
    image = np.stack((image,) * 3, axis=0)
        
    resized_images[i] = image/255

print("shape",resized_images[0])



class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X) 
        self.y = torch.from_numpy(y) 
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx],dtype=torch.float32), torch.tensor(self.y[idx],dtype=torch.long)


X_train = resized_images[0:4500]
y_train = data_age[0:4500]

X_test = resized_images[4500:6000]
y_test = data_age[4500:6000]

X_val = resized_images[6000:7500]
y_val = data_age[6000:7500]


np.save('X_train.npy',X_train)
np.save('y_train.npy',y_train)
np.save('X_test.npy',X_test)
np.save('y_test.npy',y_test)
np.save('X_val.npy',X_val)
np.save('y_val.npy',y_val)
