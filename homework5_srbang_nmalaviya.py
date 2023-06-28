import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt



X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X) 
        self.y = torch.from_numpy(y) 
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        image = self.X[idx].clone().detach().requires_grad_(False).float()
        image = normalize(image)
        age = self.y[idx].clone().detach().requires_grad_(True).long()
        return image, age

print("check3")

# Using pretrained weights:

pretrained_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

batch_size = 8
batch_size_val = 8
dataset = MyDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_dataset = MyDataset(X_val, y_val)
val_loader =  torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)


print("done")

# Replace the last layer with a new fully connected layer
in_features = pretrained_model.fc.in_features
num_classes = 1
# pretrained_model.fc = nn.Linear(in_features, num_classes)
pretrained_model.fc = nn.Sequential(nn.Linear(in_features, 1024), nn.ReLU(),nn.Dropout(0.5),nn.Linear(1024, 512), nn.ReLU(),nn.Dropout(0.3),nn.Linear(512, 64), nn.ReLU(),nn.Dropout(0.2),nn.Linear(64, 8), nn.ReLU(),nn.Dropout(0.2), nn.Linear(8,1))
print("check4")
pretrained_model = pretrained_model.to(device)
print("check4")

for param in pretrained_model.parameters():
    param.requires_grad = True  #2048 features, converts to 1000 so 2048 down 1
pretrained_model.fc.requires_grad = True


# Define loss function and optimizer
criterion = nn.MSELoss()# mse
optimizer = torch.optim.Adam(pretrained_model.parameters(), lr=0.0005, weight_decay=4e-5)

print("hua")

num_epochs = 20

training_loss = []
validation_loss = []

# Train the model
for epoch in range(num_epochs):
    total_loss = 0
    pretrained_model.train()
    print("current epoch", epoch)
    for i,data in enumerate(train_loader):
    # Forward pass
    # pretrained_model= pretrained_model.double()
        inputs, labels = data
        labels = torch.unsqueeze(labels,dim =1)

        inputs = inputs.float()
        inputs = inputs.to(device)
        labels = labels.to(device)

        labels = labels.float()
        outputs = pretrained_model(inputs)
        outputs = outputs.to(device)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    # Print progress
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, np.sqrt(total_loss*batch_size/len(X_train))))
    training_loss.append(np.sqrt(total_loss*batch_size/len(X_train)))

    pretrained_model.eval()
    #torch.cuda.empty_cache()
    total_loss = 0
    with torch.no_grad():
        for i,data in enumerate(val_loader):
        # Forward pass
        # pretrained_model= pretrained_model.double()
            inputs, labels = data
            labels = torch.unsqueeze(labels,dim =1)

            inputs = inputs.float()
            inputs = inputs.to(device)
            labels = labels.to(device)

            labels = labels.float()
            outputs = pretrained_model(inputs)
            #outputs = outputs.to(device)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            total_loss += loss.item()
            
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, np.sqrt(total_loss*batch_size_val/len(X_val))))
        validation_loss.append(np.sqrt(total_loss*batch_size_val/len(X_val)))


  
del X_train
del y_train
del X_val
del y_val
del dataset
del train_loader

validation_loss = np.asarray(validation_loss)
training_loss = np.asarray(training_loss)

print(validation_loss)
print(training_loss)

np.save('training_loss.npy',training_loss)
np.save('validation_loss.npy',validation_loss)


X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

x_range = range(num_epochs)

plt.plot(x_range,validation_loss,label='validation_loss')
plt.plot(x_range,training_loss,label='training_loss')
plt.xlabel('Number of epochs') 
plt.ylabel('Loss') 
  
# displaying the title
plt.title("Training and validation loss vs epochs")
plt.legend()
plt.show()

# X_val = np.load('X_val.npy')
# y_val = np.load('y_val.npy')

batch_size = 8
dataset = MyDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

# batch_size = 2
# dataset=list(zip(X_train,y_train)) #wrong, input in batchsize in classes
# train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) 


print("done")


pretrained_model.eval()

total_loss = 0


for i,data in enumerate(test_loader):
# Forward pass
# pretrained_model= pretrained_model.double()
    inputs, labels = data
    labels = torch.unsqueeze(labels,dim =1)

    inputs = inputs.float()
    inputs = inputs.to(device)
    labels = labels.to(device)

    labels = labels.float()
    outputs = pretrained_model(inputs)
    outputs = outputs.to(device)
    loss = criterion(outputs, labels)
    total_loss += loss.item()
    
    # Backward pass and optimization

    
# Print progress
print(len(X_test))
print("testing loss",np.sqrt((total_loss*batch_size/len(X_test))))



# X_train = np.load('X_val.npy')
# y_train = np.load('y_val.npy')
# # X_val = np.load('X_val.npy')
# # y_val = np.load('y_val.npy')

# batch_size = 10
# dataset = MyDataset(X_train, y_train)
# train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

# # batch_size = 2
# # dataset=list(zip(X_train,y_train)) #wrong, input in batchsize in classes
# # train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) 


# print("done")


# pretrained_model.eval()

# total_loss = 0


# for i,data in enumerate(train_loader):
# # Forward pass
# # pretrained_model= pretrained_model.double()
#     inputs, labels = data
#     labels = torch.unsqueeze(labels,dim =1)

#     inputs = inputs.float()
#     inputs = inputs.to(device)
#     labels = labels.to(device)

#     labels = labels.float()
#     outputs = pretrained_model(inputs)
#     outputs = outputs.to(device)
#     loss = criterion(outputs, labels)
#     total_loss += loss.item()
    
#     # Backward pass and optimization

    
# # Print progress
# print(len(X_train))
# print(np.sqrt((total_loss*batch_size/len(X_train))))

