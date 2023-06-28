# Facial_Age_Estimation

Here we have used PyTorch with the help of which we pre-trained and added
Convolutional and Relu to train our model and achieve better accuracy. We
have used ADAM optimizer with the following parameters:
• Learning rate = 0.0005
• Weight decay = 4e-5 (L2 penalty)
The hyperparameters are:
• Learning rate = 0.0005
• Batch size = 8
• Number of epochs = 20
Furthermore, due to computational limitations, for partition of the data into
training, validation, and testing, we have created a Python file called “Wrapper”
which would create .npy files for X_train, y_train, X_Val, y_val, X_test and
y_test. This helped us train our final code in a Python file named “ResNet50”.
RMSE losses –

Training:
[17.83030536 14.53167456 13.39173478 12.90225339 11.80618891
11.40607023 10.17670815 9.41862352 9.01958858 8.1717321 7.99084143
7.72224851 7.75391551 7.40045151 7.2985783 7.35081178 7.1979003
6.99527553 6.64132012 6.70200313]

Validation: 
[16.64563913 17.02206461 13.65341548 14.41499651
13.32973921 14.81964601 13.66807969 13.31877572 14.79746835
13.80046012 13.76470854 13.8190339 12.92743897 12.90926331
12.73436824 12.90561434 12.80094331 12.94288752 13.30139966
12.76232665]

Testing Age Accuracy: 12.767920696597056


