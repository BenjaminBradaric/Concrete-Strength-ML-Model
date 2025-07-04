import torch
from torch import nn
from Processing import features_training_set, labels_train_set
from model import model
import matplotlib.pyplot as plt


epochs = 50

loss_rate = []

lossfunc = nn.HuberLoss(delta = 0.4, reduction= "mean")

#optimizer = torch.optim.Adam(params= model.parameters(), lr=1e-4, weight_decay=1e-6)
optimizer = torch.optim.RMSprop(params= model.parameters(), lr=1e-1,)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

for i in range(epochs):
    model.train()

    predictions = model(features_training_set)

    loss = lossfunc(predictions, labels_train_set)
    print(f'The loss is: {loss.item()}')
    loss_rate.append(loss.item())

    optimizer.zero_grad()

    loss.backward() 

    optimizer.step()

    scheduler.step()

torch.save(model.state_dict(), "saved_model.pth")

#Ploting the loss curve
plt.plot( range(0, epochs ,1) ,loss_rate)
plt.xlabel("Epochs")
plt.ylabel("Loss Rate")
plt.show()    