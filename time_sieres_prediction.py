#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
torch.manual_seed(0)



data = pd.read_csv("Sales_Transactions_Dataset_Weekly.csv")
data = data.iloc[:,1:53]
data.head()



plot_data = data.sample(5, random_state=0)
x = range(1,53)

plt.figure(figsize=(10,5))

for i,row in plot_data.iterrows():
    plt.plot(x,row)

plt.legend(plot_data.index)
plt.xlabel("Weeks")
plt.ylabel("Sales transactions per product")
plt.show()



data_train = data.iloc[:,:-1]
inputs = torch.Tensor(data_train.values).unsqueeze(1)
targets = data_train.shift(-1, axis="columns", fill_value=data.iloc[:,-1]).astype(dtype = "float32")
targets = torch.Tensor(targets.values)




class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)
        
    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = out.view(-1, self.hidden_size)
        out = self.output(out)
        
        return out, hidden




model = RNN(data_train.shape[1], 10, 1)
model




loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10000




losses = []
for i in range(1, epochs+1):
    
    hidden = None
    pred, hidden = model(inputs, hidden)
    
    loss = loss_function(pred, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if i%1000 == 0:
        print("epoch: ", i, "... Loss function: ", losses[-1])




x_range = range(len(losses))
plt.plot(x_range, losses)
plt.xlabel("epochs")
plt.ylabel("Loss function")
plt.show()




x_range = range(len(data))
target = data.iloc[:,-1].values.reshape(len(data),1)
plt.figure(figsize=(15,5))
plt.scatter(x_range[:20], target[:20])
plt.scatter(x_range[:20], pred.detach().numpy()[:20])
plt.legend(["Ground truth", "Prediction"])
plt.xlabel("Product")
plt.ylabel("Sales Transactions")
plt.xticks(range(0, 20))
plt.show()






