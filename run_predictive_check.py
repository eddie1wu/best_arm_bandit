### Compare the predictive MSE of lin reg, LASSO and FFNN

import numpy as np
import pandas as pd

from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import statsmodels.api as sm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from MLP import MLP
from utils import *

# Generate data
p = 100
n = 400
q = 5
sigma = 1
seed = 31

X, y = gen_linear_data(p, n, q, sigma, seed)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = seed)

# Lin reg
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)
model = sm.OLS(Y_train, X_train_sm).fit()
Y_pred = model.predict(X_test_sm)
mse = mean_squared_error(Y_test, Y_pred)
print(f"The OLS MSE is {mse: .5f}")

# LASSO
lasso = LassoCV(cv = 5, random_state = 0)
lasso.fit(X_train, Y_train)

Y_pred = lasso.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
print(f"The LASSO MSE is {mse: .5f}")


# Feedforward neural net
dataset = MyDataset( pd.DataFrame(np.column_stack( (X_train, Y_train) )) )
loader_train = DataLoader(dataset, batch_size = 8, shuffle=True) # Set up dataloader

input_size = X_train.shape[1]
hidden_size = 32
net = MLP(input_size, hidden_size)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)

for epoch in range(15):

    net.train()
    epoch_loss = 0

    for batch_idx, (feature, target) in enumerate(loader_train):

        optimizer.zero_grad()
        output = net(feature)

        loss = loss_fn(output, target.view(-1, 1))
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    # print(f'Epoch:{epoch}, Loss = {epoch_loss / len(loader_train):.5f}')

net.eval()
with torch.no_grad():
    Y_pred = net(torch.tensor(X_test, dtype=torch.float32))
mse = mean_squared_error(Y_test, Y_pred)
print(f"The FFNN MSE is {mse: .5f}")
