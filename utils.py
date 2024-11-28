import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

def gen_linear_data(x_dim, n_sample, q, sigma, random_state = 4):
    np.random.seed(random_state)

    X = np.random.uniform(size = (n_sample, x_dim))
    fX = 1
    for i in range(q):
        fX += (-1) ** i * 0.5 * X[:, i]
    y = fX + np.random.normal(scale = sigma, size = fX.shape)

    return X, y

def gen_friedman_data(x_dim, n_sample, sigma, random_state = 4):
    np.random.seed(random_state)

    X = np.random.uniform(size=(n_sample, x_dim))
    fX = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]
    y = fX + np.random.normal(scale = sigma, size = fX.shape)

    return X, y

def plot_posterior_mean(out, time, q):
    time_range = np.arange(time)
    out_array = np.array(out)

    # Plot each variable
    plt.figure()
    for i in range(out_array.shape[1]):
        if i <= q-1:
            plt.plot(time_range, out_array[:, i], color = 'red', alpha = 1)
        else:
            plt.plot(time_range, out_array[:, i], color = 'blue', alpha = 0.4)
    plt.grid(True)
    plt.title("Posterior probabilities over Time")
    plt.xlabel("Time step")
    plt.ylabel("pi_k")
    plt.yticks(np.arange(0, 1.1, 0.05))
    # plt.savefig('figure.png')
    plt.show()

def plot_statistic(result_dict, title, save_name):
    # Extract methods and values for plotting
    methods = list(result_dict.keys())
    values = list(result_dict.values())

    plt.bar(methods, values)
    plt.xticks(rotation=10)
    plt.xlabel('Algorithm')
    plt.ylabel('Time to Convergence (0.99)')
    plt.title(title)

    plt.savefig(save_name, dpi=400)
    plt.close()

class MyDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = torch.tensor(row.values[:-1], dtype=torch.float32)  # All columns except last are features
        label = torch.tensor(row.values[-1], dtype=torch.float32)  # Last column is label, integer
        return features, label
