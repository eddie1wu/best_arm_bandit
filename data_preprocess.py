import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gc

# pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Load data
data_path = "/Users/eddiewu/Downloads/gu_kelly_xiu/"
file = "gkx_clean.csv"

df = pd.read_csv(data_path + file)

start = "2000-01"
end = "2019-12"
df = df.query("date >= @start and date <= @end")
gc.collect()

print(df.shape)
print(df.isna().sum())

df.to_csv(data_path + "gkx_subset.csv", index=False)
