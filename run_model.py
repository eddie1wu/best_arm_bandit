from random import Random

import numpy as np
import pandas as pd

from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

from data_clean import run_prelim_analysis

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data_path = "/Users/eddiewu/Downloads/gu_kelly_xiu/"

# file = "1970-01_1991-01.csv"

file = 'gkx_clean.csv'

df = pd.read_csv(data_path + file)
df = df.sort_values(by = ['date', 'permno'])

run_prelim_analysis(df)

# # Create industry dummies
# print(f"Num unique industries is {df['sic2'].nunique()}")
# df = pd.get_dummies(df, columns = ['sic2'])
#
# # Train test split
# train_cut_off = "1975-11"
# test_cut_off = "1976-11"
# df_train, df_test = df.query("date < @train_cut_off"), df.query("@train_cut_off <= date < @test_cut_off")
#
# # Define X, y
# features = df.columns[3:]
# target = ['ret_excess']
#
# X_train, y_train = df_train[features], df_train[target]
# X_test, y_test = df_test[features], df_test[target]
#
# print("The training and testing details are:")
# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)
# print(df_train['date'].min(), df_train['date'].max())
# print(df_test['date'].min(), df_test['date'].max())
#
#
# # Model
# tss = TimeSeriesSplit()
#
# rf = RandomForestRegressor(max_depth = 5)
# rf.fit(X_train, y_train.values.ravel())
# y_pred = rf.predict(X_test)
#
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
#
# print(f"RF has out of sample mse {mse: .5f} and R^2 {r2: .5f}")
#
# print(rf.feature_importances_)
#
#




#
# model = LassoCV(cv = tss, max_iter = int(1e5))
#
# model.fit(X_train, y_train.values.ravel())
#
# y_pred = model.predict(X_test)
#
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
#
# print(f"LASSO has out of sample mse {mse: .5f} and R^2 {r2: .5f}")
# print(model.alpha_)
# print(model.intercept_)
#
# feature_filter = np.abs(model.coef_ ) > 0.001
#
# print(f"Important variables {X_train.columns[feature_filter].values}")