import numpy as np
import pandas as pd

import gc
import sqlite3

pd.set_option('display.max_columns', None)

def merge_data():
    # File names
    data_path = "/Users/eddiewu/Downloads/gu_kelly_xiu/"
    features_file = "datashare/datashare.csv"
    returns_file = "data/tidy_finance_python.sqlite"
    macroecon_file = "PredictorData2023Monthly.csv"

    # Process firm level features
    df = (
        pd.read_csv(data_path + features_file)
        .assign(
            date = lambda x: pd.to_datetime(x['DATE'], format = '%Y%m%d').dt.to_period('M')
        )
        .drop(columns = ['DATE'])
    )

    # Process returns
    tidy_fin = sqlite3.connect(database = data_path + returns_file)

    df_returns = pd.read_sql_query(
      sql = "SELECT permno, date, ret_excess FROM crsp_monthly",
      con = tidy_fin,
      parse_dates = {"date"}
    )

    tidy_fin.close()

    df_returns['date'] = df_returns['date'].dt.to_period('M')

    # Merge returns to features
    df = df.merge(df_returns, how = 'left', on = ['permno', 'date'])
    del df_returns
    gc.collect()

    # Load macroeconomic predictors
    df_macro = pd.read_csv(data_path + macroecon_file)
    df_macro = (
        df_macro.assign(
            date = lambda x: pd.to_datetime(x['yyyymm'], format = '%Y%m').dt.to_period('M').shift(-1), # Lag all variables by 1 period
            Index = lambda x: pd.to_numeric(x['Index'].str.replace(',', ''), errors = 'coerce'),
            dp = lambda x: np.log(x['D12']) - np.log(x['Index']),
            ep = lambda x: np.log(x['E12']) - np.log(x['Index']),
            tms = lambda x: x['lty'] - x['tbl'],
            dfy = lambda x: x['BAA'] - x['AAA']
        )
        .rename(columns = {'b/m': 'bm'})
        .get(['date', 'dp', 'ep', 'bm', 'ntis', 'tbl', 'tms', 'dfy', 'svar'])
        .dropna()
    )

    # Merge macroeconomic predictors
    df = df.merge(df_macro, how = 'left', on = ['date'])

    # Rearrange the columns
    first_cols = ['permno', 'date', 'ret_excess']
    other_cols = list(df.columns.difference(first_cols, sort = False))
    df = df[first_cols + other_cols]

    # Check the merged dataframe
    print(df.shape)
    print(df.columns)
    print(df.info())
    print(df.head(10))
    print(df.tail(10))

    # Save to sql database
    df.to_csv(
        data_path + 'gkx_merged.csv',
        index = False
    )

    print("Merged data saved to csv.")


if __name__ == "__main__":
    merge_data()
