import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

def run_prelim_analysis(df):
    # Count the number of days for each stock
    date_counts = df.groupby('permno')['date'].nunique()
    print(date_counts.describe())
    plt.hist(date_counts, bins=15)
    plt.show()

    # Count the number of stocks for each period
    stock_counts = df.groupby('date')['permno'].nunique()
    print(stock_counts.describe())
    plt.hist(stock_counts, bins=15)
    plt.show()

    # Count the number of non-missing rows for each variable for each date
    var_count_per_period = df.groupby('date').count()
    num_per_plot = 20
    for i in range(int(np.ceil(var_count_per_period.shape[1] / num_per_plot))):
        var_count_per_period.iloc[:, i * num_per_plot:(i + 1) * num_per_plot].plot()
        df.groupby('date')['permno'].nunique().plot()
        plt.show()

def clean_data():

    df = pd.read_csv(data_path + file)

    print("DONE loading data")
    print(df.columns)

    ### Preprocess stock characteristics
    # Rank each characteristics column
    def cross_section_rank(x):

        x = x.rank(method = 'min', na_option = 'keep')

        if not x.isna().all():
            x = 2 * (x - x.min()) / (x.max() - x.min()) - 1

        return x

    def fill_median(x):

        if not x.isna().all():
            x = x.fillna(x.median())

        return x

    # Rank for each cross section, fill missing with cross section median, then fill missing with 0
    stock_rank = df.columns[3:97]
    df[stock_rank] = (
        df.groupby("date")[stock_rank]
        .transform(lambda x: fill_median( cross_section_rank(x) ))
        .fillna(0)
    )

    print("DONE ranking and filling na")

    # Drop the few observations without any return observation
    df = df.dropna(subset = ['ret_excess'])

    # Inspect the new df
    # run_prelim_analysis(df)

    # Fill missing sic2 with 0, i.e. treat missing as one industry (or just drop na)
    df['sic2'] = df['sic2'].fillna(0)

    df.to_csv(
        data_path + "gkx_clean.csv",
        index = False
    )


if __name__ == "__main__":

    # Load data
    data_path = "/Users/eddiewu/Downloads/gu_kelly_xiu/"
    file = "gkx_merged.csv"

    # Clean data
    clean_data()




