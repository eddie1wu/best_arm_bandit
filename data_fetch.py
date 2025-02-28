import numpy as np
import pandas as pd
import pandas_datareader as pdr

import sqlite3
from sqlalchemy import create_engine

import os

# Download Fama-French 3-factor data
def get_ff3_monthly():

    ff3_monthly_raw = pdr.DataReader(
        name = "F-F_Research_Data_Factors",
        data_source = "famafrench",
        start = start_date,
        end = end_date
    )[0]

    ff3_monthly = (
        ff3_monthly_raw
        .divide(100)
        .reset_index(names = "date")
        .assign(date = lambda x: pd.to_datetime(x["date"].astype(str)))
        .rename(str.lower, axis = "columns")
        .rename(columns = {"mkt-rf": "mkt_excess"})
    )

    tidy_fin = sqlite3.connect(database=data_path + "data/tidy_finance_python.sqlite")

    (ff3_monthly.to_sql(
        name = "ff3_monthly",
        con = tidy_fin,
        if_exists = "replace",
        index = False)
    )

    tidy_fin.execute("VACUUM")
    tidy_fin.close()


# Download CRSP returns data
def get_crsp_returns():
    connection_string = (
        "postgresql+psycopg2://"
        f"{'USERNAME'}:{'PASSWORD'}"
        "@wrds-pgdata.wharton.upenn.edu:9737/wrds"
    )

    wrds = create_engine(connection_string, pool_pre_ping=True)

    crsp_monthly_query = (
        "SELECT msf.permno, date_trunc('month', msf.mthcaldt)::date AS date, "
        "msf.mthret AS ret, msf.shrout, msf.mthprc AS altprc "
        # "ssih.primaryexch, ssih.siccd "
        "FROM crsp.msf_v2 AS msf "
        # "INNER JOIN crsp.stksecurityinfohist AS ssih "
        # "ON msf.permno = ssih.permno AND "
        # "ssih.secinfostartdt <= msf.mthcaldt AND "
        # "msf.mthcaldt <= ssih.secinfoenddt "
        f"WHERE msf.mthcaldt BETWEEN '{start_date}' AND '{end_date}' "
        # "AND ssih.sharetype = 'NS' "
        # "AND ssih.securitytype = 'EQTY' "
        # "AND ssih.securitysubtype = 'COM' "
        # "AND ssih.usincflg = 'Y' "
        # "AND ssih.issuertype in ('ACOR', 'CORP') "
        # "AND ssih.primaryexch in ('N', 'A', 'Q') "
        # "AND ssih.conditionaltype in ('RW', 'NW') "
        # "AND ssih.tradingstatusflg = 'A'"
    )

    crsp_monthly = (pd.read_sql_query(
        sql = crsp_monthly_query,
        con = wrds,
        # dtype = {"permno": int, "siccd": int},
        dtype={"permno": int},
        parse_dates = {"date"}
    ).assign(
        shrout = lambda x: x["shrout"] * 1000)
    )

    crsp_monthly = (
        crsp_monthly
        .assign(mktcap = lambda x: x["shrout"] * x["altprc"] / 1000000)
        .assign(mktcap = lambda x: x["mktcap"].replace(0, np.nan))
    )

    tidy_fin = sqlite3.connect(database=data_path + "data/tidy_finance_python.sqlite")

    ff3_monthly = pd.read_sql_query(
        sql = "SELECT date, rf FROM ff3_monthly",
        con = tidy_fin,
        parse_dates = {"date"}
    )

    crsp_monthly = (
        crsp_monthly
        .merge(ff3_monthly, how = "left", on = "date")
        .assign(ret_excess = lambda x: x["ret"] - x["rf"])
        .assign(ret_excess = lambda x: x["ret_excess"].clip(lower = -1))
        .drop(columns = ["rf"])
    )

    crsp_monthly = (crsp_monthly
                    .dropna(subset = ["ret_excess", "mktcap"])
    )

    (crsp_monthly
     .to_sql(name = "crsp_monthly",
             con = tidy_fin,
             if_exists = "replace",
             index = False)
     )

    tidy_fin.execute("VACUUM")
    tidy_fin.close()


if __name__ == "__main__":

    # Set start and end dates
    start_date = "01/31/1957"
    end_date = "12/31/2021"

    # Set up the database
    data_path = "/Users/eddiewu/Downloads/gu_kelly_xiu/"
    if not os.path.exists(data_path + "data"):
        os.makedirs(data_path + "data")

    # Get Fama French data
    get_ff3_monthly()

    # Get CRSP returns
    get_crsp_returns()
