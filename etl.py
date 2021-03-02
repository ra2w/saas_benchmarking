import datetime

import numpy as np
import pandas as pd
import streamlit as st

global cols
cols = ['date',
        'arr__m',
        'subscription', 'services',
        'total_revenue', 'current_quarter_revenue_growth', 'growth_persistance', 'cost_of_revenue', 'gross_profit',
        'gross_margin', 'research_&_development', 'r&d_%', 'sales_&_marketing', 's&m_%', 'general_&_administrative',
        'g&a_%', 'total_operating_expense', 'net_income', 'deprecation_&_amoritization',
        'cash_from_operating_activites',
        'capex', 'free_cash_flow', 'free_cash_flow_%', 'cash', 'short_term_investments', 'long_term_debt',
        'short_term_debt', 'total_debt',
        'magic_number', 'ltm_cac_ratio', 'ltm_magic_number', 'current_cac_ratio', 'arr_per_employee__k',
        'net_dollar_retention', 'customers', 'other']

filter_cols = ['date', 'arr__m', 'total_revenue', 'growth_persistance',
               'gross_profit', 'gross_margin',
               'research_&_development', 'sales_&_marketing', 's&m_%', 'general_&_administrative',
               'total_operating_expense', 'deprecation_&_amoritization',
               'net_income', 'free_cash_flow', 'free_cash_flow_%', 'cash_from_operating_activites',
               'ltm_cac_ratio', 'net_dollar_retention', 'customers']

global standard

def convert_currency(val):
    """
    Convert the string number value to a float
     - Remove $
     - Remove commas
     - Convert to float type
    """
    try:
        new_val = val.replace(',', '').replace('$', '')
    except AttributeError:
        new_val = np.NaN
    return float(new_val)


def convert_date(val):
    date_time_str = val
    try:
        date_time_obj = datetime.datetime.strptime(date_time_str, '%Y_%b')
    except ValueError:
        date_time_obj = datetime.datetime.strptime(date_time_str, '%Y_%B')
    return date_time_obj.strftime("%m/%Y")


def convert_currency_k_to_mil(val):
    return float(convert_currency(val) / 1000)


def convert_float(val):
    try:
        new_val = val.replace(',', '').replace('x', '')
    except AttributeError:
        new_val = np.NaN
    return float(new_val)


def convert_percent(val):
    """
    Convert the percentage string to an actual floating point percent
    - Remove %
    """
    try:
        new_val = val.replace('%', '')
        new_val = float(new_val)
    except AttributeError:
        new_val = np.NaN

    return new_val


"""
#dollar_fmt = '{0:,.0f}'
dollar_fmt = '{0:.0f}'
percent_fmt = '{:.0f}%'
int_fmt = '{:d}'
float_fmt = '{:.2}'

col_formats = {'arr__m': dollar_fmt,
               'subscription': dollar_fmt,
               'services': dollar_fmt,
               'total_revenue': dollar_fmt,
               'ltm_rev': dollar_fmt,
               'current_quarter_revenue_growth': percent_fmt,
               'growth_persistance': percent_fmt,
               'cost_of_revenue': dollar_fmt,
               'gross_profit': dollar_fmt,
               'gross_margin': percent_fmt,
               'research_&_development': dollar_fmt,
               'r&d_%': percent_fmt,
               'sales_&_marketing': dollar_fmt,
               's&m_%': percent_fmt,
               'general_&_administrative': dollar_fmt,
               'g&a_%': convert_percent,
               'total_operating_expense': dollar_fmt,
               'net_income': dollar_fmt,
               'deprecation_&_amoritization': dollar_fmt,
               'cash_from_operating_activites': dollar_fmt,
               'capex': dollar_fmt,
               'free_cash_flow': dollar_fmt,
               'free_cash_flow_%': percent_fmt,
               'cash': dollar_fmt,
               'short_term_investments': dollar_fmt,
               'short_term_debt': dollar_fmt,
               'long_term_debt': dollar_fmt,
               'total_debt': dollar_fmt,
               'net_dollar_retention': percent_fmt,
               'customers': float_fmt,
               'magic_number': float_fmt,
               'ltm_magic_number': float_fmt,
               'ltm_cac_ratio': float_fmt,
               'current_cac_ratio': float_fmt,
               'arr_per_employee__k': dollar_fmt}
"""

col_types = {'date': convert_date,
             'arr__m': convert_currency,
             'subscription': convert_currency_k_to_mil,
             'services': convert_currency_k_to_mil,
             'total_revenue': convert_currency_k_to_mil,
             'current_quarter_revenue_growth': convert_percent,
             'growth_persistance': convert_percent,
             'cost_of_revenue': convert_currency_k_to_mil,
             'gross_profit': convert_currency_k_to_mil,
             'gross_margin': convert_percent,
             'research_&_development': convert_currency_k_to_mil,
             'r&d_%': convert_percent,
             'sales_&_marketing': convert_currency_k_to_mil,
             's&m_%': convert_percent,
             'general_&_administrative': convert_currency_k_to_mil,
             'g&a_%': convert_percent,
             'total_operating_expense': convert_currency_k_to_mil,
             'net_income': convert_currency_k_to_mil,
             'deprecation_&_amoritization': convert_currency_k_to_mil,
             'cash_from_operating_activites': convert_currency_k_to_mil,
             'capex': convert_currency_k_to_mil,
             'free_cash_flow': convert_currency_k_to_mil,
             'free_cash_flow_%': convert_percent,
             'cash': convert_currency_k_to_mil,
             'short_term_investments': convert_currency_k_to_mil,
             'short_term_debt': convert_currency_k_to_mil,
             'long_term_debt': convert_currency_k_to_mil,
             'total_debt': convert_currency_k_to_mil,
             'net_dollar_retention': convert_percent,
             'customers': convert_float,
             'magic_number': convert_float,
             'ltm_magic_number': convert_float,
             'ltm_cac_ratio': convert_float,
             'current_cac_ratio': convert_float,
             'arr_per_employee__k': convert_float}


def normalize_column_names(df):
    # remove $ and whitespace from column names; add _ between words, add __m for $m
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')',
                                                                                                           '').str.replace(
        '$', '_').str.replace('/', '_')
    # .str.replace('%','pct')
    return df


def add_ltm_rev(df):
    df['ltm_rev'] = df['total_revenue'] + df['total_revenue'].shift(1)+ \
                    df['total_revenue'].shift(2) + df['total_revenue'].shift(3)
    return df


def add_growth_rates(df):
    df['ltm_rev_g'] = np.NaN
    df['ntm_rev_g'] = np.NaN

    df['next_q_total_revenue'] = df['total_revenue'].shift(-1)
    # Year-over-Year quarterly revenue growth [r_q(i)/(r_q(i-4))]-1
    df['ltm_rev_g'] = (df['total_revenue'] / df['total_revenue'].shift(4) - 1) * 100

    # quarter over quarter revenue growth [(r_q(i)/r_q(i-1))]-1
    df['qoq_rev_g'] = (df['total_revenue'] / (df['total_revenue'].shift(1)) - 1) * 100

    # 1 year forward quarterly revenue growth [r_q(i+4)/r_q(i)]-1
    df['ntm_rev_g'] = (df['total_revenue'].shift(-4) / df['total_revenue'] - 1) * 100

    # 2 year forward quarterly revenue growth [r_q(i+8)/r_q(i)]^(1/2)-1
    df['ntm_rev_2yr_g'] = ((df['total_revenue'].shift(-8) / df['total_revenue']) ** (0.5) - 1) * 100

    df['rule_of_40'] = df['ltm_rev_g'] + df['free_cash_flow_%'] * 100

    df['new_arr'] = df['total_revenue'] - df['total_revenue'].shift(1)

    df['q_sales_ef'] = df['new_arr'] / df['sales_&_marketing'].shift(1)

    df['rol_s&m_%'] = df['s&m_%']
    for i in range(3):
        df['rol_s&m_%'] += df['s&m_%'].shift(i + 1)
    df['rol_s&m_%'] = df['rol_s&m_%'] / 4

    expected_rev_g = ((df['sales_&_marketing'] / (df['rol_s&m_%'].shift(1) / 100)) / df['total_revenue'].shift(
        1) - 1) * 100
    df['growth_ef'] = df['qoq_rev_g'] - expected_rev_g

    df['rol_growth_ef'] = df['growth_ef']
    df['rol_qoq_rev_g'] = df['qoq_rev_g']

    for i in range(3):
        df['rol_growth_ef'] += df['growth_ef'].shift(i + 1)
        df['rol_qoq_rev_g'] += df['rol_qoq_rev_g'].shift(i + 1)

    df['rol_growth_ef'] = df['rol_growth_ef'] / 4
    df['rol_qoq_rev_g'] = df['rol_qoq_rev_g'] / 4
    return df


def add_sales_efficiency(df):
    '''
    df['new_revenue']=df['total_revenue']-df['total_revenue'].shift(1)
    acc = 0
    # smooth out CAC estimate over last quarters
    for t in range(0,4):
        acc += df['new_revenue'].shift(t)/df['sales_&_marketing'].shift(t+1)
    acc = acc/4

    df['sales_ef']=acc
    '''
    return df


def set_ipo_timelines(ticker, df, ipo_month, ipo_year):
    df['year'] = df['date'].apply(lambda x: float(datetime.datetime.strptime(x, '%m/%Y').strftime("%Y"))) - ipo_year
    df['month'] = df['date'].apply(lambda x: float(datetime.datetime.strptime(x, '%m/%Y').strftime("%m")))

    # truncate year (e.g. 2020 --> 20) to reduce the length of the field
    df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(x, '%m/%Y').strftime("%m/%y"))

    df['error'] = (df['month'] - ipo_month) % 3
    if df['error'].sum() != 0:
        st.write("Error: Invalid quarter month ", df['month'])
    df['quarter'] = (df['month'] - ipo_month) / 3
    df['t'] = df['year'] * 4 + df['quarter']  # y=quarters since IPO. +x = x quarters after IPO, -x = x quarters before IPO
    df.drop(['error', 'month', 'year'], 1, inplace=True)
    return df


def read_company_csv(filename):
    return pd.read_csv(filename)


def load_financials(ticker, ipo_month, ipo_year):
    filename = 'data/' + ticker + '.htm.csv'

    df = read_company_csv(filename)
    # Normalize columns
    # remove $ and whitespace from column names; add _ between words, add __m for $m
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_',regex=False).str.replace('(', '',regex=False).str.replace(')',
                                                                                                           '',regex=False).str.replace(
        '$', '_',regex=False).str.replace('/', '_',regex=False)

    df = df.set_index('fiscal_year')
    # Drop empty rows
    df.dropna(axis='rows', how='all', inplace=True)
    # Drop empty columns
    df.dropna(axis='columns', how='all', inplace=True)
    # Transpose
    df = df.T
    # Normalize transpose columns
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_',regex=False).str.replace('(', '',regex=False).str.replace(')',
                                                                                                           '',regex=False).str.replace(
        '$', '_',regex=False).str.replace('/', '_',regex=False)
    df = df[df['total_revenue'].notna()]

    df.reset_index(inplace=True)
    df.rename(columns={'index': 'date'}, inplace=True)

    missing_cols = [v for v in cols if v not in df.columns]
    for v in missing_cols:
        # print(ticker,":adding ",v," which was not found!")
        df[v] = [np.NaN] * df.index.size
    new_cols = [v for v in df.columns if v not in cols]
    for v in new_cols:
        st.write(ticker, "WARNING!: Found ", v, " missing in master column set!")

    # set column types
    for c in df.columns:
        if c in col_types:
            df[c] = df[c].apply(col_types[c])

    df = df[filter_cols]  ## XXX REMOVE!
    df = set_ipo_timelines(ticker, df, ipo_month, ipo_year)

    df['s&m_%'] = df['sales_&_marketing'] / df['total_revenue'] * 100
    #df = add_growth_rates(df)
    #df = add_sales_efficiency(df)
    #df = add_ltm_rev(df)

    return df


# Company db is a dict with the following schema:
# [ticker]: {name | sector | gtm | ipo_year | ipo_month}
# Examples:
# ZM: {apps | bottom_up| 2019 |jan}
@st.cache(suppress_st_warning=True)
def load_companies(_meta_df):
    _c = {}
    dfs_to_concat = []
    tickers_to_concat = []

    st.write("ETL Hello3")
    cnt = 0
    my_bar = st.progress(0)
    ticker_list = list(_meta_df['ticker'])
    for ticker in ticker_list:
        _c[ticker] = load_financials(ticker,
                                     int(_meta_df[_meta_df['ticker'] == ticker]['ipo_month']),
                                     float(_meta_df[_meta_df['ticker'] == ticker]['ipo_year']))

        _meta_df.loc[_meta_df['ticker'] == ticker, 'earliest'] = _c[ticker]['t'].min()
        _meta_df.loc[_meta_df['ticker'] == ticker, 'latest'] = _c[ticker]['t'].max()
        _c[ticker].set_index('t', inplace=True)
        _c[ticker].columns.names = ['']
        #        st.table(_c[ticker])
        dfs_to_concat.append(_c[ticker])
        tickers_to_concat.append(ticker)
        cnt = cnt + 1
        my_bar.progress(cnt / len(ticker_list))
    # _m is the master dataframe with all companies merged indexed to a common ipo timeline
    # t=0 is the last quarter before IPO

    _m = pd.concat(dfs_to_concat, axis=0, keys=tickers_to_concat)
    return _m

@st.cache(suppress_st_warning=True)
def load_companies_refactored(_meta_df):
    _c = {}
    dfs_to_concat = []
    tickers_to_concat = []

    cnt = 0
    my_bar = st.progress(0)
    ticker_list = list(_meta_df['ticker'])
    for ticker in ticker_list:
        _c[ticker] = load_financials(ticker,
                                     int(_meta_df[_meta_df['ticker'] == ticker]['ipo_month']),
                                     float(_meta_df[_meta_df['ticker'] == ticker]['ipo_year']))

        _c[ticker].set_index('t', inplace=True)
        _c[ticker].columns.names = ['']
        cnt = cnt + 1
        my_bar.progress(cnt / len(ticker_list))
    return _c
