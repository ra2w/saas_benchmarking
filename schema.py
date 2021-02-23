import pandas as pd
import streamlit as st

# local imports
import etl



dollar_fmt = '{0:,.0f}M'
percent_fmt = '{:.0f}%'
int_fmt = '{:d}'
float_fmt = '{:.1f}'

col_field_formats = {'ARR': dollar_fmt,
                     'ARR growth':percent_fmt,
                     'ARR(0)':dollar_fmt,
                     'ARR(1)':dollar_fmt,
                     'Years':float_fmt,
                     'CAGR':percent_fmt,
                     'LTM Rev': dollar_fmt,
                     'LTM Rev(0)':dollar_fmt,
                     'LTM Rev(1)':dollar_fmt,
                     'Rev growth':percent_fmt,
                     'GM': percent_fmt,
                     'S&M': dollar_fmt,
                     'S&M%': percent_fmt,
                     'Opex': dollar_fmt,
                     'LTM Opex': dollar_fmt,
                     'EBIT': dollar_fmt,
                     'LTM EBIT': dollar_fmt,
                     'Cap Ef.':float_fmt,
                     'NI': dollar_fmt,
                     'FCF': dollar_fmt,
                     'FCF%:': percent_fmt,
                     'Op. CF': dollar_fmt,
                     'LTM CAC': float_fmt,
                     'NDR': percent_fmt,
                     't_x':int_fmt,
                     't_y':int_fmt,
                     '':float_fmt
                     }



col_plot_labels = {'ARR': "($M)",
                     'ARR growth':"%",
                     'ARR(0)':"($M)",
                     'ARR(1)':"($M)",
                     'CAGR':"%",
                     'LTM Rev':"($M)",
                     'LTM Rev(0)':"($M)",
                     'LTM Rev(1)':"($M)",
                     'Rev growth':"%",
                     'GM': "%",
                     'S&M': "($M)",
                     'S&M%': "%",
                     'Opex': "($M)",
                     'LTM Opex': "($M)",
                     'EBIT': "($M)",
                     'LTM EBIT': "($M)",
                     'Cap Ef.':"",
                     'NI': "($M)",
                     'FCF': "($M)",
                     'FCF%:': "%",
                     'Op. CF': "($M)",
                     'LTM CAC': "",
                     'NDR': "%",
                     }


col_names_transform = {'arr__m': 'ARR',
                       'ltm_rev':'LTM Rev',
                       'gross_profit': 'GP',
                       'gross_margin': 'GM',
                       'sales_&_marketing': 'S&M',
                       's&m_%': 'S&M%',
                       'total_operating_expense': 'Opex',
                       'net_income': 'NI',
                       'free_cash_flow': 'FCF',
                       'free_cash_flow_%': 'FCF%',
                       'cash_from_operating_activites': 'Op. CF',
                       'ltm_cac_ratio': 'LTM CAC',
                       'net_dollar_retention': 'NDR'}



rev = ['ARR','LTM Rev']
growth = ['ARR growth','Rev growth']
expense = ['S&M', 'S&M%','LTM Opex','Opex']
profit = ['GM','NI','FCF', 'FCF%', 'Op. CF', 'LTM EBIT','EBIT']
efficiency = ['LTM CAC', 'NDR']
non_metrics = ['date']
projected_cols = non_metrics + rev + growth + expense + profit + efficiency


metric_to_column_name = {'ARR':'ARR',
                        'LTM Revenue':'LTM Rev',
                         'Sales & Marketing':'S&M',
                         'Sales & Marketing %':'S&M%',
                         'Operating Expense':'Opex',
                         'Gross Margin':'GM',
                         'Free Cash Flows':'FCF',
                         'Free Cash Flow %':'FCF%',
                         'Operating Cash Flows':'Op. CF',
                         'LTM CAC':'LTM CAC',
                         'Net Dollar Retention':'NDR'
                 }


def defined_gtm_types():
    return ['enterprise','bottom_up_enterprise','top_down_enterprise','smb','all']

def tickers_by_gtm(gtm):
    gtm_filters = {
        'enterprise': "meta_df['gtm'] != 'smb'",
        'bottom_up_enterprise': "meta_df['gtm'] == 'bottom_up_saas'",
        'top_down_enterprise': "meta_df['gtm'] == 'top_down_saas'",
        'smb': "meta_df['gtm'] == 'smb'",
        'all': "True"
    }
    return meta_df[eval(gtm_filters[gtm])]['ticker']


#metrics = ['ARR','ARR growth','LTM Rev','Rev growth','GM','S&M%','LTM CAC','NDR']
metrics = rev+growth+profit+efficiency
analysis_types = ['Point in Time', 'Company', 'Revenue range']

# Print stylized dataframe
def pretty_print(df):
    return df.style.format(col_field_formats, na_rep="-")

def load_metadata(filename = 'data/company_db.csv'):
    global meta_df
    tickers_to_exclude = ['SYMC', 'GDDY', 'CARB', 'RP','NTNX','RNG','SWI']
    meta_df = pd.read_csv(filename)
    for i in tickers_to_exclude:
        meta_df = meta_df[meta_df.ticker != i]
    return len(meta_df)

def append_derived_ltm_metrics(df):
    df['LTM Rev'] = df['total_revenue'] + df['total_revenue'].shift(1) + \
                    df['total_revenue'].shift(2) + df['total_revenue'].shift(3)

    df['LTM Opex'] = df['Opex'] + df['Opex'].shift(1) + \
                     df['Opex'].shift(2) + df['Opex'].shift(3)

    df['Rev growth'] = (df['LTM Rev'] / df['LTM Rev'].shift(4) - 1) * 100
    df['LTM EBIT'] = df['LTM Rev'] - df['LTM Opex']
    return df

def append_derived_metrics(df):
    df['ARR growth'] = (df['ARR'] / df['ARR'].shift(4) - 1) * 100
    df['EBIT'] = df['total_revenue']-df['Opex'] # Quarterly EBIT
    return append_derived_ltm_metrics(df)

def transform_dataset(df):
    df = df.rename(columns=col_names_transform)  # Make column names easier to read and understand
    df = append_derived_metrics(df)
    df = df[projected_cols]
    return df

def load_dataset():
    dict_of_dfs = etl.load_companies_refactored(meta_df)

    dict_of_dfs_post = {ticker: transform_dataset(df) for (ticker,df) in dict_of_dfs.items()}
    _m = pd.concat(dict_of_dfs_post, axis=0, keys=dict_of_dfs_post.keys())
    _m.index.set_names(['ticker','t'], inplace=True)
    _m = _m.reset_index()
    return _m





