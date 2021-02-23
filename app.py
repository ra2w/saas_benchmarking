import importlib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# Custom packages
import etl
import schema
import streamlit_ui
import filter
import benchmarking
import ui
import time_series


def sklearn_vif(exogs, data):
    for c in exogs:
        data = data[data[c].notna()]

    # initialize dictionaries
    vif_dict, tolerance_dict = {}, {}

    # form input data for each exogenous variable
    for exog in exogs:
        not_exog = [i for i in exogs if i != exog]
        X, y = data[not_exog], data[exog]

        # extract r-squared from the fit
        r_squared = LinearRegression().fit(X, y).score(X, y)

        # calculate VIF
        vif = 1 / (1 - r_squared)
        vif_dict[exog] = vif

        # calculate tolerance
        tolerance = 1 - r_squared
        tolerance_dict[exog] = tolerance

    # return VIF DataFrame
    df_vif = pd.DataFrame({'VIF': vif_dict, 'Tolerance': tolerance_dict})

    return df_vif

def _not_inplemented(m_df):
    st.title("Coming soon!")


def ui_toplevel_sidebar_navigation(m_df):
    pages_dispatch = {'Timeline based': benchmarking.benchmarking_main,
                      'Metric range based': _not_inplemented,
                      'Regression analysis':_not_inplemented}

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Benchmarking:", list(pages_dispatch.keys()))
    return pages_dispatch[selection](m_df)

def main():
    num_companies = schema.load_metadata()
    with st.spinner('Loading companies and creating dataset...'):
        m = schema.load_dataset()

    st.success(f'{num_companies} companies loaded')
    ui_toplevel_sidebar_navigation(m)
    # Filter based on columns and timelines


### BEGIN ##########################################################


def single_company_worker(m_df):
    ticker = 'CRWD'
    st.title(f'Single Company Analysis')

    #all_companies = list(_meta_df['name'])

    #option = st.selectbox('Company Name', all_companies)
    #ticker = _meta_df.loc[_meta_df['name'] == option].iloc[0]['ticker']
    #st.write(f'You selected {option} [ticker:{ticker}]')
    #m_df
    #df = m_df.loc[[ticker]]
    #st.dataframe(df)
    # st.write(_c[ticker][basic])



# Hack to fix watchdog related bug where file changes don't seem to trigger rerun
if st.sidebar.button("Reload modules"):
    importlib.reload(etl)
    importlib.reload(schema)
    importlib.reload(ui)
    importlib.reload(streamlit_ui)
    importlib.reload(filter)
    importlib.reload(benchmarking)
    importlib.reload(time_series)

    st.experimental_rerun()

if __name__ == "__main__":
    main()
