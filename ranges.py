import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import altair as alt
import streamlit as st
from vega_datasets import data

# local imports
from schema import pretty_print as p
import ui
import filter
import schema


def plot_time_series_quantiles(main_metric,df_series):
    range_str = main_metric+'_range'
    a = df_series.groupby([range_str, 'hi_range']).describe()['CAGR'].reset_index()
    a.sort_values('hi_range')
    a[range_str] = a[range_str].astype(str)
    a = a.sort_values('hi_range', ascending=True)
    st.write(a)
    #st.line_chart(a[[range_str,'50%']])

    median_df = pd.DataFrame(data={range_str:a[range_str],'CAGR':a['50%'],'Type':'Median'})
    top_q_df = pd.DataFrame(data={range_str:a[range_str],'CAGR':a['75%'],'Type':'Top'})
    bot_q_df = pd.DataFrame(data={range_str: a[range_str], 'CAGR': a['25%'], 'Type': 'Bottom'})
    crm_df = df_series[df_series['ticker']=='CRM']
    crm_df = pd.DataFrame(data={range_str: crm_df[range_str],'CAGR': crm_df['CAGR'], 'Type': 'CRM'})
    b = pd.concat([median_df,top_q_df,bot_q_df,crm_df])
    #c = alt.Chart(a).mark_line(interpolate='step-after').encode(x=range_str, y='50%')
    c = alt.Chart(b).mark_line(interpolate='step-after').encode(x=range_str, y='CAGR',color='Type',
    strokeDash='Type')
    st.altair_chart(c, use_container_width=True)


def time_series_ranges(main_metric,m_df,selected_range,incr):
    st.write(f"**{main_metric} range **: ${selected_range[0]}M to ${selected_range[1]}M")

    (lo_range, hi_range) = selected_range
    hi_range = lo_range+incr if hi_range < lo_range + incr else hi_range

    df_series = filter.add_range_metrics(main_metric, m_df,
                                         filter.by_metric_range(main_metric, m_df, (lo_range, lo_range + incr)))
    df_series['hi_range'] = lo_range+incr
    df_series[main_metric+'_range']=f"({lo_range},{lo_range+incr})"

    for i in range(lo_range+incr,hi_range,incr):
        r = (i, i + incr)
        temp_df = filter.add_range_metrics(main_metric, m_df,
                                           filter.by_metric_range(main_metric, m_df, r))
        temp_df['hi_range'] = i + incr
        temp_df[main_metric + '_range'] = f"({i},{i + incr})"
        df_series = pd.concat([df_series, temp_df])

    plot_time_series_quantiles(main_metric,df_series)
    st.write(df_series)


def time_series_high_freq(ticker,m_df):
    #df = m_df[m_df['ticker']==ticker]
    df = m_df
    df = df.sort_values('t',ascending=True)
    end_arr = df['ARR']
    start_arr = df['ARR'].shift(4)
    df['YoY Growth'] = ((end_arr/start_arr)-1)*100
    #interpolate='step-after'
    brush = alt.selection(type='interval', encodings=['x','y'])
    base = alt.Chart(df).mark_point().encode(x='ARR', y='ARR growth',color='ticker')
    upper = base.encode(
        alt.X('ARR',scale=alt.Scale(domain=brush)),
        alt.Y('ARR growth', scale=alt.Scale(domain=brush))
    )
    lower = base.properties(
        height=60
    ).add_selection(brush)

    st.altair_chart(upper&lower, use_container_width=True)

def foobar():
    st.write('In foobar')
    source = data.sp500.url
    brush = alt.selection(type='interval', encodings=['x'])
    base = alt.Chart(source).mark_area().encode(
        x='date:T',
        y='price:Q'
    ).properties(
        width=600,
        height=200
    )
    upper = base.encode(
        alt.X('date:T', scale=alt.Scale(domain=brush))
    )
    lower = base.properties(
        height=60
    ).add_selection(brush)
    st.altair_chart(upper & lower, use_container_width=True)

def time_series_main(df):
    st.sidebar.write("## Analysis filters")
    df = filter.by_gtm(ui.input_gtm(schema.defined_gtm_types()), df)
    main_metric = 'ARR'
    st.title(f'{main_metric} Time-series analysis')
    """
    selected_range = controller.select_metric_range(main_metric,
                                                    minmax_range=(50, 1300), step=10,
                                                    range1=(200, 1000))

    time_series_ranges(main_metric,m_df,selected_range,incr=50)
    """
    time_series_high_freq('ZM', df)
    foobar()

# OLD STUFF

def add_range_metrics(metric, df, df_range):
    df_1 = pd.merge(df, df_range[['ticker', 't_x', 't_y']], on=['ticker'], how='inner')
    df_1['ebit_in_range'] = np.where((df_1['t'] >= df_1['t_x']) & (df_1['t'] < df_1['t_y']),
                                     df_1['EBIT'], 0)
    df_1['opex_in_range'] = np.where((df_1['t'] >= df_1['t_x']) & (df_1['t'] < df_1['t_y']),
                                     df_1['Opex'], 0)

    df_2 = pd.DataFrame(df_1.groupby('ticker').sum()[['ebit_in_range', 'opex_in_range']])
    df_range = pd.merge(df_range, df_2, on=['ticker'], how='inner')
    df_range['CAGR'] = cagr(df_range[metric + '(1)'], df_range[metric + '(0)'],
                            df_range['t_y'] - df_range['t_x'])
    df_range = df_range.dropna(axis=0, subset=['CAGR'])
    # Cap Ef. = NEW ARR(or Revenue) / opex in range
    df_range['Cap Ef.'] = np.round(
        ((df_range[metric + '(1)'] - df_range[metric + '(0)']) / df_range['opex_in_range']) * 100, 2)

    df_range['Years'] = (df_range['t_y'] - df_range['t_x']) / 4

    df_range = df_range.astype({'Years': 'float'})
    return df_range

# Get CAGR from quarterly number
def cagr_q(end_arr, start_arr, t_in_q):
    #assert t_in_q >= 0 or np.isnan(float(t_in_q))
    return ((end_arr / start_arr) ** (1 / t_in_q) - 1) * 400

def time_between_arr_ranges(end_arr,start_arr,r):
    return np.log(end_arr / start_arr) / np.log(1 + r/400)

def interpolate_arr(df,arr_i):
    df = df[(df['ARR_p'] < arr_i) & (df['ARR'] >= arr_i)]
    df['ARR_i'] = arr_i
    r = cagr_q(end_arr=df['ARR'],start_arr=df['ARR_p'],t_in_q=df['t']-df['t_p'])
    df['t_i'] = df['t']-time_between_arr_ranges(end_arr=df['ARR'],start_arr=df['ARR_i'],r=r)
    return df


def benchmark_arr_ranges(df):
    st.write('Range analysis')

    st.sidebar.write("## Analysis filters")
    df = filter.by_gtm(ui.input_gtm(schema.defined_gtm_types()), df)

    df[['ARR_p', 't_p']] = df.groupby('ticker')[['ARR', 't']].shift(1)
    df[['ARR_n', 't_n']] = df.groupby('ticker')[['ARR', 't']].shift(-1)

    #df = df[(df['ticker'] == 'DDOG') | (df['ticker'] == 'ESTC')]
    #df = df[df['ticker']=='PANW']
    df = df[['ticker', 'date','t', 'ARR', 'ARR_p', 't_p', 'ARR_n', 't_n']]
    st.table(df)
    arr_begin,arr_end = st.slider('Select ARR ranges', min_value=50, max_value=3000,
                           value=(100, 200), step=50, key="arr_range_slider")
    st.write(arr_begin,arr_end)

    select_cols = ['ticker','t_i','ARR_i']
    df_i1 = interpolate_arr(df, arr_begin)[select_cols]
    st.write(df_i1['ticker'].nunique())
    df_i2 = interpolate_arr(df, arr_end)[select_cols]
    st.write(df_i2['ticker'].nunique())
    df_i2['t_min'] = df_i2.groupby('ticker')['t_i'].transform('min') # Find the earliest time a company hits arr_end
    df_m = pd.merge(df_i1,df_i2,on='ticker',how='inner',suffixes=('_lo', '_hi'))
    df_m = df_m[(df_m['t_i_lo']<df_m['t_min']) & (df_m['t_i_hi']==df_m['t_min'])]
    df_m['CAGR']=cagr_q(df_m['ARR_i_hi'],df_m['ARR_i_lo'], df_m['t_i_hi']-df_m['t_i_lo'])
    st.table(df_m)

