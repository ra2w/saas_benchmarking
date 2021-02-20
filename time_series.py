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



