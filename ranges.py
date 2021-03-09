import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from vega_datasets import data
import functools

import filter
import schema
# local imports
import ui
import finance as fin



def plot_time_series_quantiles(main_metric, df_series):
    range_str = main_metric + '_range'
    a = df_series.groupby([range_str, 'hi_range']).describe()['CAGR'].reset_index()
    a.sort_values('hi_range')
    a[range_str] = a[range_str].astype(str)
    a = a.sort_values('hi_range', ascending=True)
    st.write(a)
    # st.line_chart(a[[range_str,'50%']])

    median_df = pd.DataFrame(data={range_str: a[range_str], 'CAGR': a['50%'], 'Type': 'Median'})
    top_q_df = pd.DataFrame(data={range_str: a[range_str], 'CAGR': a['75%'], 'Type': 'Top'})
    bot_q_df = pd.DataFrame(data={range_str: a[range_str], 'CAGR': a['25%'], 'Type': 'Bottom'})
    crm_df = df_series[df_series['ticker'] == 'CRM']
    crm_df = pd.DataFrame(data={range_str: crm_df[range_str], 'CAGR': crm_df['CAGR'], 'Type': 'CRM'})
    b = pd.concat([median_df, top_q_df, bot_q_df, crm_df])
    # c = alt.Chart(a).mark_line(interpolate='step-after').encode(x=range_str, y='50%')
    c = alt.Chart(b).mark_line(interpolate='step-after').encode(x=range_str, y='CAGR', color='Type',
                                                                strokeDash='Type')
    st.altair_chart(c, use_container_width=True)


def time_series_ranges(main_metric, m_df, selected_range, incr):
    st.write(f"**{main_metric} range **: ${selected_range[0]}M to ${selected_range[1]}M")

    (lo_range, hi_range) = selected_range
    hi_range = lo_range + incr if hi_range < lo_range + incr else hi_range

    df_series = filter.add_range_metrics(main_metric, m_df,
                                         filter.by_metric_range(main_metric, m_df, (lo_range, lo_range + incr)))
    df_series['hi_range'] = lo_range + incr
    df_series[main_metric + '_range'] = f"({lo_range},{lo_range + incr})"

    for i in range(lo_range + incr, hi_range, incr):
        r = (i, i + incr)
        temp_df = filter.add_range_metrics(main_metric, m_df,
                                           filter.by_metric_range(main_metric, m_df, r))
        temp_df['hi_range'] = i + incr
        temp_df[main_metric + '_range'] = f"({i},{i + incr})"
        df_series = pd.concat([df_series, temp_df])

    plot_time_series_quantiles(main_metric, df_series)
    st.write(df_series)


def time_series_high_freq(ticker, m_df):
    # df = m_df[m_df['ticker']==ticker]
    df = m_df
    df = df.sort_values('t', ascending=True)
    end_arr = df['ARR']
    start_arr = df['ARR'].shift(4)
    df['YoY Growth'] = ((end_arr / start_arr) - 1) * 100
    # interpolate='step-after'
    brush = alt.selection(type='interval', encodings=['x', 'y'])
    base = alt.Chart(df).mark_point().encode(x='ARR', y='ARR growth', color='ticker')
    upper = base.encode(
        alt.X('ARR', scale=alt.Scale(domain=brush)),
        alt.Y('ARR growth', scale=alt.Scale(domain=brush))
    )
    lower = base.properties(
        height=60
    ).add_selection(brush)

    st.altair_chart(upper & lower, use_container_width=True)


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




def compute_range_metrics_outer(df_c, arr_range, df_arr_index):
    c0, c1 = ([], [])
    ticker = df_c.name
    #st.write(ticker_md)
    # df_c = df_c.sort_index()
    # st.table(df_c)
    for i in arr_range:
        cagr = compute_cagr_for_marker_range(ticker, df_c, i.left, i.right, df_arr_index)
        c0.append(i)
        c1.append(cagr)
    rdf = pd.DataFrame.from_dict(data={'ARR range': c0, 'CAGR': c1})
    rdf['ARR range'] = rdf['ARR range'].astype('str')
    rdf = rdf.set_index('ARR range')
    return rdf


def compute_cagr_for_marker_range(ticker, df_c, arr_ix0, arr_ix1, df_metric_index):
    global c_time

    r0_num = arr_marker_to_row_num(ticker, df_metric_index, arr_ix0)
    r1_num = arr_marker_to_row_num(ticker, df_metric_index, arr_ix1)

    if pd.isna(r0_num) or pd.isna(r1_num):
        return np.NaN

    a = timeit.default_timer()
    row0 = df_c.loc[(ticker, r0_num), :]
    row1 = df_c.loc[(ticker, r1_num), :]

    r0 = fin.cagr_q(end_arr=row0['ARR'],start_arr=row0['ARR_p'],t_in_q = row0['t'] - row0['t_p'])
    t_ix0 = row0['t'] - fin.time_between_arr_ranges(end_arr=row0['ARR'],start_arr=arr_ix0,r=r0)
    r1 = fin.cagr_q(end_arr=row1['ARR'], start_arr=row1['ARR_p'], t_in_q=row1['t'] - row1['t_p'])
    t_ix1 = row1['t'] - fin.time_between_arr_ranges(end_arr=row1['ARR'],start_arr=arr_ix1,r=r1)


    a = timeit.default_timer() - a
    c_time = c_time + a
    range_cagr = fin.cagr_q(arr_ix1, arr_ix0, t_ix1 - t_ix0)
    return range_cagr


def create_metric_markers(df,metric='ARR'):
    if metric != 'ARR':
        raise NotImplementedError

    m0 = metric+'_lo'
    m1 = metric+'_hi'
    df[m1] = np.floor(df['ARR'] / 50) * 50
    df[m0] = np.floor((df['ARR_p']+50) / 50) * 50

    def create_interval(row):
        if np.isnan(row['ARR_p']) or \
                np.isnan(row['ARR']) or \
                row[m0] > row[m1]:
            return list(range(0, 0))

        return list(range(int(row[m0]), int(row[m1]) + 50, 50))

    return df.apply(create_interval, axis=1).astype('object')


def marker_to_row_num(metric_name,ticker,df_metric_index,metric_value):
    if metric_name != 'ARR':
        raise NotImplementedError

    if metric_value not in df_metric_index.index:
        return np.NaN

    return df_metric_index.loc[metric_value, ticker]

arr_marker_to_row_num = functools.partial(marker_to_row_num, 'ARR')


import timeit

def benchmark_arr_ranges(df):
    global c_time
    st.write('Range analysis')
    ticker_md = {}

    st.sidebar.write("## Analysis filters")
    df = filter.by_gtm(ui.input_gtm(schema.defined_gtm_types()), df)

    df[['ARR_p', 't_p']] = df.groupby('ticker')[['ARR', 't']].shift(1)
    df[['ARR_n', 't_n']] = df.groupby('ticker')[['ARR', 't']].shift(-1)


    # df = df[(df['ticker'] == 'DDOG') | (df['ticker'] == 'ESTC')]
    #df = df[df['ticker']=='CRM']
    df = df[['ticker', 'date', 't', 'ARR', 'ARR_p', 't_p']]
    df = df.sort_values(['ticker','t'],ascending=True)
    df['row_num'] = range(len(df))

    arr_begin, arr_end = st.slider('Select ARR ranges', min_value=50, max_value=3000,
                                   value=(100, 200), step=50, key="arr_range_slider")

    df['arr_marker'] = create_metric_markers(df)

    df_marker_index = df.explode('arr_marker').groupby(['ticker', 'arr_marker'])['row_num'].min().unstack(0)

    df_plot = pd.DataFrame(columns=['ARR range', 'CAGR', 'Time (Quarters)'])
    arr_range = pd.interval_range(arr_begin, arr_end, freq=50)
    df_list = [df_plot]
    df.set_index(['ticker', 'row_num'], inplace=True)
    # df = df.sort_index()

    c_time = 0
    starttime = timeit.default_timer()
    df_r = df.groupby('ticker') \
        .apply(compute_range_metrics_outer, arr_range=arr_range, df_arr_index=df_marker_index)
    df_r = df_r.unstack(1)
    st.write("The time difference is :", timeit.default_timer() - starttime)
    st.write(f"c_time = {c_time}")

    st.table(df_r)

