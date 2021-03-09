import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from vega_datasets import data

import filter
import schema
# local imports
import ui


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


# Get CAGR from quarterly number
def cagr_q(end_arr, start_arr, t_in_q):
    # assert t_in_q >= 0 or np.isnan(float(t_in_q))
    return ((end_arr / start_arr) ** (1 / t_in_q) - 1) * 400


def time_between_arr_ranges(end_arr, start_arr, r):
    return np.log(end_arr / start_arr) / np.log(1 + r / 400)


def interpolate_arr(arr, arr_p, t_in_q, arr_i):
    r = cagr_q(end_arr=arr, start_arr=arr_p, t_in_q=t_in_q)
    t_x = time_between_arr_ranges(end_arr=arr, start_arr=arr_i, r=r)
    return t_x


def compute_range_metrics_outer(df_c, arr_range, df_arr_index):
    c0, c1 = ([], [])
    ticker = df_c.name
    #st.write(ticker_md)
    # df_c = df_c.sort_index()
    # st.table(df_c)
    for i in arr_range:
        cagr = compute_range_metrics(ticker, df_c, i.left, i.right, df_arr_index)
        c0.append(i)
        c1.append(cagr)
    rdf = pd.DataFrame.from_dict(data={'ARR range': c0, 'CAGR': c1})
    rdf['ARR range'] = rdf['ARR range'].astype('str')
    rdf = rdf.set_index('ARR range')
    # st.table(rdf)
    return rdf


def compute_range_metrics(ticker, df_c, arr_begin, arr_end, df_arr_index):
    global c_time

    if arr_begin not in df_arr_index.index or arr_end not in df_arr_index.index:
        return np.NaN

    # st.table(df_c)

    t_begin = df_arr_index.loc[arr_begin, ticker]
    t_end = df_arr_index.loc[arr_end, ticker]

    if pd.isna(t_begin) or pd.isna(t_end):
        return np.NaN

    row = df_c.loc[(ticker, t_begin), :]
    a = timeit.default_timer()
    arr = row['ARR']
    arr_p = row['ARR_p']
    t_in_q = t_begin - row['t_p']

    t_x0 = t_begin - interpolate_arr(arr, arr_p, t_in_q, arr_begin)

    row = df_c.loc[(ticker, t_end), :]
    arr = row['ARR']
    arr_p = row['ARR_p']
    t_in_q = t_end - row['t_p']
    t_x1 = t_end - interpolate_arr(arr, arr_p, t_in_q, arr_end)

    # return cagr_q(arr_end, arr_begin,t_x1-t_x0)
    # return {'ARR range':pd.Interval(arr_begin,arr_end),
    #                  'CAGR':cagr_q(arr_end, arr_begin,t_x1-t_x0),
    #                  'Time (Quarters)':t_x1-t_x0}

    # return (pd.Interval(arr_begin,arr_end),
    #        cagr_q(arr_end, arr_begin,t_x1-t_x0),
    #        t_x1-t_x0)
    a = timeit.default_timer() - a
    c_time = c_time + a
    cagr = cagr_q(arr_end, arr_begin, t_x1 - t_x0)
    return cagr


def compute_arr_ranges(df):
    df['ARR_rnd_down'] = np.floor(df['ARR'] / 50) * 50
    df['ARR_p_rnd_up'] = np.ceil(df['ARR_p'] / 50) * 50

    def create_interval(row):
        if np.isnan(row['ARR_p']) or \
                np.isnan(row['ARR']) or \
                row['ARR_p_rnd_up'] > row['ARR_rnd_down']:
            return list(range(0, 0))

        return list(range(int(row['ARR_p_rnd_up']), int(row['ARR_rnd_down']) + 50, 50))

    df['range'] = df.apply(create_interval, axis=1).astype('object')
    return df


def benchmark_arr_ranges(df):
    global c_time
    st.write('Range analysis')
    ticker_md = {}

    st.sidebar.write("## Analysis filters")
    df = filter.by_gtm(ui.input_gtm(schema.defined_gtm_types()), df)

    df[['ARR_p', 't_p']] = df.groupby('ticker')[['ARR', 't']].shift(1)
    df[['ARR_n', 't_n']] = df.groupby('ticker')[['ARR', 't']].shift(-1)


    # df = df[(df['ticker'] == 'DDOG') | (df['ticker'] == 'ESTC')]
    # df = df[df['ticker']=='ESTC']
    df = df[['ticker', 'date', 't', 'ARR', 'ARR_p', 't_p']]
    arr_begin, arr_end = st.slider('Select ARR ranges', min_value=50, max_value=3000,
                                   value=(100, 200), step=50, key="arr_range_slider")

    df_arr_index = compute_arr_ranges(df).explode('range').groupby(['ticker', 'range'])['t'].min().unstack(0)
    df_plot = pd.DataFrame(columns=['ARR range', 'CAGR', 'Time (Quarters)'])
    arr_range = pd.interval_range(arr_begin, arr_end, freq=50)
    df_list = [df_plot]
    df.set_index(['ticker', 't'], inplace=True)
    # df = df.sort_index()

    c_time = 0
    starttime = timeit.default_timer()
    df_r = df.groupby('ticker') \
        .apply(compute_range_metrics_outer, arr_range=arr_range, df_arr_index=df_arr_index)
    df_r = df_r.unstack(1)
    st.write("The time difference is :", timeit.default_timer() - starttime)
    st.write(f"c_time = {c_time}")

    st.table(df_r)


# #####################################################################################
# OLD Stuff

def interpolate_arr_old(df, arr_i):
    '''
    With 'cond' we filter for two things:
        (1.) Only data points with +ve growth (i.e ARR > ARR_p or ARR(t) > ARR(t-1))
        (2.) ARR has equaled or broken our ARR threshold (i.e ARR > arr_i)
    Implications:
        (1.) We ignore data points where growth is negative, because that means at some point earlier the ARR already crossed our threshold
        (2.) t_i <= t (because arr_i <= ARR)
    '''
    df = df.reset_index()
    arr_i = 200
    st.table(df.reset_index(drop=True))
    st.write(df.index.get_loc(arr_i))
    df_t = df.iloc[df.index.get_loc(arr_i)]
    x = df_t.reset_index(drop=True)
    # x=x.astype('str')
    # st.table(x.columns)
    st.table(x)

    df.loc[cond] = arr_i
    df_x = df[cond]
    r = cagr_q(end_arr=df_x['ARR'], start_arr=df_x['ARR_p'], t_in_q=df_x['t'] - df_x['t_p'])
    t_x = time_between_arr_ranges(end_arr=df_x['ARR'], start_arr=df_x['ARR_i'], r=r)
    df.loc[cond, 't_i'] = df.loc[cond, 't_i'] - t_x
    return df


import timeit


def compute_range_metrics_old(df_c, arr_begin, arr_end):
    df_c['t_i'] = df_c['t']
    df_c = interpolate_arr_old(df_c, arr_begin)
    df_c = interpolate_arr_old(df_c, arr_end)

    # t_max = the last time point we're interested in because it represents the earliest time where ticker crossed arr_end
    t_max = df_c[df_c['ARR_i'] == arr_end]['t_i'].min()
    # t_min = latest time point where ticker crossed arr_begin
    t_min = df_c[(df_c['ARR_i'] == arr_begin) & (df_c['t_i'] < t_max)]['t_i'].max()
    df_c = df_c[(df_c['t_i'] >= t_min) & (df_c['t_i'] <= t_max)]

    assert df_c['ARR_i'].count() <= 2
    assert df_c[df_c['ARR_i'] == arr_begin]['ARR_i'].count() <= 1
    assert df_c[df_c['ARR_i'] == arr_end]['ARR_i'].count() <= 1

    # st.table(df_c)
    # st.write(t_min,t_max)
    return pd.Series({'ARR range': pd.Interval(arr_begin, arr_end),
                      'CAGR': cagr_q(arr_end, arr_begin, t_max - t_min),
                      'Time (Quarters)': t_max - t_min})


def benchmark_arr_ranges_old(df, arr_begin, arr_end):
    st.write('Range analysis')

    st.sidebar.write("## Analysis filters")

    #    df = filter.by_gtm(ui.input_gtm(schema.defined_gtm_types()), df)

    #    df[['ARR_p', 't_p']] = df.groupby('ticker')[['ARR', 't']].shift(1)
    #    df[['ARR_n', 't_n']] = df.groupby('ticker')[['ARR', 't']].shift(-1)

    # df = df[(df['ticker'] == 'DDOG') | (df['ticker'] == 'ESTC')]
    #    df = df[df['ticker']=='ESTC']
    #    df = df[['ticker', 'date','t', 'ARR', 'ARR_p', 't_p', 'ARR_n', 't_n']]
    #    arr_begin,arr_end = st.slider('Select ARR ranges', min_value=50, max_value=3000,
    #                          value=(100, 200), step=50, key="arr_range_slider")
    #    st.write(arr_begin,arr_end)

    def create_interval(row):
        if np.isnan(row['ARR_p']) or np.isnan(row['ARR']) or (row['ARR_p'] > row['ARR']):
            return pd.Interval(-1, 0)
        return pd.Interval(row['ARR_p'], row['ARR'])

    df['range'] = df.apply(create_interval, axis=1).astype('object')
    df = df.set_index('range')
    df = df.sort_index()

    # df[['ARR_index','ARR_p_index']] = df[['ARR','ARR_p']]
    # df = df.set_index(['ARR_p_index','ARR_index'])
    # st.table(df)
    # profiler = cProfile.Profile()
    # profiler.enable()

    df_plot = pd.DataFrame(columns=['ARR range', 'CAGR', 'Time (Quarters)'])
    arr_range = pd.interval_range(arr_begin, arr_end, freq=50)
    df_list = [df_plot]
    for i in arr_range:
        a0 = i.left
        a1 = i.right
        # starttime = timeit.default_timer()
        df_r = df.groupby('ticker') \
            .apply(compute_range_metrics_old, arr_begin=i.left, arr_end=i.right)
        # .dropna(axis=0, how='any')\
        # .sort_values('CAGR', ascending=False)
        # st.write("The time difference is :", timeit.default_timer() - starttime)
        df_list.append(df_r)
    df_plot = pd.concat(df_list)
    st.table(df_plot)

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('ncalls')
    # stats.print_stats()
