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
    '''
    With 'cond' we filter for two things:
        (1.) Only data points with +ve growth (i.e ARR > ARR_p or ARR(t) > ARR(t-1))
        (2.) ARR has equaled or broken our ARR threshold (i.e ARR > arr_i)
    Implications:
        (1.) We ignore data points where growth is negative, because that means at some point earlier the ARR already crossed our threshold
        (2.) t_i <= t (because arr_i <= ARR)
    '''
    cond = (df['ARR_p'] < arr_i) & (df['ARR'] >= arr_i)
    df.loc[cond,'ARR_i']= arr_i
    r = cagr_q(end_arr=df['ARR'], start_arr=df['ARR_p'], t_in_q=df['t'] - df['t_p'])
    df.loc[cond,'t_i'] = df.loc[cond,'t_i'] - time_between_arr_ranges(end_arr=df['ARR'], start_arr=df['ARR_i'], r=r)
    return df

def compute_range_metrics(df_c,arr_begin,arr_end):
    df_c['t_i']=df_c['t']
    df_c = interpolate_arr(df_c,arr_begin)
    df_c = interpolate_arr(df_c,arr_end)

    # t_max = the last time point we're interested in because it represents the earliest time where ticker crossed arr_end
    t_max= df_c[df_c['ARR_i']==arr_end]['t_i'].min()
    # t_min = latest time point where ticker crossed arr_begin
    t_min = df_c[(df_c['ARR_i']==arr_begin) & (df_c['t_i']<t_max)]['t_i'].max()
    df_c = df_c[(df_c['t_i']>=t_min) & (df_c['t_i']<=t_max)]

    assert df_c['ARR_i'].count() <= 2
    assert df_c[df_c['ARR_i'] == arr_begin]['ARR_i'].count() <= 1
    assert df_c[df_c['ARR_i'] == arr_end]['ARR_i'].count() <= 1

    #st.table(df_c)
    #st.write(t_min,t_max)
    return pd.Series({'ARR range':pd.Interval(arr_begin,arr_end),
                      'CAGR':cagr_q(arr_end, arr_begin,t_max-t_min),
                      'Time (Quarters)':t_max-t_min})
import time

def benchmark_arr_ranges(df):
    st.write('Range analysis')

    st.sidebar.write("## Analysis filters")
    df = filter.by_gtm(ui.input_gtm(schema.defined_gtm_types()), df)

    df[['ARR_p', 't_p']] = df.groupby('ticker')[['ARR', 't']].shift(1)
    df[['ARR_n', 't_n']] = df.groupby('ticker')[['ARR', 't']].shift(-1)

    #df = df[(df['ticker'] == 'DDOG') | (df['ticker'] == 'ESTC')]
    #df = df[df['ticker']=='ESTC']
    df = df[['ticker', 'date','t', 'ARR', 'ARR_p', 't_p', 'ARR_n', 't_n']]
    df_2 = df
    arr_begin,arr_end = st.slider('Select ARR ranges', min_value=50, max_value=3000,
                           value=(100, 200), step=50, key="arr_range_slider")
    st.write(arr_begin,arr_end)

    #df.set_index(['ticker','ARR','ARR_p'],inplace=True)
    #st.table(df)
    df_plot = pd.DataFrame(columns=['ARR range','CAGR','Time (Quarters)'])
    st.table(df_plot)
    arr_range=pd.interval_range(arr_begin,arr_end,freq=50)
    df_list = [df_plot]
    total = 0
    cnt = 0
    for i in arr_range:
        a0 = i.left
        a1 = i.right
        s0 = time.time()
        df_r = df.groupby('ticker',as_index=False)\
            .apply(compute_range_metrics, arr_begin=i.left, arr_end=i.right)\
            .dropna(axis=0, how='any')\
            .sort_values('CAGR', ascending=False)
        s1 = time.time()
        total = total + s1-s0
        cnt = cnt+1
        df_list.append(df_r)

    st.write(total)
    st.write(cnt)
    st.write(total/cnt)
    df_plot=pd.concat(df_list)
    st.table(df_plot)