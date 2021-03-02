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



# Get CAGR from quarterly number
def cagr_q(s1, s0, t_in_q):
    assert t_in_q >= 0 or np.isnan(float(t_in_q))
    return ((s1 / s0) ** (1 / t_in_q) - 1) * 400

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


def benchmark_ranges(df):
    st.write('Range analysis')

    st.sidebar.write("## Analysis filters")
    df = filter.by_gtm(ui.input_gtm(schema.defined_gtm_types()), df)

    selected_metric = ui.input_main_metric(schema.metrics)
    ticker = ui.input_ticker(list(df['ticker'].unique()))
    st.title(f"**Benchmarking {schema.ticker_to_name(ticker)}**")
    df_c = df[df['ticker']==ticker]
    df_c['type']='Actual'

    df_c=df_c[['ticker','t','ARR','ARR growth','type']]

    def create_interpolation(row):
        r = (cagr_q(row['ARR'],row['ARR_p'],row['t']-row['t_p']))/400
        t_x = np.log(arr/row['ARR_p'])/np.log(1+r)
        row['t']= row['t_p'] + t_x
        row['ARR'] = arr
        row['type']='Interpolated'
        return row


    arr_min = 100
    arr_max = 2000
    arr_increment = 50
    arr_ranges = st.slider('Select ARR ranges', min_value=arr_min, max_value=arr_max,
                          value=(100,200), step = arr_increment, key="arr_range_slider")
    st.write(arr_ranges)

    arr_ranges = range(arr_ranges[0],arr_ranges[1],arr_increment)

    df_c = df_c.assign(ARR_p=df_c['ARR'].shift(1), t_p=df_c['t'].shift(1),
                       ARR_n=df_c['ARR'].shift(-1), t_n=df_c['t'].shift(-1))
    for arr in arr_ranges:
        df_t = df_c[(df_c['ARR']>arr) & (df_c['ARR'].shift(1)<arr)]

        assert df_t.shape[0] <= 1
        if df_t.shape[0] == 1:
            new_row = create_interpolation(df_t.iloc[0])
            df_c = df_c.append(new_row).sort_values('t',ascending=True)
            df_c = df_c.assign(ARR_p=df_c['ARR'].shift(1), t_p=df_c['t'].shift(1),
                               ARR_n=df_c['ARR'].shift(-1), t_n=df_c['t'].shift(-1))



    df_c['CAGR'] = df_c.apply(lambda row: cagr_q(row['ARR'],row['ARR_p'],row['t']-row['t_p']),axis=1)
    df_c = df_c[['ticker', 't', 'ARR', 'CAGR','type']]
    st.table(df_c)



'''
for a given range [l,h]

case I:
 


'''





'''
     selected_range = ui.input_metric_range(selected_metric,
                                               minmax_range=(50, 1300), step=10,
                                               range1=(50, 100), range2=(100, 200), range3=(200, 1000))
        df_range = filter.by_metric_range(selected_metric, df, selected_range)
    
        st.write(selected_range)
    
        df_range = add_range_metrics(selected_metric, df, df_range)
        # summary_statistics(m_df, col, selected_time)
        # metric_hist(col, df, title="", xlabel=col)
        st.write(f"""
                - **N =  {df_range['ticker'].nunique()}** companies
                - from ${selected_range[0]}M to ${selected_range[1]}M
                """)
    
        #metric_hist('CAGR', df_range, title="", xlabel='CAGR')
        #ui.output_table(main_metric, df_range, title='Table',
        #                cols=['Start Date', main_metric + '(0)', 'End Date', main_metric + '(1)',
        #                      'CAGR', 'Years', 'Cap Ef.'])
    
'''