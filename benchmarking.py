import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import altair as alt
import streamlit as st

# local imports
from schema import pretty_print as p
import ui
import filter
import schema

use_pyplot = True

def old_pyplot_code():
    """
    fig, ax = plt.subplots()
    ax.hist(df['ARR'], bins=24)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    n = df['ARR'].describe()['count']
    ax.set_title(f'ARR distribution (N={n})')
    ax.boxplot(df['ARR'], showfliers=False)
    st.pyplot(fig)

    #    df['1'] = ''
    #    c2 = alt.Chart(df).mark_boxplot().encode(
    #        x='1',
    #        y='ARR'
    #    )
    #    #col2.altair_chart(c2)
"""



def time_to_string(selected_time):
    if selected_time == 0:
        return "**@ IPO**"
    elif selected_time <0:
        return f"**{abs(selected_time)}** quarters before IPO"
    else:
        return f"**{selected_time}** quarters post IPO"

def summary_statistics(df,metric,selected_time):
    st.markdown("### Summary statistics")
    st.markdown(f"({metric} for **{df[metric].count()}** companies measured {time_to_string((selected_time))})")
    st.markdown(f"""
> - Median {metric}: ${np.round(df[metric].median(), 0)}M
> - Median {metric} at {time_to_string(selected_time)} = ${np.round(df[metric].median(), 0)}Mhikhaiaktw o.
""")

def benchmark_timelines(df):
    selected_time = ui.input_timeline(int(df['t'].min()), int(df['t'].max()))
    df = filter.by_time(selected_time, df)

    selected_metric = ui.input_main_metric(schema.metrics)

    col1,col2 = st.beta_columns(2)
    col = selected_metric

    st.write(f"**{selected_metric} distribution**")
    st.write(f"""
- **N =  {df[selected_metric].count()}** companies
- {time_to_string(selected_time)}
""")

    metric_hist(col, df, title="", xlabel=col)

    n = 5
    st.write(f"Top {n} performers")
    df_top5 = df.sort_values(col,ascending=False).head(n)
    #c = alt.Chart(df_top5).mark_bar().encode(
    #    x='ticker',
    #    y=col).properties(width=650, height=400)
    c = alt.Chart(df_top5).mark_bar().encode(
        alt.X('ticker',sort = alt.EncodingSortField(field=col, op="count", order='descending')),
        y = col).properties(width=650,height=400
    )
    st.write(c)

    st.write(f"Bottom {n} performers")
    df_bot5 = df[df[col]!=0]
    df_bot5 = df_bot5.sort_values(col, ascending=True).head(n)

    c = alt.Chart(df_bot5).mark_bar().encode(
        alt.X('ticker', sort=alt.EncodingSortField(field=col, op="count", order='ascending')),
        y=col).properties(width=650, height=400
                            )
    st.write(c)

    ui.output_table(selected_metric, df, 'Table')
    return

def metric_hist(col,df,title,xlabel,st_col=False):
    if use_pyplot:
        fig, ax = plt.subplots(figsize=(4, 3))
        num_bins = filter.get_optimal_hist_bin_size(col, df)
        ax.hist(df[col], bins=num_bins)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        #plt.figtext(0.1, 0.5, p(df[col].describe()))

        if st_col:
            st_col.pyplot(fig)
        else:
            st.pyplot(fig)
    else:
        c = alt.Chart(df).mark_bar().encode(
            alt.X(col, bin=alt.Bin(extent=[0, 1200], step=50)),
            alt.Y('count()'),
        )
        if st_col:
            st_col.altair_chart(c)
        else:
            st.altair_chart(c)

def cagr(s1,s0,t_in_q):
    return ((s1/s0)**(1/t_in_q)-1)*400

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
    df_range['Cap Ef.'] = np.round(((df_range[metric + '(1)'] - df_range[metric + '(0)']) / df_range['opex_in_range']) * 100, 2)

    df_range['Years'] = (df_range['t_y'] - df_range['t_x']) / 4

    df_range = df_range.astype({'Years': 'float'})
    return df_range



def benchmark_ranges(main_metric,df):
    st.write('Range analysis')
    st.write(f"**{main_metric} distribution**")

    selected_range = ui.input_metric_range(main_metric,
                                           minmax_range=(50, 1300), step=10,
                                           range1=(50, 100), range2=(100, 200), range3=(200, 1000))
    df_range = filter.by_metric_range(main_metric, df, selected_range)

    st.write(selected_range)

    df_range = add_range_metrics(main_metric, df, df_range)
    # summary_statistics(m_df, col, selected_time)
    #metric_hist(col, df, title="", xlabel=col)
    st.write(f"""
            - **N =  {df_range['ticker'].nunique()}** companies
            - from ${selected_range[0]}M to ${selected_range[1]}M
            """)

    metric_hist('CAGR', df_range, title="", xlabel='CAGR')
    ui.output_table(main_metric, df_range, title='Table',
                    cols=['Start Date', main_metric+'(0)', 'End Date', main_metric+'(1)',
                                      'CAGR', 'Years', 'Cap Ef.'])


def benchmarking_main(df):
    st.sidebar.write("## Analysis filters")
    df = filter.by_gtm(ui.input_gtm(schema.defined_gtm_types()), df)
    st.write(df)

    selected_analysis = ui.input_analysis_type(schema.analysis_types)
    if selected_analysis == 'IPO timeline':
            benchmark_timelines(df)
    else:
        main_metric = 'ARR' if selected_analysis == 'ARR range' else 'LTM Rev'
        benchmark_ranges(main_metric,df)


"""
Types of analysis:
(1.) Pick a time
(2.) Pick a revenue/ARR range

On (2.):
[Pick a revenue range]:
    Give a range of revenue ($100-200M) ask the following questions:
        (a.) Growth:
            o which company grew the fastest from $100M - $200M?
            o Median growth rate from $100M - $200M
            o Top 5 / Bottom 5
            
        (b.) Sales efficiency
            o which company had the highest sales efficiency from $100M to $200M
            o Median sales efficiency
            o Top 5 / Bottom 5
        


"""