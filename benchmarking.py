import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import altair as alt
import streamlit as st
import plotly.express as px

# local imports
from schema import pretty_print as p
import ui
import filter
import schema

# Available templates  ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
px_template = "plotly_dark"


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


def metric_hist(col,df,title,xlabel,st_col=False):
    if df[col].count() <= 2:
        return
    st.subheader(title)
    #summary_statistics(df,col,0)
    df_s = pd.DataFrame(df[col].describe()).T
    df_s =df_s.drop('count',axis=1)
    st.table(df_s.style.format(schema.col_field_formats[col], na_rep="-"))
    fig = px.histogram(df, x=col,template=px_template,labels={col:f"{col} {schema.col_plot_labels[col]}"})
    st.plotly_chart(fig)



def benchmark_timelines(df):
    selected_time = ui.input_timeline(int(df['t'].min()), int(df['t'].max()))
    df = filter.by_time(selected_time, df)

    selected_metric = ui.input_main_metric(schema.metrics)

    col1,col2 = st.beta_columns(2)
    col = selected_metric

    st.subheader(f"**Bechmarking {selected_metric}**: {time_to_string(selected_time)} (N={df[selected_metric].count()})")
    metric_hist(col, df, title=f"**{selected_metric} histogram **", xlabel=col)

    n = 5
    df_top5 = df.sort_values(col,ascending=False).head(n)
    st.subheader(f"Top {n} performers")
    fig = px.bar(df_top5, x='ticker', y=col, template=px_template,labels={col:f"{col} {schema.col_plot_labels[col]}"})
    st.plotly_chart(fig)

    st.subheader(f"Bottom {n} performers")
    df_bot5 = df[df[col]!=0]
    df_bot5 = df_bot5.sort_values(col, ascending=True).head(n)
    fig = px.bar(df_bot5, x='ticker', y=col, template=px_template,labels={col:f"{col} {schema.col_plot_labels[col]}"})
    st.plotly_chart(fig)

    ui.output_table(selected_metric, df, 'Table')
    return


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