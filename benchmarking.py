import inflect
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import filter
import schema
# local imports
import ui

# Available templates  ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
px_template = "plotly_dark"


def time_to_string(selected_time):
    p = inflect.engine()
    if selected_time == 0:
        return "at IPO"
    time_str = f"{p.number_to_words(abs(selected_time))}"
    time_str = time_str + ' quarter' if abs(selected_time) == 1 else time_str + ' quarters'
    time_str = time_str + ' after IPO' if selected_time > 0 else time_str + ' before IPO'
    return time_str


def summary_statistics(df, metric, selected_time):
    st.markdown("### Summary statistics")
    st.markdown(f"({metric} for **{df[metric].count()}** companies measured {time_to_string((selected_time))})")
    st.markdown(f"""
> - Median {metric}: ${np.round(df[metric].median(), 0)}M
> - Median {metric} at {time_to_string(selected_time)} = ${np.round(df[metric].median(), 0)}Mhikhaiaktw o.
""")


def metric_hist(col, df, title, xlabel, st_col=False):
    if df[col].count() <= 2:
        return
    st.header(title)
    # summary_statistics(df,col,0)
    df_s = pd.DataFrame(df[col].describe()).T
    df_s = df_s.drop('count', axis=1)
    fig = px.histogram(df, x=col, template=px_template, labels={col: f"{col} {schema.col_plot_labels[col]}"})
    st.plotly_chart(fig)
    st.table(df_s.style.format(schema.col_field_formats[col], na_rep="-"))


def cagr(s1, s0, t_in_q):
    return ((s1 / s0) ** (1 / t_in_q) - 1) * 400


def ticker_bar_chart(ticker, df, col, n=10, top=True):
    df_n = df[df['ticker'] != ticker].sort_values(col, ascending=not (top)).head(n)
    df_n = pd.concat([df[df['ticker'] == ticker], df_n])
    df_n['color'] = np.where(df_n['ticker'] == ticker, 'crimson', 'lightslategray')
    df_n = df_n.sort_values(col, ascending=False)
    fig = px.bar(df_n, x='ticker', y=col, template=px_template, color='color', color_discrete_map="identity",
                 category_orders={"ticker": list(df_n['ticker'])},
                 labels={col: f"{col} {schema.col_plot_labels[col]}"})
    return fig


def benchmark_ticker_point_in_time(ticker, selected_metric, df):
    st.markdown("---")
    container = st.beta_container()

    t_min = int(df[df['ticker'] == ticker]['t'].min())
    t_max = int(df[df['ticker'] == ticker]['t'].max())
    selected_time = st.slider('Select quarter:',
                              min_value=t_min, max_value=t_max, value=0, key="ticker_point_in_time")

    df = filter.by_time(selected_time, df)

    col = selected_metric

    container.header(f" **{selected_metric} ** {time_to_string(selected_time)} ")

    container2 = st.beta_container()
    n = ui.input_bar_limit(df[col].count())
    container.info(f"**(N={df[selected_metric].count()} companies)**")

    container.subheader(f"**{schema.ticker_to_name(ticker)} relative to Top {n}**")
    fig = ticker_bar_chart(ticker, df, col, n, top=True)
    container.plotly_chart(fig)

    st.subheader(f"**{schema.ticker_to_name(ticker)} relative to Bottom {n}**")
    fig = ticker_bar_chart(ticker, df, col, n, top=False)
    st.plotly_chart(fig)

    st.subheader(f"**Peer group {selected_metric}**")
    metric_hist(col, df, title="", xlabel=col)

    ui.output_table(selected_metric, df, 'Table')
    return


def bechmark_ticker_across_time(ticker, selected_metric, df):

    st.header(f" **{selected_metric} over time** ({schema.ticker_to_name(ticker)})")
    container = st.beta_container()

    t_min = int(df[df['ticker'] == ticker]['t'].min())
    t_max = int(df[df['ticker'] == ticker]['t'].max())
    t_start = max(-4,t_min)
    t_end = min(4,t_max)
    selected_time = st.slider('Select time range:',
                              min_value=t_min, max_value=t_max, value=(t_start,t_end), key="ticker_across_time")
    df = filter.by_time(selected_time, df,range=True)

    df_c = df[df['ticker'] == ticker]
    df_c['median'] = df.groupby('t')[selected_metric].transform('median')
    df_c['top quartile'] = df.groupby('t')[selected_metric].transform(lambda x: x.quantile(.75))
    df_c['bot quartile'] = df.groupby('t')[selected_metric].transform(lambda x: x.quantile(.25))


    df_plot = df_c[['t', selected_metric, 'median', 'top quartile', 'bot quartile']]
    df_plot.rename(columns={selected_metric: ticker}, inplace=True)
    df_plot = df_plot.melt(id_vars='t', var_name='Legend', value_name=selected_metric)

    fig = px.line(df_plot, x='t', y=selected_metric, template=px_template, line_dash='Legend', color='Legend',
                  labels={selected_metric: f"{selected_metric} {schema.col_plot_labels[selected_metric]}"})
    container.plotly_chart(fig)


def benchmarking_main(df):
    st.sidebar.write("## Analysis filters")
    df = filter.by_gtm(ui.input_gtm(schema.defined_gtm_types()), df)

    selected_metric = ui.input_main_metric(schema.metrics)
    ticker = ui.input_ticker(list(df['ticker'].unique()))
    st.title(f"**Bechmarking {schema.ticker_to_name(ticker)}**")
    st.info(
        """
        To make comparisons possible, all companies have been *indexed to a common timeline* where:

        > **t = 0**: quarter in which a company goes public

        > **t = &#177;x** : **x** quarters prior/following the IPO

        """)
    bechmark_ticker_across_time(ticker, selected_metric, df)
    benchmark_ticker_point_in_time(ticker, selected_metric, df)


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
