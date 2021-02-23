import streamlit as st
from schema import pretty_print as p
import schema
import ui
import numpy as np

""" Filtering by GTM
"""

@ui.register_plugin(name="input_gtm")
def sidebar_select_gtm(gtm_options_list):
    selected_gtm_option = st.sidebar.selectbox('GTM', gtm_options_list)
    return selected_gtm_option

@ui.register_plugin(name="input_analysis_type")
def sidebar_select_analysis_type(list_of_anaysis_types):
    selection = st.sidebar.selectbox("Analysis", list_of_anaysis_types)
    return selection

""" UI Selectbox to filtering by Columns
"""

""" UI Selectbox to pick a metric to analyze
rev --> rev g
arr --> arr g
rev g --> rev
arr g --> arr
"""

growth = {'LTM Rev':'Rev growth',
          'Rev growth':'LTM Rev',
          'ARR':'ARR growth',
          'ARR growth':'ARR'}

@ui.register_plugin(name="input_main_metric")
def select_main_metric(list_of_metrics):
    main_metric = st.sidebar.selectbox('Target metric', list_of_metrics)
    return main_metric

@ui.register_plugin(name="input_bar_limit")
def select_bar_limit(t_max):
    default = np.minimum(t_max,5)
    bar_len = st.sidebar.slider('Panel width', 1, int(t_max), 10)
    return bar_len


""" Filtering by Timeline
"""

@ui.register_plugin(name="input_timeline")
def sidebar_select_timeline(t_min,t_max) -> object:
    selected_time = st.sidebar.slider('Select quarter to benchmark (t)', t_min, t_max, 0,key="timeline")
    return selected_time

""" Filtering by Metric Range
"""

@ui.register_plugin(name="input_metric_range")
def sidebar_select_metric_range(metric_name,minmax_range, step = 10, **kwargs):
    if metric_name != 'ARR' and metric_name == 'LTM Rev':
        raise NotImplementedError

    range_choices = {f"${value[0]}M:${value[1]}M": value for (key, value) in kwargs.items()}
    range_choices['Custom'] = 'Custom'

    selection = st.sidebar.radio(f'Pick an {metric_name} range to benchmark',list(range_choices.keys()))

    selected_range = st.sidebar.slider(f'Pick an {metric_name} range to benchmark',
                                       minmax_range[0], minmax_range[1], (100, 200),
                                       step=step, key="metric_range") if selection == 'Custom' else range_choices[selection]
    return selected_range


def _select_table_columns_to_display(topline_metric):
    selected_cols = ['date'] + [topline_metric] + [growth[topline_metric]]
    selected_cols.append(st.selectbox('Expenses', schema.expense))
    selected_cols.append(st.selectbox('Profitability', schema.profit))
    selected_cols += schema.efficiency
    return selected_cols


@ui.register_plugin(name="input_ticker")
def sidebar_select_ticker(ticker_list):
    ticker = st.sidebar.selectbox('Ticker:  ', ticker_list)
    return ticker

""" Display a dataframe with the right set of columns and column styles and field formats
"""

@ui.register_plugin(name="expandable_table")
def expandable_table(main_metric,m_df,title="",cols=[]):
    df = m_df.set_index('ticker')
    topline_metric = 'ARR' if main_metric not in schema.rev else main_metric
    with st.beta_expander(title):
        cols = _select_table_columns_to_display(topline_metric) if cols == [] else cols
        st.write(p(df[cols]))

