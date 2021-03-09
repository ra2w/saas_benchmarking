import numpy as np

# Get CAGR from quarterly number
def cagr_q(end_arr, start_arr, t_in_q):
    # assert t_in_q >= 0 or np.isnan(float(t_in_q))
    return ((end_arr / start_arr) ** (1 / t_in_q) - 1) * 400


def time_between_arr_ranges(end_arr, start_arr, r):
    return np.log(end_arr / start_arr) / np.log(1 + r / 400)


def interpolate_time(arr, arr_p, t_in_q, arr_i):
    r = cagr_q(end_arr=arr, start_arr=arr_p, t_in_q=t_in_q)
    t_x = time_between_arr_ranges(end_arr=arr, start_arr=arr_i, r=r)
    return t_x
