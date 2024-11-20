import numpy as np
import pandas as pd

# Traffic Classes and EALF
traffic_ealf = {
    1:0.0,
    2:0.0,
    3:0.0,
    4:0.57,
    5:0.26,
    6:0.42,
    7:0.42,
    8:0.30,
    9:1.20,
    10:0.93,
    11:0.82,
    12:1.06,
    13:1.39
}

# input should be AADT and percentaion of each traffic class
# e.g.:
# AADT=1000, traffic_perc = {4:0.20, 5:0.12}

def get_daily_ESAL(AADT, traffic_perc, G=0.04, design_years=21, D=0.5, L=0.79):
    total_truck_traffic = 0
    for traffic_class in traffic_perc.keys():
        total_truck_traffic += traffic_ealf[traffic_class] * traffic_perc[traffic_class]
    # print(total_truck_traffic)
    if G == 0.0:
        yearly_values = [AADT * total_truck_traffic * D * L * 365 for _ in range(design_years)]
    else:
        yearly_values = [AADT * total_truck_traffic * D * L * 365 * (((1 + G)**max(0.01, year)) - 1) / G for year in range(design_years)]
    daily_values = np.zeros(design_years * 365)  # Array to hold daily ESAL values
    # Fill daily values by interpolating yearly increases
    start_index = 0
    for year in range(design_years):
        end_index = start_index + 365
        if year == 0:
            daily_values[start_index:end_index] = yearly_values[year] / 365
        else:
            # Calculate daily increase from the last year's total to this year's total
            daily_increase = (yearly_values[year] - yearly_values[year - 1]) / 365
            daily_values[start_index:end_index] = daily_values[start_index - 1] + daily_increase
        start_index = end_index

    return daily_values