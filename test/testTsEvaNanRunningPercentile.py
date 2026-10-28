import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file output
import matplotlib.pyplot as plt
from monovariate_plot_style import configure_plotting, save_figure
configure_plotting()
import matplotlib.dates as mdates
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsEva import tsEvaNanRunningPercentile
from tsEva import tsEvaRunningMeanTrend
from tsEva import datenum_to_datetime
from tsEva import tsEvaPandasDate2DateNum

# this script tests the function for computing of the running percentile

# Load data
current_working_directory = os.path.dirname(os.path.abspath(__file__))
data_file_name= current_working_directory+"/data/timeAndSeriesHebrides.csv"
data = pd.read_csv(data_file_name, header=None, names=['year','month','day','hour','value'])
# Compute tsEva timestamps (MATLAB serial days) from year/month/day/hour
dates = pd.to_datetime(data[['year','month','day','hour']])
timestamps = tsEvaPandasDate2DateNum(dates)
timeAndSeries = np.column_stack([timestamps, data['value'].values])
seriesDescr = 'Hebrides'

timeWindow = 365.25 * 6  # 6 years
percent = 80

timeStamps = timeAndSeries[:,0]
series = timeAndSeries[:,1]

axisFontSize = 20
axisFontSize3d = 16
labelFontSize = 24
titleFontSize = 26

   
# Compute running mean trend
trendSeries, filledTimeStamps, filledSeries, nRunMn = tsEvaRunningMeanTrend( timeStamps, series, timeWindow)

# Plot the raw series lightly so the percentile curves remain visible.
filledTimeStamps_dt = np.array([datenum_to_datetime(ts) for ts in filledTimeStamps])
fig, ax = plt.subplots(figsize=(13.5, 7.25))
ax.plot(filledTimeStamps_dt, filledSeries, color='#6384a1', linewidth=0.55,
        alpha=0.40, rasterized=True, label='Observed series', zorder=1)
percentile_colors = ['#e19a48', '#268b8a', '#bb626e', '#8665b5', '#5c533f']
for pct, color in zip([80, 90, 95, 98, 99], percentile_colors):
    rnprcnt, err = tsEvaNanRunningPercentile(filledSeries, nRunMn, pct)
    print(f"Error = {err/np.nanmean(rnprcnt)*100} %")
    ax.plot(filledTimeStamps_dt, rnprcnt, color=color, linewidth=2.1,
            label=f'{pct}th percentile', zorder=3)

ax.set_xlim(filledTimeStamps_dt.min(), filledTimeStamps_dt.max())
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_xlabel('Year')
ax.set_ylabel('Series value')
ax.set_title('Running Mean Trend and Moving Percentiles')
ax.legend(loc='upper center', ncol=3)  # moved above axes by save_figure()
save_figure('RunningPercentile.png', fig=fig)
print('Figure saved to RunningPercentile.png')
plt.close('all')  # Agg: release figure memory after saving