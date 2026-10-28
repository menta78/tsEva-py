import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tsEva import tsEvaStationary
from tsEva import tsEvaComputeReturnLevelsGEVFromAnalysisObj
from tsEva import tsEvaComputeReturnLevelsGPDFromAnalysisObj
from tsEva import tsEvaPlotReturnLevelsGEVFromAnalysisObj
from tsEva import tsEvaPlotReturnLevelsGPDFromAnalysisObj
from tsEva import tsPlotSeriesYearMaxGEVRetLevStationary
from tsEva import tsPlotSeriesPotGPDRetLevStationary
from tsEva import tsEvaPandasDate2DateNum

# Load the Hebrides dataset from the project, independent of working directory.
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
data_file_name = project_root / 'test_monovariate' / 'data' / 'timeAndSeriesHebrides.csv'
if not data_file_name.is_file():
    raise FileNotFoundError(f'Hebrides dataset not found: {data_file_name}')
data = pd.read_csv(data_file_name, header=None, names=['year','month','day','hour','value'])

# Compute tsEva timestamps (MATLAB serial days) from year/month/day/hour
dates = pd.to_datetime(data[['year','month','day','hour']])
timestamps = tsEvaPandasDate2DateNum(dates)
timeAndSeries = np.column_stack([timestamps, data['value'].values])
minPeakDistanceInDays = 3

return_periods = [10, 20, 50, 100]
potPercentiles = [99] # testing just 1 percentile




print('Stationary fit of extreme value distributions (GEV, GPD) to a time series')

# Stationary fitting (Here, you would fit the GEV and GPD distributions)
statEvaParams = tsEvaStationary(timeAndSeries, minPeakDistanceInDays=minPeakDistanceInDays,
                                potPercentiles=potPercentiles)


# Compute return levels for GEV
rlevGEV,rlevGEVErr = tsEvaComputeReturnLevelsGEVFromAnalysisObj(statEvaParams, return_periods)
print("rlevGEV=", rlevGEV)
print("rlevGEVErr=", rlevGEVErr)

# Plotting the GEV return levels
hndl = tsEvaPlotReturnLevelsGEVFromAnalysisObj(statEvaParams, 0, ylim=[0.5, 1.5])
plt.title('GEV')
plt.savefig('GEV_ReturnLevels_STATIONARY.png')
plt.show()

# Plotting the series with GEV return levels
hndl = tsPlotSeriesYearMaxGEVRetLevStationary(statEvaParams, timeAndSeries)
hndl['fig'].suptitle('GEV')
hndl['fig'].savefig('GEV_SeriesReturnLevels_STATIONARY.png')
plt.show()

# For GPD fitting, we use the generalized Pareto distribution
rlevGPD,rlevGPDErr = tsEvaComputeReturnLevelsGPDFromAnalysisObj(statEvaParams, return_periods)
print("rlevGPD=", rlevGPD)
print("rlevGPDErr=", rlevGPDErr)

# Plotting the GPD return levels
hndl = tsEvaPlotReturnLevelsGPDFromAnalysisObj(statEvaParams, 0, ylim=[0.5, 1.5])
plt.title('GPD')
plt.savefig('GPD_ReturnLevels_STATIONARY.png')
plt.show()

# Plotting the series with GPD return levels
hndl = tsPlotSeriesPotGPDRetLevStationary(statEvaParams, timeAndSeries)
hndl['fig'].suptitle('GPD')
hndl['fig'].savefig('GPD_SeriesReturnLevels_STATIONARY.png')
plt.show()



