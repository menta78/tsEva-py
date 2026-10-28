import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file output
import matplotlib.pyplot as plt
from monovariate_plot_style import configure_plotting, save_figure
configure_plotting()
from datetime import datetime
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsEva import tsEvaComputeReturnLevelsGEVFromAnalysisObj
from tsEva import tsEvaNonStationary
from tsEva import tsEvaPlotSeriesTrendStdDevFromAnalysisObj
from tsEva import datetime_to_datenum
from tsEva import tsEvaPlotGEVImageScFromAnalysisObj
from tsEva import tsEvaPlotReturnLevelsGEVFromAnalysisObj

# A series of TAS (Temperature of Air Surface) yerly maxima is useful to understand how heat waves evolve (Alessandro Dosio, jrc).
# It is a series of yearly maxima, therefore it is fit for a GEV analysis, while a GPD analysis is meaningless.

# Load data
current_working_directory = os.path.dirname(os.path.abspath(__file__))
data_file_name= current_working_directory+"/data/timeAndSeriesTASMax.csv"
data = pd.read_csv(data_file_name, header=None)
timeAndSeries = data.values
seriesDescr = 'TASMax'

timeWindow = 365.25 * 50  # 50 years
minPeakDistanceInDays = 5*30.2
returnPeriodsInYears = [20, 50, 100, 300]


axisFontSize = 20
axisFontSize3d = 16
labelFontSize = 24
titleFontSize = 26

   
nonStationaryEvaParams, stationaryTransformData, isValid = tsEvaNonStationary( timeAndSeries, timeWindow, minPeakDistanceInDays=minPeakDistanceInDays, extremeLowThreshold=.1, evdType=['GEV'])

tsEvaPlotSeriesTrendStdDevFromAnalysisObj(nonStationaryEvaParams, stationaryTransformData, ylabel='TAS', legendLocation='upper left')
plt.title('TAS annual maxima: series and trends')
plt.ylim(0, 40)
save_figure('seriesAndTrends_TAS.png')
plt.close('all')  # Agg: release figure memory after saving

tsEvaPlotGEVImageScFromAnalysisObj(np.arange(0, 40, 0.025), nonStationaryEvaParams, stationaryTransformData, ylabel='TAS', nPlottedTimesByYear=1)
plt.title('Time-varying GEV density for TAS')
plt.ylim(0, 40)
save_figure('TimeVaryingGEV_TAS.png')
plt.close('all')  # Agg: release figure memory after saving

returnLevels, returnLevelsErr = tsEvaComputeReturnLevelsGEVFromAnalysisObj(nonStationaryEvaParams, returnPeriodsInYears)
print("Return Levels (TAS): ", returnLevels)

timeIndex = 26
rlRange = [0, 14]
hndl = tsEvaPlotReturnLevelsGEVFromAnalysisObj(nonStationaryEvaParams, timeIndex, ylim=rlRange, ylabel='return levels (TAS)')
ax = plt.gca()
ax.set_yticks(range(0, 17, 2))
plt.title('TAS return levels, 1995')
plt.ylabel('TAS return level')
save_figure('ReturnLevel1995.png')
plt.close('all')  # Agg: release figure memory after saving

timeIndex = len(timeAndSeries) - 4
rlRange = [0, 70]
hndl = tsEvaPlotReturnLevelsGEVFromAnalysisObj(nonStationaryEvaParams, timeIndex, ylim=rlRange, ylabel='return levels (TAS)')
ax = plt.gca()
ax.set_yticks(range(0, 75, 5))
plt.title('TAS return levels, 2095')
plt.ylabel('TAS return level')
save_figure('ReturnLevel2095.png')
plt.close('all')  # Agg: release figure memory after saving