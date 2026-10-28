import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file output
import matplotlib.pyplot as plt
from monovariate_plot_style import configure_plotting, save_figure
configure_plotting()
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import scipy.io
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsEva import tsEvaNonStationary
from tsEva import tsEvaPlotGEVImageScFromAnalysisObj
from tsEva import tsEvaPlotTransfToStatFromAnalysisObj

# This example illustrates the fact that the time-varying amplitude of
# the signal, necessary for the estimation of the extremes using the ts
# approach, can be estimated by means of a moving standard deviation or
# or by means of a moving percentile.
# The approach using the moving percentile models better the variations of the extremes
# than the approach based on the standard deviation, but is subject to stronger
# uncertainty.
 
# Load data
current_working_directory = os.path.dirname(os.path.abspath(__file__))
data_file_name= current_working_directory+"/data/timeAndSeries_waves_015_220E_055_509N.csv"
#data_file_name= current_working_directory+"/test/data/timeAndSeries_waves_023_688E_059_519N.csv"
data = pd.read_csv(data_file_name, header=None)
timeAndSeries = data.values
seriesDescr = 'waves_015_220E_055_509N'
#sereesDescr = 'waves_023_688E_059_519N'

timeWindow = 365.25 * 30  # 30 years
minPeakDistanceInDays = 3
ylabel = 'Hs (m)'

axisFontSize = 20
axisFontSize3d = 16
labelFontSize = 24
titleFontSize = 26

print('trying moving standard deviation ...')
nonStatEvaParams, statTransfData, _ = tsEvaNonStationary(timeAndSeries, timeWindow, minPeakDistanceInDays=minPeakDistanceInDays)
minext = (np.nanmax(statTransfData.nonStatSeries) + 3*np.nanmin(statTransfData.nonStatSeries))/4
maxext = np.nanmax(statTransfData.nonStatSeries)*1.2
xext = np.arange(minext, maxext, 0.01)
print('  plotting time varying GEV')

tsEvaPlotGEVImageScFromAnalysisObj(xext, nonStatEvaParams, statTransfData, ylabel=ylabel,axisFontSize=axisFontSize, axisFontSize3d=axisFontSize3d, labelFontSize=labelFontSize, titleFontSize=titleFontSize, nPlottedTimesByYear=24)
plt.title('standard deviation', fontsize=titleFontSize)
save_figure('DifferentCI_stdDev.png')
plt.close('all')  # Agg: release figure memory after saving

print('  transformation diagnostic plot')
tsEvaPlotTransfToStatFromAnalysisObj(nonStatEvaParams, statTransfData, ylabel=ylabel, xlabel='Year',axisFontSize=axisFontSize, labelFontSize=labelFontSize, titleFontSize=titleFontSize)
plt.title('standard deviation', fontsize=titleFontSize)
save_figure('DifferentCI_stdDev_Diagnostic.png')
plt.close('all')  # Agg: release figure memory after saving

print('trying moving 98th percentile ...')
nonStatEvaParams, statTransfData, _ = tsEvaNonStationary(timeAndSeries, timeWindow, transfType='trendCIPercentile', ciPercentile=98, minPeakDistanceInDays=minPeakDistanceInDays)
print('  plotting time varying GEV')
tsEvaPlotGEVImageScFromAnalysisObj(xext, nonStatEvaParams, statTransfData, ylabel=ylabel,axisFontSize=axisFontSize, axisFontSize3d=axisFontSize3d, labelFontSize=labelFontSize, titleFontSize=titleFontSize, nPlottedTimesByYear=24)
plt.title('98th percentile', fontsize=titleFontSize)
save_figure('DifferentCI_98thPercentile.png')
plt.close('all')  # Agg: release figure memory after saving

print('  transformation diagnostic plot')
tsEvaPlotTransfToStatFromAnalysisObj(nonStatEvaParams, statTransfData, ylabel=ylabel, xlabel='Year', axisFontSize=axisFontSize, labelFontSize=labelFontSize, titleFontSize=titleFontSize)
plt.title('98th percentile', fontsize=titleFontSize)
save_figure('DifferentCI_98thPercentile_Diagnostic.png')
plt.close('all')  # Agg: release figure memory after saving

print('trying moving 98.5th percentile ...')
nonStatEvaParams, statTransfData, _ = tsEvaNonStationary(timeAndSeries, timeWindow, transfType='trendCIPercentile', ciPercentile=98.5, minPeakDistanceInDays=minPeakDistanceInDays)
print('  plotting time varying GEV')
tsEvaPlotGEVImageScFromAnalysisObj(xext, nonStatEvaParams, statTransfData, ylabel=ylabel,axisFontSize=axisFontSize, axisFontSize3d=axisFontSize3d, labelFontSize=labelFontSize, titleFontSize=titleFontSize, nPlottedTimesByYear=24)
plt.title('98.5th percentile', fontsize=titleFontSize)
save_figure('DifferentCI_98.5thPercentile.png')
plt.close('all')  # Agg: release figure memory after saving

print('  transformation diagnostic plot')
tsEvaPlotTransfToStatFromAnalysisObj(nonStatEvaParams, statTransfData, ylabel=ylabel, xlabel='Year', axisFontSize=axisFontSize, labelFontSize=labelFontSize, titleFontSize=titleFontSize)
plt.title('98.5th percentile', fontsize=titleFontSize)
save_figure('DifferentCI_98.5thPercentile_Diagnostic.png')
plt.close('all')  # Agg: release figure memory after saving

print('trying moving 99th percentile ...')
nonStatEvaParams, statTransfData, _ = tsEvaNonStationary(timeAndSeries, timeWindow, transfType='trendCIPercentile', ciPercentile=99, minPeakDistanceInDays=minPeakDistanceInDays)
print('  plotting time varying GEV')
tsEvaPlotGEVImageScFromAnalysisObj(xext, nonStatEvaParams, statTransfData, ylabel=ylabel,axisFontSize=axisFontSize, axisFontSize3d=axisFontSize3d, labelFontSize=labelFontSize, titleFontSize=titleFontSize, nPlottedTimesByYear=24)
plt.title('99th percentile', fontsize=titleFontSize)
save_figure('DifferentCI_99thPercentile.png')
plt.close('all')  # Agg: release figure memory after saving

print('  transformation diagnostic plot')
tsEvaPlotTransfToStatFromAnalysisObj(nonStatEvaParams, statTransfData, ylabel=ylabel, xlabel='Year', axisFontSize=axisFontSize, labelFontSize=labelFontSize, titleFontSize=titleFontSize)
plt.title('99th percentile', fontsize=titleFontSize)
save_figure('DifferentCI_99thPercentile_Diagnostic.png')
plt.close('all')  # Agg: release figure memory after saving
