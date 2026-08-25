import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.interpolate import interp1d
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file output
import matplotlib.pyplot as plt
import tsEvaMultivariate as tsm
np.random.seed(42)

# =============================================================================
# Data Loading and Preparation
# =============================================================================
# Load MATLAB .mat file (Assuming variables are 1D arrays)
data = loadmat('caseStudy01_data.mat')
timeSWH = data['timeSWH'].flatten()
timeRiverDisch = data['timeRiverDisch'].flatten()
riverineDischarge = data['riverineDischarge'].flatten()
SWH = data['SWH'].flatten()

# Find the overlapping part of both data sources; define a 3-hourly time frame
t_start = max(timeSWH[0], timeRiverDisch[0])
t_end = min(timeSWH[-1], timeRiverDisch[-1])
# Add small epsilon to the end for floating-point precision in np.arange
timeCommon = np.arange(t_start, t_end + (3/24)/2, 3/24) 

# Find indices of non-NaN data
indexGoodData = ~np.isnan(riverineDischarge)
indexGoodDataW = ~np.isnan(SWH)

# Interpolate river and SWH data using the common time frame
f_river = interp1d(timeRiverDisch[indexGoodData], riverineDischarge[indexGoodData], kind='linear', bounds_error=False, fill_value=np.nan)
riverineDischarge_ = f_river(timeCommon)
timeSeriesRiver = np.column_stack((timeCommon, riverineDischarge_))

f_swh = interp1d(timeSWH[indexGoodDataW], SWH[indexGoodDataW], kind='linear', bounds_error=False, fill_value=np.nan)
SWH_ = f_swh(timeCommon)
timeSeriesSWH = np.column_stack((timeCommon, SWH_))

# =============================================================================
# Parameter Definitions
# =============================================================================

ciPercentile = [99, 99]
potPercentiles = [[95.0], [99.0]]
timeWindowJointDist = 14610
minDeltaUnivarSampli = [30, 30]
maxDeltaMultivarSampli = 45
copulaFamily = 'gumbel'
transfType = 'trendlinear'
marginalDistributions = 'gpd'
samplingOrder = [1, 0]

# SET THE EXTREME VALUE DISTRIBUTION TYPE FROM THE MAIN SCRIPT
evdType = ['GEV', 'GPD']

# =============================================================================
# Analysis and Visualization (Using tsEvaMulti Library)
# =============================================================================

# 1. Copula Extremes Analysis
copulaAnalysis = tsm.tsCopulaExtremes(
    timeSeriesRiver[:, 0], 
    np.column_stack((timeSeriesRiver[:, 1], timeSeriesSWH[:, 1])),
    minPeakDistanceInDaysMonovarSampling=minDeltaUnivarSampli,
    maxPeakDistanceInDaysMultivarSampling=maxDeltaMultivarSampli,
    copulaFamily=copulaFamily,
    transfType=transfType,
    timewindow=timeWindowJointDist,
    ciPercentile=ciPercentile,
    potPercentiles=potPercentiles,
    marginalDistributions=marginalDistributions,
    samplingOrder=samplingOrder,
    timeVaryingCopula=True,
    evdType=evdType  # PASSING THE COMMAND TO THE MULTI MODULE
)

# 2. Monte Carlo Analysis
monteCarloAnalysis = tsm.tsCopulaMontecarlo(
    copulaAnalysis,
    nResample=1000,
    timeIndex='middle'
)

# 3. Goodness of Fit (GOF) Statistics
gofStatistics = tsm.tsCopulaGOFNonStat(copulaAnalysis, monteCarloAnalysis)

# 4. Return Period Analysis
retPerAnalysis = tsm.tsCopulaComputeBivarRP(copulaAnalysis, monteCarloAnalysis)

# 5. Bivariate Visualization
axxArray = tsm.tsCopulaPlotBivariate(
    copulaAnalysis, 
    monteCarloAnalysis,
    gofStatistics=gofStatistics,
    retPerAnalysis=retPerAnalysis,
    ylbl=['River discharge (m^3s^{-1})', 'SWH (m)']
)

# Save the figure as PNG (use plt.show() for interactive display)
plt.savefig('CaseStudy01_output.png', dpi=150, bbox_inches='tight')
print('Figure saved to CaseStudy01_output.png')
plt.show()