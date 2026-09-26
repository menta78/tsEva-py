import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file output
import matplotlib.pyplot as plt
import tsEvaMultivariate as tsm
np.random.seed(42)

# =============================================================================
# Case Study 02 — Bahmanpour et al., 2025
# Trivariate spatial dependence of SWH across three Marshall-Islands locations.
# 3-hourly SWH, 1950-2020, non-stationary Gumbel copula with GPD margins.
# Python port of caseStudy02.m (Gumbel / tsCopulaPlotTrivariate variant).
# =============================================================================

# =============================================================================
# Data Loading and Preparation
# =============================================================================
# timeAndSeries{1,2,3}: col0 = time (MATLAB datenum), col1 = SWH at each location
data = loadmat('caseStudy02_data.mat')
timeAndSeries1 = data['timeAndSeries1']
timeAndSeries2 = data['timeAndSeries2']
timeAndSeries3 = data['timeAndSeries3']

# =============================================================================
# Parameter Definitions (match caseStudy02.m exactly)
# =============================================================================

# Percentile levels of univariate series (used for transformation)
ciPercentile = [99, 99, 99]

# Peak-over-threshold levels used for sampling of univariate series
potPercentiles = [[99.0], [99.0], [99.0]]

# Non-stationary time window (in days) used for time-varying joint distribution
timeWindowNonStat = 365 * 40

# Minimum distance (in days) between univariate peaks
minDeltaUnivarSampli = [0.5, 0.5, 0.5]

# Maximum distance (in days) between multivariate peaks
maxDeltaMultivarSampli = 0.5

# Copula family
copulaFamily = 'gumbel'

# Methodology to perform univariate transformation
transfType = 'trendlinear'
peakType = 'allExceedThreshold'

# =============================================================================
# Analysis and Visualization (Using tsEvaMultivariate Library)
# =============================================================================

# 1. Copula Extremes Analysis (trivariate)
# MATLAB: tsCopulaExtremes(timeAndSeries1(:,1),
#           [timeAndSeries1(:,2), timeAndSeries2(:,2), timeAndSeries3(:,2)], ...)
copulaAnalysis = tsm.tsCopulaExtremes(
    timeAndSeries1[:, 0],
    np.column_stack((timeAndSeries1[:, 1],
                     timeAndSeries2[:, 1],
                     timeAndSeries3[:, 1])),
    minPeakDistanceInDaysMonovarSampling=minDeltaUnivarSampli,
    maxPeakDistanceInDaysMultivarSampling=maxDeltaMultivarSampli,
    copulaFamily=copulaFamily,
    transfType=transfType,
    timewindow=timeWindowNonStat,
    ciPercentile=ciPercentile,
    potPercentiles=potPercentiles,
    peakType=peakType,
)

# 2. Monte Carlo Analysis - large (for statistics computation)
monteCarloAnalysis1 = tsm.tsCopulaMontecarlo(
    copulaAnalysis,
    nResample=10000,
    timeIndex='middle',
)

# 3. Monte Carlo Analysis - small (for plotting)
monteCarloAnalysis2 = tsm.tsCopulaMontecarlo(
    copulaAnalysis,
    nResample=300,
    timeIndex='middle',
)

# 4. Goodness of Fit (GOF) Statistics
gofStatistics = tsm.tsCopulaGOFNonStat(copulaAnalysis, monteCarloAnalysis1, smoothInd=10)

# 5. Trivariate Visualization
axxArray = tsm.tsCopulaPlotTrivariate(
    copulaAnalysis,
    monteCarloAnalysis2,
    gofStatistics=gofStatistics,
    varLabels=['Loc 1 - SWH (m)', 'Loc 2 - SWH (m)', 'Loc 3 - SWH (m)'],
)

# Save the figure as PNG
plt.savefig('CaseStudy02_output.png', dpi=150, bbox_inches='tight')
print('Figure saved to CaseStudy02_output.png')
plt.show()
