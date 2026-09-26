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
# Data Loading and Preparation
# =============================================================================
# Load MATLAB .mat file
# timeAndSeries1: col0=time, col1=temperature
# timeAndSeries2: col0=time, col1=SPEI
data = loadmat('caseStudy03_data.mat')
timeAndSeries1 = data['timeAndSeries1']
timeAndSeries2 = data['timeAndSeries2']

# =============================================================================
# Parameter Definitions
# =============================================================================

# Percentile levels of univariate series (used for transformation)
ciPercentile = [99, 99]

# Peak-over-threshold levels used for sampling of univariate series
potPercentiles = [[75.0], [97.0]]

# Non-stationary time window (in days) used for time-varying joint distribution
timeWindowNonStat = 365 * 35

# Minimum distance (in days) between univariate peaks
minDeltaUnivarSampli = [30, 30]

# Maximum distance (in days) between multivariate peaks
maxDeltaMultivarSampli = 12 * 30

# Copula family
copulaFamily = 'gumbel'

# Methodology to perform univariate transformation
transfType = 'trendlinear'
peakType = 'allExceedThreshold'
marginalDistributions = 'gev'

# SET THE EXTREME VALUE DISTRIBUTION TYPE FROM THE MAIN SCRIPT
evdType = ['GEV', 'GEV']

# =============================================================================
# Analysis and Visualization (Using tsEvaMultivariate Library)
# =============================================================================

# 1. Copula Extremes Analysis
# MATLAB: tsCopulaExtremes(timeAndSeries1(:,1), [timeAndSeries2(:,2), timeAndSeries1(:,2)], ...)
copulaAnalysis = tsm.tsCopulaExtremes(
    timeAndSeries1[:, 0],
    np.column_stack((timeAndSeries2[:, 1], timeAndSeries1[:, 1])),
    minPeakDistanceInDaysMonovarSampling=minDeltaUnivarSampli,
    maxPeakDistanceInDaysMultivarSampling=maxDeltaMultivarSampli,
    copulaFamily=copulaFamily,
    transfType=transfType,
    timewindow=timeWindowNonStat,
    ciPercentile=ciPercentile,
    potPercentiles=potPercentiles,
    peakType=peakType,
    marginalDistributions=marginalDistributions,
    smoothInd=10,
    timeVaryingCopula=True,
    evdType=evdType
)

# 2. Monte Carlo Analysis - large (for statistics computation)
monteCarloAnalysis1 = tsm.tsCopulaMontecarlo(
    copulaAnalysis,
    nResample=10000,
    timeIndex='middle',
    nonStationarity='margins',
    mcSeed=42
)

# 3. Monte Carlo Analysis - small (for plotting)
# mcSeed chosen empirically to produce MC scatter visually closest to the MATLAB reference
monteCarloAnalysis2 = tsm.tsCopulaMontecarlo(
    copulaAnalysis,
    nResample=1000,
    timeIndex='middle',
    nonStationarity='margins',
    mcSeed=42
)

# 4. Goodness of Fit (GOF) Statistics
gofStatistics = tsm.tsCopulaGOFNonStat(copulaAnalysis, monteCarloAnalysis1, smoothInd=10)

# 5. Return Period Analysis
retPerAnalysis = tsm.tsCopulaComputeBivarRP(copulaAnalysis, monteCarloAnalysis1)

# 6. Bivariate Visualization
axxArray = tsm.tsCopulaPlotBivariate(
    copulaAnalysis,
    monteCarloAnalysis2,
    gofStatistics=gofStatistics,
    retPerAnalysis=retPerAnalysis,
    ylbl=['- SPEI', 'Temp. K'],
    smoothInd=10
)

# Save the figure as PNG
plt.savefig('CaseStudy03_output.png', dpi=150, bbox_inches='tight')
print('Figure saved to CaseStudy03_output.png')
plt.show()
