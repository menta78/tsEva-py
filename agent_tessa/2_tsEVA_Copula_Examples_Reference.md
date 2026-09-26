# tsEVA 2.0 – Copula / Multivariate Examples Reference (Python)

This document describes the copula-based multivariate extreme value analysis workflows
implemented in the official tsEva-py example scripts:

- `caseStudy01.py` – bivariate compound flooding (GPD margins)
- `caseStudy02.py` – trivariate spatial dependence (GPD margins)
- `caseStudy03.py` – bivariate compound climate extremes (GEV margins)

All examples implement the Transformed-Stationary (TS) approach:

1. Transform non-stationary marginals to stationary
2. Apply stationary EVA (GPD or GEV)
3. Model dependence with a copula
4. Generate joint extremes via Monte Carlo simulation
5. Assess goodness of fit and compute return periods
6. Visualize joint distributions and diagnostics

All code examples below are taken directly from the Python case studies in
`test_multivariate/`. The multivariate module is imported as:

```python
import tsEvaMultivariate as tsm
```

---

## Quick Function Index

### Core Analysis Functions
- `tsCopulaExtremes(inputtimestamps, inputtimeseries, **kwargs)` – Main copula analysis (margins + dependence + event pairing)
- `tsCopulaMontecarlo(copulaAnalysis, **kwargs)` – Monte Carlo simulation of joint extremes
- `tsCopulaGOFNonStat(copulaAnalysis, monteCarloAnalysis, **kwargs)` – Goodness-of-fit testing for copula model
- `tsCopulaComputeBivarRP(copulaAnalysis, monteCarloAnalysis, **kwargs)` – Compute bivariate return periods (OR/AND)

### Plotting Functions
- `tsCopulaPlotBivariate(copulaAnalysis, monteCarloAnalysis, **kwargs)` – Comprehensive bivariate diagnostic plots (scatter, marginals, GOF, return period)
- `tsCopulaPlotTrivariate(copulaAnalysis, monteCarloAnalysis, **kwargs)` – Comprehensive trivariate diagnostic plots (scatter matrix, marginals, GOF)
- `tsCopulaPlotTrivariateWithMap(copulaAnalysis, monteCarloAnalysis, **kwargs)` – Trivariate plots with spatial map overlay
- `tsCopulaPlotJointReturnPeriod(copulaAnalysis, monteCarloAnalysis, **kwargs)` – Joint return period contour plot (AND/OR)
- `tsCopulaPeakExtrPlotSctrBivar(monteCarloRsmpl, yMaxLevel, **kwargs)` – Peak extremes scatter plot (bivariate)

### Advanced/Specialized Functions
- `tsCopulaFit(copulaFamily, uProb)` – Low-level copula fitting from empirical probabilities (use `tsCopulaExtremes` instead in most cases)
- `tsCopulaRnd(family, copulaPar, N, uProb)` – Generate random samples from fitted copula
- `tsCopulaCdfFromSamples(u, Usample, return_se=False)` – Empirical CDF from sample points
- `tsCopulaSampleJointPeaksMultiVariatePruning(inputtimestamps, inputtimeseries, **kwargs)` – Advanced event pairing with pruning (called internally by `tsCopulaExtremes` for GPD margins)

### Year-Extremes Functions (GEV margins)
- `tsCopulaYearExtrFit(retPeriod, retLev, yMax, **kwargs)` – Fit copula to annual extremes
- `tsCopulaYearExtrRnd(retPeriod, retLev, copulaParam, nResample, **kwargs)` – Generate samples from year-extremes copula
- `tsCopulaYearExtrDistribution(retPeriod, copulaParam, computeCdf=False)` – Year-extremes joint distribution
- `tsCopulaYearExtrGetMltvrtRetPeriod(randomSample, level)` – Multivariate return period for annual extremes
- `tsCopulaYearExtrPlotSctrBivar(monteCarloRsmpl, yMaxLevel, **kwargs)` – Bivariate scatter (annual extremes)
- `tsCopulaYearExtrPlotSctrTrivar(monteCarloRsmpl, yMaxLevel, **kwargs)` – Trivariate scatter (annual extremes)
- `tsCopulaYearExtrPlotJdistTrivar(retLev, jdist, **kwargs)` – Joint distribution plot (annual extremes, trivariate)

### Helper Functions
- `tsCopulaGetFamilyId(copulaFamily)` – Get numeric ID from copula family name
- `tsCopulaGetFamilyFromId(familyId)` – Get copula family name from numeric ID
- `tsRankmax(X)` – One-based ranks with ties assigned the maximum rank (used for pseudo-observations)
- `tsPseudoObservations(X)` – Convert samples to pseudo-observations via empirical CDF, divided by (n+1)

### Trivariate Gumbel C-vine
- `tsGumbelCVine` – Class implementing a root-first C-vine of Gumbel pair-copulas for trivariate dependence: `cvineOrder(alpha)` derives the vine order from the theta matrix, `fit(uProb, alpha, order)` fits all pair-copulas using pseudo-observations, `simulate(N, order, theta)` samples from the fitted vine. Used automatically by `tsCopulaRnd` when a Gumbel copula has more than two variables.

---

## Copula Families Implemented

All four families listed in the MATLAB documentation are implemented in
`tsEvaMultivariate.py`:

| Family | Fitting method (in `tsCopulaFit`) | Sampling (in `tsCopulaRnd`) |
|--------|-----------------------------------|------------------------------|
| Gaussian | Spearman correlation matrix of pseudo-observations | Multivariate normal → norm.cdf transform |
| Gumbel | Kendall's tau per pair, inverted to theta (negative taus clipped to 0; infinities replaced with 1) | Bivariate: `_gumbel_copula_rnd`; trivariate+: `tsGumbelCVine` C-vine |
| Clayton | Kendall's tau per pair, inverted to theta | Iwata–Kojima inversion sampling |
| Frank | Kendall's tau per pair, inverted numerically (no closed-form inverse; Brent root-finding) | Inversion sampling with exponential transform |

Note: Clayton and Frank are implemented for bivariate use — `tsCopulaRnd` extracts a single theta from the parameter matrix (`copulaPar[0, 1]`). Trivariate dependence is supported through the Gumbel C-vine path.

---

## Common Copula Workflow

### 1. Copula analysis object

```python
copulaAnalysis = tsm.tsCopulaExtremes(
    timeSeriesRiver[:, 0],                          # timestamps (MATLAB datenum)
    np.column_stack((timeSeriesRiver[:, 1],         # variable 1
                     timeSeriesSWH[:, 1])),          # variable 2
    minPeakDistanceInDaysMonovarSampling=minDeltaUnivarSampli,
    maxPeakDistanceInDaysMultivarSampling=maxDeltaMultivarSampli,
    copulaFamily=copulaFamily,
    transfType=transfType,
    timewindow=timeWindowJointDist,
    ciPercentile=ciPercentile,
    potPercentiles=potPercentiles,
    marginalDistributions=marginalDistributions,     # 'gpd' or 'gev'
    samplingOrder=samplingOrder,                    # 0-based indices (Python)
    timeVaryingCopula=True,
    evdType=evdType                                 # e.g. ['GEV', 'GPD']
)
```

- `inputtimestamps` must be numeric MATLAB datenum values (ordinal day + fractional day + 366 offset), as loaded from `.mat` files via `scipy.io.loadmat`
- each column of `inputtimeseries` represents one variable
- non-stationarity is handled through the TS transformation internally

### 2. Monte Carlo simulation

```python
monteCarloAnalysis = tsm.tsCopulaMontecarlo(
    copulaAnalysis,
    nResample=1000,
    timeIndex='middle'
)
```

Optional non-stationarity control: `nonStationarity='margins'` applies
time-varying parameters to the margins only (used in caseStudy03).

### 3. Goodness-of-fit

```python
gofStatistics = tsm.tsCopulaGOFNonStat(copulaAnalysis, monteCarloAnalysis)
```

Optional smoothing:

```python
gofStatistics = tsm.tsCopulaGOFNonStat(copulaAnalysis, monteCarloAnalysis, smoothInd=10)
```

### 4. Return periods (bivariate only)

```python
retPerAnalysis = tsm.tsCopulaComputeBivarRP(copulaAnalysis, monteCarloAnalysis)
```

### 5. Plotting

Bivariate:

```python
axxArray = tsm.tsCopulaPlotBivariate(
    copulaAnalysis, monteCarloAnalysis,
    gofStatistics=gofStatistics,
    retPerAnalysis=retPerAnalysis,
    ylbl=['River discharge (m^3s^{-1})', 'SWH (m)']
)
```

Trivariate:

```python
axxArray = tsm.tsCopulaPlotTrivariate(
    copulaAnalysis, monteCarloAnalysis2,
    gofStatistics=gofStatistics,
    varLabels=['Loc 1 - SWH (m)', 'Loc 2 - SWH (m)', 'Loc 3 - SWH (m)']
)
```

---

## Case Study 01 – Compound Flooding (Bivariate, GPD)

File: `test_multivariate/caseStudy01.py`
Variables: river discharge and significant wave height
Copula: Gumbel
Marginal distributions: GPD
Transformation: `trendlinear`

### Key parameters

```python
ciPercentile = [99, 99]
potPercentiles = [[95.0], [99.0]]

timeWindowJointDist = 14610          # ≈ 365.25 * 40 days

minDeltaUnivarSampli = [30, 30]
maxDeltaMultivarSampli = 45

copulaFamily = 'gumbel'
transfType = 'trendlinear'
marginalDistributions = 'gpd'
samplingOrder = [1, 0]               # 0-based (Python); MATLAB used [2,1]
evdType = ['GEV', 'GPD']             # GPD active for this case
```

### Full code:

```python
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.interpolate import interp1d
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file output
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tsEvaMultivariate as tsm
np.random.seed(42)

# =============================================================================
# Data Loading and Preparation
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data = loadmat(os.path.join(script_dir, "data", "caseStudy01_data.mat"))
timeSWH = data['timeSWH'].flatten()
timeRiverDisch = data['timeRiverDisch'].flatten()
riverineDischarge = data['riverineDischarge'].flatten()
SWH = data['SWH'].flatten()

# Find the overlapping part of both data sources; define a 3-hourly time frame
t_start = max(timeSWH[0], timeRiverDisch[0])
t_end = min(timeSWH[-1], timeRiverDisch[-1])
timeCommon = np.arange(t_start, t_end + (3/24)/2, 3/24)

# Interpolate river and SWH data using the common time frame
indexGoodData = ~np.isnan(riverineDischarge)
indexGoodDataW = ~np.isnan(SWH)

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
    evdType=evdType
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
```

---

## Case Study 02 – Spatial Wave Extremes (Trivariate, GPD)

File: `test_multivariate/caseStudy02.py`
Variables: significant wave height at three locations (Marshall Islands, 3-hourly SWH, 1950–2020)
Copula: Gumbel (non-stationary)
Marginal distributions: GPD
Transformation: `trendlinear`

### Key parameters

```python
ciPercentile = [99, 99, 99]
potPercentiles = [[99.0], [99.0], [99.0]]

timeWindowNonStat = 365 * 40

minDeltaUnivarSampli = [0.5, 0.5, 0.5]
maxDeltaMultivarSampli = 0.5

copulaFamily = 'gumbel'
transfType = 'trendlinear'
peakType = 'allExceedThreshold'
```

### Two-stage Monte Carlo (large for statistics, small for plotting)

```python
monteCarloAnalysis1 = tsm.tsCopulaMontecarlo(copulaAnalysis, nResample=10000, timeIndex='middle')
monteCarloAnalysis2 = tsm.tsCopulaMontecarlo(copulaAnalysis, nResample=300,  timeIndex='middle')
```

### Full code:

```python
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file output
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tsEvaMultivariate as tsm
np.random.seed(42)

# =============================================================================
# Case Study 02 — Bahmanpour et al., 2025
# Trivariate spatial dependence of SWH across three Marshall-Islands locations.
# 3-hourly SWH, 1950-2020, non-stationary Gumbel copula with GPD margins.
# =============================================================================

# timeAndSeries{1,2,3}: col0 = time (MATLAB datenum), col1 = SWH at each location
script_dir = os.path.dirname(os.path.abspath(__file__))
data = loadmat(os.path.join(script_dir, "data", "caseStudy02_data.mat"))
timeAndSeries1 = data['timeAndSeries1']
timeAndSeries2 = data['timeAndSeries2']
timeAndSeries3 = data['timeAndSeries3']

# =============================================================================
# Parameter Definitions (match caseStudy02.m exactly)
# =============================================================================
ciPercentile = [99, 99, 99]
potPercentiles = [[99.0], [99.0], [99.0]]
timeWindowNonStat = 365 * 40
minDeltaUnivarSampli = [0.5, 0.5, 0.5]
maxDeltaMultivarSampli = 0.5
copulaFamily = 'gumbel'
transfType = 'trendlinear'
peakType = 'allExceedThreshold'

# =============================================================================
# Analysis and Visualization (Using tsEvaMultivariate Library)
# =============================================================================

# 1. Copula Extremes Analysis (trivariate)
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
    copulaAnalysis, monteCarloAnalysis2,
    gofStatistics=gofStatistics,
    varLabels=['Loc 1 - SWH (m)', 'Loc 2 - SWH (m)', 'Loc 3 - SWH (m)'],
)

# Save the figure as PNG
plt.savefig('CaseStudy02_output.png', dpi=150, bbox_inches='tight')
print('Figure saved to CaseStudy02_output.png')
plt.show()
```

---

## Case Study 03 – Temperature and Drought (Bivariate, GEV)

File: `test_multivariate/caseStudy03.py`
Variables: surface temperature and SPEI
Copula: Gumbel
Marginal distributions: GEV
Transformation: `trendlinear`

### Key parameters

```python
marginalDistributions = 'gev'
peakType = 'allExceedThreshold'

ciPercentile = [99, 99]
potPercentiles = [[75.0], [97.0]]

timeWindowNonStat = 365 * 35

minDeltaUnivarSampli = [30, 30]
maxDeltaMultivarSampli = 12 * 30

evdType = ['GEV', 'GEV']
```

### Monte Carlo with margins-only non-stationarity

```python
monteCarloAnalysis = tsm.tsCopulaMontecarlo(
    copulaAnalysis,
    nResample=10000,
    timeIndex='middle',
    nonStationarity='margins'
)
```

### Full code:

```python
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file output
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tsEvaMultivariate as tsm
np.random.seed(42)

# =============================================================================
# Data Loading and Preparation
# =============================================================================
# timeAndSeries1: col0=time, col1=temperature
# timeAndSeries2: col0=time, col1=SPEI
script_dir = os.path.dirname(os.path.abspath(__file__))
data = loadmat(os.path.join(script_dir, "data", "caseStudy03_data.mat"))
timeAndSeries1 = data['timeAndSeries1']
timeAndSeries2 = data['timeAndSeries2']

# =============================================================================
# Parameter Definitions
# =============================================================================
ciPercentile = [99, 99]
potPercentiles = [[75.0], [97.0]]
timeWindowNonStat = 365 * 35
minDeltaUnivarSampli = [30, 30]
maxDeltaMultivarSampli = 12 * 30
copulaFamily = 'gumbel'
transfType = 'trendlinear'
peakType = 'allExceedThreshold'
marginalDistributions = 'gev'

evdType = ['GEV', 'GEV']

# =============================================================================
# Analysis and Visualization (Using tsEvaMultivariate Library)
# =============================================================================

# 1. Copula Extremes Analysis
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
```

---

## Notes and Conventions

- `copulaFamily` is passed as a plain string in Python (e.g. `'gumbel'`). The MATLAB cell-array form (`{'Gumbel'}`) has no equivalent — the Python API accepts strings only, case-insensitive internally.
- Two Monte Carlo runs (large for statistics, small for plotting) are recommended and used in cases 02 and 03.
- GEV margins are fully supported in copula workflows when block-maxima logic is required (caseStudy03). For GEV marginals, joint extremes are annual maxima of each series — no threshold-based multivariate pruning is performed.
- Non-stationarity can be applied to margins only via `nonStationarity='margins'` in the Monte Carlo step.
- **Timestamps**: all case studies work directly with MATLAB datenum values loaded from `.mat` files (ordinal day + fractional day + 366 offset). Conversion helpers exist in `tsEva.py`: `datetime_to_datenum(dt)` for a single datetime, `tsEvaPandasDate2DateNum(dates)` for pandas Series/Index.
- **GEV sign convention**: internally the codebase uses MATLAB's shape parameter ε (epsilon) — e.g. `params['epsilon']` in marginal analysis dictionaries. When interfacing with scipy, remember that `scipy.stats.genextreme` uses c = -ε. GPD needs no sign flip: `genpareto.cdf(x, c=shapeParam, loc=threshold, scale=sigma)` maps directly.
- **Reproducibility**: caseStudy03 passes `mcSeed=42` to both Monte Carlo runs; the seed was chosen empirically so the MC scatter is visually closest to the MATLAB reference output.
- **Clayton/Frank are bivariate in practice** — `tsCopulaRnd` extracts a single theta from the parameter matrix for these families. Trivariate dependence goes through the Gumbel C-vine (`tsGumbelCVine`).
