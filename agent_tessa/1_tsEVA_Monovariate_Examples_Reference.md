# tsEVA 2.0 - Monovariate Examples Reference (Python)

This document contains all example scripts for monovariate extreme value analysis using the tsEva-py framework — the Python port of the MATLAB tsEVA toolbox.

---

## Quick Function Index

### Analysis Functions
- `tsEvaStationary(time_and_series, **kwargs)` - Stationary EVA (GEV/GPD) on time series
- `tsEvaNonStationary(timeAndSeries, timeWindow, **kwargs)` - Non-stationary EVA with transformation
- `tsEvaSampleData(ms, **kwargs)` - POT sampling and block maxima computation
- `tsGetPOT(ms, pcts, desiredEventsPerYear, **kwargs)` - Peaks-over-threshold extraction

### Return Level Computation
- `tsEvaComputeReturnLevelsGEVFromAnalysisObj(nonStationaryEvaParams, returnPeriodsInYears, **kwargs)` - Compute GEV return levels from analysis object
- `tsEvaComputeReturnLevelsGPDFromAnalysisObj(nonStationaryEvaParams, returnPeriodsInYears, **kwargs)` - Compute GPD return levels from analysis object
- `tsEvaComputeRLsGEVGPD(nonStationaryEvaParams, RPgoal, timeIndex, trans=None)` - Combined GEV/GPD return level computation

### Plotting Functions
- `tsEvaPlotReturnLevelsGEVFromAnalysisObj(nonStationaryEvaParams, timeIndex, **kwargs)` - Plot GEV return levels at a given time index
- `tsEvaPlotReturnLevelsGPDFromAnalysisObj(nonStationaryEvaParams, timeIndex, **kwargs)` - Plot GPD return levels at a given time index
- `tsEvaPlotSeriesTrendStdDevFromAnalysisObj(nonStationaryEvaParams, stationaryTransformData, **kwargs)` - Plot series with trend and confidence interval
- `tsEvaPlotGEVImageScFromAnalysisObj(X, nonStationaryEvaParams, stationaryTransformData, **kwargs)` - 2D time-varying GEV distribution plot
- `tsEvaPlotGPDImageScFromAnalysisObj(Y, nonStationaryEvaParams, stationaryTransformData, **kwargs)` - 2D time-varying GPD distribution plot
- `tsEvaPlotGEV3DFromAnalysisObj(X, nonStationaryEvaParams, stationaryTransformData, **kwargs)` - 3D GEV visualization
- `tsEvaPlotTransfToStatFromAnalysisObj(nonStationaryEvaParams, stationaryTransformData, **kwargs)` - Plot transformed stationary series

### Diagnostic Plots (Series + Extremes)
- `tsPlotSeriesPotGPDRetLevFromAnalysisObj(nonStationaryEvaParams, stationaryTransformData, **kwargs)` - Series + POT peaks + time-varying GPD return levels
- `tsPlotSeriesYearMaxGEVRetLevFromAnalysisObj(nonStationaryEvaParams, stationaryTransformData, **kwargs)` - Series + annual maxima + time-varying GEV return levels
- `tsPlotSeriesPotGPDRetLevStationary(statEvaParams, timeAndSeries, **kwargs)` - Stationary version: series + POT + GPD return levels
- `tsPlotSeriesYearMaxGEVRetLevStationary(statEvaParams, timeAndSeries, **kwargs)` - Stationary version: series + annual maxima + GEV return levels

### Timestamp Utilities
- `datetime_to_datenum(dt)` - Convert a single Python datetime to MATLAB datenum format (ordinal days)
- `tsEvaPandasDate2DateNum(dates)` - Convert pandas Series/Index of dates to datenum array
- `datenum_to_datetime(datenum_value)` - Inverse conversion: datenum back to Python datetime

---

## Example 1: Stationary EVA

**File:** `exampleEVAStationary.py`

**Purpose:** Demonstrates stationary extreme value analysis (GEV and GPD) on a time series.

**Key Features:**
- Loads data from CSV with year/month/day/hour columns, converts to datenum timestamps via `tsEvaPandasDate2DateNum`
- Stationary fit of GEV and GPD distributions using `tsEvaStationary`
- Return level computation at [10, 20, 50, 100] year return periods
- Fixed threshold POT analysis (`doSampleData=False`, explicit `potThreshold`)
- Gumbel distribution as alternative to full GEV (`gevType='Gumbel'`)
- Diagnostic plots showing series with annual maxima or POT peaks overlaid on return levels

**Main Functions Used:**
- `tsEvaStationary()` - performs stationary EVA
- `tsEvaComputeReturnLevelsGEVFromAnalysisObj()` / `tsEvaComputeReturnLevelsGPDFromAnalysisObj()` - compute return levels
- `tsEvaPlotReturnLevelsGEVFromAnalysisObj()` / `tsEvaPlotReturnLevelsGPDFromAnalysisObj()` - plot return levels
- `tsPlotSeriesYearMaxGEVRetLevStationary()` / `tsPlotSeriesPotGPDRetLevStationary()` - diagnostic plots

**Key Parameters:**
- `minPeakDistanceInDays` - minimum distance between peaks (required)
- `potThreshold` - fixed threshold for POT when `doSampleData=False`
- `gevType='Gumbel'` - fit Gumbel instead of full GEV

**Code excerpt:**
```python
data = pd.read_csv(data_file_name, header=None, names=['year','month','day','hour','value'])
dates = pd.to_datetime(data[['year','month','day','hour']])
timestamps = tsEvaPandasDate2DateNum(dates)
timeAndSeries = np.column_stack([timestamps, data['value'].values])

statEvaParams = tsEvaStationary(timeAndSeries, minPeakDistanceInDays=3)

rlevGEV, rlevGEVErr = tsEvaComputeReturnLevelsGEVFromAnalysisObj(statEvaParams, [10, 20, 50, 100])
hndl = tsEvaPlotReturnLevelsGEVFromAnalysisObj(statEvaParams, 0, ylim=[0.5, 1.5])

# Fixed threshold POT
potThreshold = np.percentile(timeAndSeries[:, 1], 98)
statEvaParams = tsEvaStationary(timeAndSeries, minPeakDistanceInDays=3, doSampleData=False, potThreshold=potThreshold)

# Gumbel variant
statEvaParams = tsEvaStationary(timeAndSeries, minPeakDistanceInDays=3, gevType='Gumbel')
```

---

## Example 2: Non-Stationary EVA with Trend and Seasonal Components

**File:** `exampleGenerateSeriesEVAGraphs.py`

**Purpose:** Analyzes non-stationary time series with trend and seasonal components using the transformed-stationary approach.

**Key Features:**
- Transformed-stationary approach for non-stationary data
- Trend-only transformation (`transfType='trend'`)
- Seasonal transformation (`transfType='seasonal'`)
- 2D visualization of time-varying GEV/GPD distributions via `tsEvaPlotGEVImageScFromAnalysisObj` / `tsEvaPlotGPDImageScFromAnalysisObj`
- Return level computation at specific time indices (e.g., index 1000)
- Stationary series diagnostic plot

**Main Functions Used:**
- `tsEvaNonStationary()` - performs non-stationary EVA with transformation, returns `(nonStatEvaParams, statTransfData, isValid)`
- `tsEvaPlotSeriesTrendStdDevFromAnalysisObj()` - plots series with trend and confidence interval
- `tsEvaPlotGEVImageScFromAnalysisObj()` / `tsEvaPlotGPDImageScFromAnalysisObj()` - 2D time-varying distribution plots
- `tsEvaPlotReturnLevelsGEVFromAnalysisObj()` / `tsEvaPlotReturnLevelsGPDFromAnalysisObj()` - return level plot at specific time index
- `tsEvaPlotTransfToStatFromAnalysisObj()` - plots transformed stationary series

**Key Parameters:**
- `timeWindow` - time window for detecting non-stationarity (e.g., `365.25 * 6` for 6 years)
- `transfType` - transformation type: `'trend'`, `'seasonal'`, `'trendlinear'`, `'trendCIPercentile'`, `'seasonalCIPercentile'`
- `minPeakDistanceInDays` - minimum distance between peaks (required)

**Code excerpt:**
```python
nonStatEvaParams, statTransfData, isValid = tsEvaNonStationary(
    timeAndSeries, timeWindow, transfType='trend', minPeakDistanceInDays=3)

hndl = tsEvaPlotSeriesTrendStdDevFromAnalysisObj(nonStatEvaParams, statTransfData, ylabel='Lvl (m)', title='Hebrides')
hndl = tsEvaPlotGEVImageScFromAnalysisObj(wr, nonStatEvaParams, statTransfData, ylabel='Lvl (m)')

timeIndex = 1000
hndl = tsEvaPlotReturnLevelsGEVFromAnalysisObj(nonStatEvaParams, timeIndex, ylim=[0.5, 1.5])

# Seasonal variant
nonStatEvaParams, statTransfData, isValid = tsEvaNonStationary(
    timeAndSeries, timeWindow, transfType='seasonal', minPeakDistanceInDays=3)
```

---

## Example 3: Confidence Interval via Moving Percentile

**File:** `exampleGenerateSeriesEVAGraphs_ciPercentile.py`

**Purpose:** Estimates long-term extreme variations using moving percentile instead of moving standard deviation for amplitude estimation.

**Key Features:**
- Uses moving percentile to estimate amplitude (more sensitive to extreme changes)
- Trend with CI percentile: `transfType='trendCIPercentile'`, `ciPercentile=98`
- Seasonal with CI percentile: `transfType='seasonalCIPercentile'`, `ciPercentile=98`
- Broader confidence intervals but better extreme modeling than std-dev approach

**Additional Parameters:**
- `ciPercentile` - percentile for confidence interval estimation (e.g., 98)

**Code excerpt:**
```python
nonStatEvaParams, statTransfData, isValid = tsEvaNonStationary(
    timeAndSeries, timeWindow, transfType='trendCIPercentile', ciPercentile=98, minPeakDistanceInDays=3)

rlevGEV, rlevGEVErr = tsEvaComputeReturnLevelsGEVFromAnalysisObj(nonStatEvaParams, [10, 20, 50, 100], timeIndex=999)
hndl = tsEvaPlotReturnLevelsGEVFromAnalysisObj(nonStatEvaParams, 999, ylim=[0.6, 1.1])

nonStatEvaParams, statTransfData, isValid = tsEvaNonStationary(
    timeAndSeries, timeWindow, transfType='seasonalCIPercentile', ciPercentile=98, minPeakDistanceInDays=3)
```

---

## Example 4: Linear Trend with Multiple POT Percentiles (Adriatic TWL)

**File:** `exampleGenerateSeriesEVAGraphs_trendLinear.py`

**Purpose:** Analyzes total water level (TWL) with linear trend transformation for coastal flood applications.

**Key Features:**
- Loads data from `.mat` file via `scipy.io.loadmat`
- Linear trend transformation: `transfType='trendlinear'`, requires `ciPercentile`
- Multiple POT percentiles scanned: `potPercentiles=[97, 97.5, 98, 98.5, 99]` to match target events/year
- Diagnostic plots showing series + POT peaks + time-varying GPD return levels, and series + annual maxima + GEV return levels
- Return level comparison at beginning vs end of series

**Main Functions Used:**
- `tsPlotSeriesPotGPDRetLevFromAnalysisObj()` - series + POT + GPD return levels (diagnostic)
- `tsPlotSeriesYearMaxGEVRetLevFromAnalysisObj()` - series + annual maxima + GEV return levels (diagnostic)

**Code excerpt:**
```python
mat = scipy.io.loadmat(os.path.join(script_dir, "data", "EOatSEE.mat"))
tm  = mat['tm'].flatten()
twl = mat['twl'].flatten()
timeAndSeries = np.column_stack((tm, twl))

nonStatEvaParams, statTransfData, isValid = tsEvaNonStationary(
    timeAndSeries, timeWindow,
    transfType='trendlinear',
    ciPercentile=99,
    potPercentiles=list(np.arange(97, 99.5, 0.5)),
    minPeakDistanceInDays=14)

hndl = tsPlotSeriesPotGPDRetLevFromAnalysisObj(nonStatEvaParams, statTransfData, ylabel='TWL (m)')
hndl = tsPlotSeriesYearMaxGEVRetLevFromAnalysisObj(nonStatEvaParams, statTransfData, ylabel='TWL (m)')

for timeIndex in [999, len(timeStamps) - 1000]:
    rlevGPD, rlevGPDErr = tsEvaComputeReturnLevelsGPDFromAnalysisObj(nonStatEvaParams, [5, 10, 30, 100], timeIndex=timeIndex)
```

---

## Example 5: Comparing Moving Std Dev vs Moving Percentile for Amplitude Estimation

**File:** `exampleCompareDifferentCI.py`

**Purpose:** Illustrates that the time-varying amplitude of the signal can be estimated by moving standard deviation or moving percentile — the latter models extremes better but with stronger uncertainty.

**Key Features:**
- Runs same analysis multiple times with different `ciPercentile` values (98, 98.5, 99) vs default std-dev approach
- Compares resulting time-varying GEV plots side by side
- Uses wave height data (`Hs`) from CSV

**Code excerpt:**
```python
# Moving standard deviation (default)
nonStatEvaParams, statTransfData, _ = tsEvaNonStationary(timeAndSeries, timeWindow, minPeakDistanceInDays=3)

# Moving percentile variants
for ciPercentile in [98, 98.5, 99]:
    nonStatEvaParams, statTransfData, _ = tsEvaNonStationary(
        timeAndSeries, timeWindow, transfType='trendCIPercentile', ciPercentile=ciPercentile, minPeakDistanceInDays=3)
```

---

## Example 6: SPI Series — Sparse Peaks, GPD Only

**File:** `exampleSPISeries.py`

**Purpose:** Analyzes Standardized Precipitation Index where peaks are widely separated (at least 5 months apart), making the "5 peaks over threshold per year" concept meaningless.

**Key Features:**
- Large minimum peak distance: `minPeakDistanceInDays = 5 * 30.2` (~151 days)
- GPD-only analysis (`evdType='GPD'`) — skips GEV due to sparse annual maxima
- Single POT percentile: `potPercentile=80` instead of scanning multiple candidates
- Inverts series (`timeAndSeries[:,1] = -timeAndSeries[:,1]`) to analyze droughts as negative SPI values

**Code excerpt:**
```python
timeAndSeries[:, 1] = -timeAndSeries[:, 1]

nonStationaryEvaParams, stationaryTransformData, isValid = tsEvaNonStationary(
    timeAndSeries, timeWindow, minPeakDistanceInDays=5*30.2,
    transfType='trendCIPercentile', ciPercentile=80, potPercentile=80, evdType='GPD')

returnLevels, returnLevelsErr = tsEvaComputeReturnLevelsGPDFromAnalysisObj(nonStationaryEvaParams, [10, 20, 50, 100])
returnLevels = returnLevels * -1  # invert back to original SPI scale
```

---

## Example 7: SPI with Gumbel Distribution (GEV Only)

**File:** `exampleSPISeries_Gumbel.py`

**Purpose:** Fits Gumbel distribution (GEV with shape parameter fixed at 0) to SPI data — suitable when tail behavior suggests exponential decay rather than heavier or lighter tails.

**Key Features:**
- Gumbel as special case of GEV: `gevType='Gumbel'`
- GEV-only analysis (`evdType='GEV'`)
- Same sparse peak structure as Example 6

**Code excerpt:**
```python
nonStationaryEvaParams, stationaryTransformData, isValid = tsEvaNonStationary(
    timeAndSeries, timeWindow, minPeakDistanceInDays=5*30.2,
    transfType='trendCIPercentile', ciPercentile=80, evdType='GEV', gevType='Gumbel')

returnLevels, returnLevelsErr = tsEvaComputeReturnLevelsGEVFromAnalysisObj(nonStationaryEvaParams, [10, 20, 50, 100])
```

---

## Example 8: Annual Maximum Temperature Series (Heat Wave Evolution)

**File:** `exampleTASMaxSeries.py`

**Purpose:** Analyzes annual maximum temperature (TAS) to understand how heat waves evolve over time. Yearly maxima series is ideal for GEV; GPD analysis is meaningless here.

**Key Features:**
- GEV-only analysis (`evdType='GEV'`)
- Low threshold for extremes: `extremeLowThreshold=0.1` — values below this are not considered as potential extremes
- Multiple time point comparisons (e.g., index 26 ≈ year 1995 vs last few indices ≈ year 2095)

**Code excerpt:**
```python
nonStationaryEvaParams, stationaryTransformData, isValid = tsEvaNonStationary(
    timeAndSeries, timeWindow, minPeakDistanceInDays=5*30.2,
    extremeLowThreshold=.1, evdType='GEV')

returnLevels, returnLevelsErr = tsEvaComputeReturnLevelsGEVFromAnalysisObj(nonStationaryEvaParams, [20, 50, 100, 300])

timeIndex = 26  # ~1995
hndl = tsEvaPlotReturnLevelsGEVFromAnalysisObj(nonStationaryEvaParams, timeIndex, ylim=[0, 14])

timeIndex = len(timeAndSeries) - 4  # ~2095
hndl = tsEvaPlotReturnLevelsGEVFromAnalysisObj(nonStationaryEvaParams, timeIndex, ylim=[0, 70])
```

---

## Test Script: Running Percentile Function

**File:** `testTsEvaNanRunningPercentile.py`

**Purpose:** Tests the internal function for computing running percentiles with NaN handling.

**Key Features:**
- Uses `tsEvaRunningMeanTrend()` to compute trend series and window size
- Calls `tsEvaNanRunningPercentile(filledSeries, nRunMn, percent)` directly at multiple percentile levels (80, 90, 95, 98, 99)
- Reports relative error as percentage of mean value

**Code excerpt:**
```python
trendSeries, filledTimeStamps, filledSeries, nRunMn = tsEvaRunningMeanTrend(timeStamps, series, timeWindow)

for percent in [80, 90, 95, 98, 99]:
    rnprcnt, err = tsEvaNanRunningPercentile(filledSeries, nRunMn, percent)
    print(f"Error = {err/np.nanmean(rnprcnt)*100} %")
```

---

## Common Workflow Patterns

### Basic Non-Stationary Analysis
```python
# 1. Load data — timestamps in datenum format (ordinal days), values in column 1
timeAndSeries = np.column_stack([timestamps, values])

# 2. Set parameters
timeWindow = 365.25 * years
minPeakDistanceInDays = days

# 3. Perform analysis
nonStatEvaParams, statTransfData, isValid = tsEvaNonStationary(
    timeAndSeries, timeWindow,
    transfType='trend',  # or 'seasonal', 'trendlinear', 'trendCIPercentile', 'seasonalCIPercentile'
    minPeakDistanceInDays=minPeakDistanceInDays)

# 4. Visualize
tsEvaPlotSeriesTrendStdDevFromAnalysisObj(nonStatEvaParams, statTransfData)
tsEvaPlotGEVImageScFromAnalysisObj(wr, nonStatEvaParams, statTransfData)
tsEvaPlotGPDImageScFromAnalysisObj(wr, nonStatEvaParams, statTransfData)

# 5. Compute return levels at a specific time index
rlevGPD, rlevGPDErr = tsEvaComputeReturnLevelsGPDFromAnalysisObj(
    nonStatEvaParams, [10, 20, 50, 100], timeIndex=timeIndex)
```

### Transformation Types
- `'trend'` - long-term trend via running mean; CI via running std dev (default)
- `'seasonal'` - long-term + seasonal variability (multiplicative); uses monthly maxima for GEV
- `'trendlinear'` - linear trend fit; requires `ciPercentile`; CI via linear fit of a percentile
- `'trendCIPercentile'` - running-mean trend; CI via running percentile; requires `ciPercentile`
- `'seasonalCIPercentile'` - seasonal transformation; CI via running percentile; requires `ciPercentile`

### Distribution Types
- `evdType='GPD'` - GPD only (skips GEV)
- `evdType='GEV'` - GEV only (skips GPD)
- Default: both GEV and GPD fitted

### Special Options
- `gevType='Gumbel'` - fit Gumbel instead of full GEV (shape fixed at 0)
- `potPercentiles=[97, 97.5, 98, 98.5, 99]` - multiple candidate thresholds scanned to match target events/year
- `potThreshold=value` - fixed POT threshold (use with `doSampleData=False` in stationary mode)
- `ciPercentile=98` - percentile for CI estimation (required for `trendlinear`, `trendCIPercentile`, `seasonalCIPercentile`)
- `extremeLowThreshold=0.1` - minimum value threshold; values below this are excluded from extreme analysis

---

## GEV Sign Convention Note

MATLAB tsEVA uses shape parameter ε (epsilon). scipy.stats.genextreme uses c = −ε — the negative of MATLAB's convention. The Python codebase stores fitted parameters under an `'epsilon'` key in the `gevParams` dict, but whether this represents MATLAB-style ε or scipy-style c depends on how `tsEVstatistics` maps them internally during fitting. When comparing parameter values between MATLAB and Python outputs, verify against actual fitted numbers rather than assuming they're directly comparable — a sign flip would produce incorrect return levels if missed.

---

## Timestamps: datenum Format Throughout

The Python port uses MATLAB's datenum format (ordinal days) consistently throughout — `timeAndSeries[:, 0]` contains ordinal day numbers just like in MATLAB. Conversion utilities exist for working with native Python/pandas datetime objects:
- `tsEvaPandasDate2DateNum(dates)` converts pandas Series/Index to datenum array
- `datetime_to_datenum(dt)` converts a single datetime object
- `datenum_to_datetime(datenum_value)` does the inverse

This means all example scripts that load CSV data with year/month/day/hour columns must convert timestamps before passing to tsEva functions. Scripts loading `.mat` files already have datenum values and can use them directly.

---

## References

Mentaschi, L., et al. (2016). The transformed-stationary approach: a generic and simplified methodology for non-stationary extreme value analysis. Hydrol. Earth Syst. Sci., 20, 3527-3547.

Bahmanpour, F., et al. (2025). Transformed-Stationary EVA 2.0: A Generalized Framework for Non-Stationary Joint Extremes Analysis. Hydrol. Earth Syst. Sci. (under review).
