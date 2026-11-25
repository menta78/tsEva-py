"""
Synthetic demonstration that the ex-post correction engages when unpaired
(marginal-only) exceedances exist — TRIVARIATE version.

We build three daily series over 60 years:
  - X: a GPD-tailed series (heavy right tail)
  - Y: a GPD-tailed series, WEAKLY correlated with X (rho ~ 0.3)
  - Z: a GPD-tailed series, MODERATELY correlated with X (rho ~ 0.5)
        and weakly with Y (rho ~ 0.2)

Because the correlations are weak/moderate, many POT peaks occur without
simultaneous POT peaks in all variables -> jointextremes2 is non-empty
-> N10/N01 > 0 -> w < 1 -> the correction actually rescales the copula params.

We run the analysis twice:
  1. Gumbel  -> compare theta before/after ex-post correction (per pair)
  2. Gaussian -> compare rho before/after ex-post correction (per pair)
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tsEvaMultivariate as tsm
import tsEva
import pandas as pd
from scipy.stats import spearmanr

np.random.seed(7)

# ---------------------------------------------------------------------------
# 1. Build timestamps: daily, 60 years
# ---------------------------------------------------------------------------
start = pd.Timestamp('1960-01-01')
n_days = 60 * 365
dates = pd.date_range(start, periods=n_days, freq='D')
timestamps = tsEva.tsEvaPandasDate2DateNum(dates)

# ---------------------------------------------------------------------------
# 2. Build three weakly/moderately-correlated GPD-tailed series
# ---------------------------------------------------------------------------
# Target correlation structure (moderately correlated, not too tight):
#   X-Y: 0.3, X-Z: 0.5, Y-Z: 0.2
# Build via Cholesky of the target correlation matrix.
R_target = np.array([
    [1.0, 0.3, 0.5],
    [0.3, 1.0, 0.2],
    [0.5, 0.2, 1.0]
])
L = np.linalg.cholesky(R_target)
Z = np.random.standard_normal((n_days, 3)) @ L.T   # (n_days, 3) correlated

# Give each a GPD-like heavy right tail via a lognormal transform
def gpd_tail(z):
    return np.exp(0.6 * z)

X = gpd_tail(Z[:, 0])
Y = gpd_tail(Z[:, 1])
Zs = gpd_tail(Z[:, 2])

# Add a mild deterministic trend so the non-stationary transform has something to do
trend = np.linspace(0, 1.0, n_days)
X = X + 0.2 * trend
Y = Y + 0.2 * trend
Zs = Zs + 0.2 * trend

inputtimeseries = np.column_stack((X, Y, Zs))
nVars = 3
varNames = ['X', 'Y', 'Z']

# ---------------------------------------------------------------------------
# 3. Run the analysis and inspect the ex-post correction
# ---------------------------------------------------------------------------
def run_and_report(family):
    print(f"\n{'='*70}\n  FAMILY = {family.upper()}  (trivariate)\n{'='*70}")
    ca = tsm.tsCopulaExtremes(
        timestamps,
        inputtimeseries,
        copulaFamily=family,
        marginalDistributions='gpd',
        transfType='trendlinear',
        timewindow=365.25 * 20,          # 20-yr windows
        timeSlide=365.25,
        potPercentiles=[[99.0], [99.0], [99.0]],
        ciPercentile=[99, 99, 99],
        minPeakDistanceInDaysMonovarSampling=[1.0, 1.0, 1.0],
        maxPeakDistanceInDaysMultivarSampling=2.0,
        peakType='anyexceedthreshold',
        timeVaryingCopula=True,
    )
    cp = ca['copulaParam']
    corr = ca.get('exPostCorrection')
    print("exPostCorrection present:", corr is not None)
    if corr is None:
        return

    N11, N10, N01 = corr['N11'], corr['N10'], corr['N01']
    nWin = N11.shape[0]
    print(f"nWindows = {nWin}")

    # Aggregate counts across all windows, per pair
    pairs = [(0, 1), (0, 2), (1, 2)]
    print(f"\n--- Aggregate exceedance counts (all windows, per pair) ---")
    for (i, j) in pairs:
        t11 = N11[:, i, j].sum()
        t10 = N10[:, i, j].sum()
        t01 = N01[:, i, j].sum()
        denom = 2*t11 + t10 + t01
        w = (2*t11)/denom if denom > 0 else float('nan')
        print(f"  {varNames[i]}-{varNames[j]}:  N11={t11:.0f}  N10={t10:.0f}  N01={t01:.0f}  w={w:.4f}")

    # Before/after per pair
    before = cp.get('rhoBeforeExPostCorrection')
    after = cp.get('rho')
    label = 'theta' if family == 'gumbel' else 'rho'
    print(f"\n--- {label} BEFORE vs AFTER ex-post correction (per pair, mean over windows) ---")
    print(f"{'pair':>6} | {'before':>8} | {'after':>8} | {'change':>8} | {'|before|':>8} | {'|after|':>8}")
    for (i, j) in pairs:
        b_all = np.array([before[k][i, j] for k in range(nWin)])
        a_all = np.array([after[k][i, j] for k in range(nWin)])
        print(f"{varNames[i]}-{varNames[j]:>3} | {b_all.mean():>8.4f} | {a_all.mean():>8.4f} | "
              f"{a_all.mean()-b_all.mean():>+8.4f} | {np.abs(b_all).mean():>8.4f} | {np.abs(a_all).mean():>8.4f}")

    # Show a few windows in detail
    print(f"\n--- {label} per window (pair X-Y) ---")
    print(f"{'win':>4} | {'before':>8} | {'after':>8} | {'change':>8}")
    for k in range(min(nWin, 8)):
        b = before[k][0, 1]
        a = after[k][0, 1]
        print(f"{k:>4} | {b:>8.4f} | {a:>8.4f} | {a-b:>+8.4f}")
    if nWin > 8:
        print(f"     ... ({nWin - 8} more windows)")

# ---------------------------------------------------------------------------
# 3b. Run the analysis
# ---------------------------------------------------------------------------
run_and_report('gumbel')
run_and_report('gaussian')

# ---------------------------------------------------------------------------
# 4. Directly inspect the sampling to understand the peak structure
# ---------------------------------------------------------------------------
print(f"\n{'='*70}\n  SAMPLING DIAGNOSTICS\n{'='*70}")
marginalAnalysis = []
samplingThresholdPrct = np.zeros(nVars)
for ii in range(nVars):
    timeAndSeries = np.column_stack((timestamps, inputtimeseries[:, ii]))
    nonStatEvaParams, statTransfData, is_valid = tsEva.tsEvaNonStationary(
        timeAndSeries, 365.25*20, transfType='trendlinear',
        ciPercentile=99, potPercentiles=[99.0], potEventsPerYear=-1,
        minPeakDistanceInDays=1.0, evdType=['GEV','GPD'])
    marginalAnalysis.append((nonStatEvaParams, statTransfData))
    p = nonStatEvaParams[1]['parameters']
    print(f"  var {ii} ({varNames[ii]}) GPD: eps={p['epsilon']:.3f}, "
          f"sigma[0]={p['sigma'][0]:.4f}, pct={p['percentile']}, nPeaks={p['nPeaks']}")
    samplingThresholdPrct[ii] = p.get('percentile', np.nan) if p else np.nan

statInputTimeSeries = np.column_stack([ma[1].stationarySeries for ma in marginalAnalysis])
ts_ = marginalAnalysis[0][1].timeStamps

sa = tsm.tsCopulaSampleJointPeaksMultiVariatePruning(
    ts_, statInputTimeSeries,
    samplingThresholdPrct=samplingThresholdPrct,
    minPeakDistanceInDaysMonovarSampling=[1.0, 1.0, 1.0],
    maxPeakDistanceInDaysMultivarSampling=2.0,
    marginalAnalysis=marginalAnalysis,
    marginalDistributions='gpd',
    peakType='anyexceedthreshold',
    samplingOrder=[])

je = sa['jointextremes']; je2 = sa['jointextremes2']
print(f"jointextremes  (all-exceed / joint peaks): {je.shape[0] if je.size else 0} events")
print(f"jointextremes2 (any-exceed / non-peaks)  : {je2.shape[0] if je2.size else 0} events")

# How many POT peaks per variable in the stationary series?
from scipy.signal import find_peaks
for ii in range(nVars):
    locs, _ = find_peaks(statInputTimeSeries[:, ii], distance=1)
    thr = samplingThresholdPrct[ii]
    n_above = np.sum(statInputTimeSeries[locs, ii] >= thr)
    print(f"  var {ii} ({varNames[ii]}): {len(locs)} POT peaks, {n_above} above threshold")

# Cross-correlation of the raw series
print(f"\nraw Pearson correlation matrix:")
C = np.corrcoef(inputtimeseries, rowvar=False)
for i in range(nVars):
    print("  " + "  ".join(f"{varNames[j]:>6}" for j in range(nVars)))
    print("  " + "  ".join(f"{C[i,j]:>6.3f}" for j in range(nVars)))
    print()
print("raw Spearman correlations:")
for i in range(nVars):
    for j in range(i+1, nVars):
        print(f"  {varNames[i]}-{varNames[j]}: {spearmanr(inputtimeseries[:,i], inputtimeseries[:,j])[0]:.3f}")

print("\nDone.")
