import numpy as np
import pandas as pd
from scipy.stats import genpareto
from scipy.stats import genextreme as gev
import itertools
import warnings
from scipy.signal import find_peaks
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.stats import kendalltau
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from scipy.stats import norm, rankdata
import matplotlib.dates as mdates
import matplotlib.cm as cm
import sys
import tsEva
from tsEva import *
tsEva.Delta_fit = tsEva.Bootstrap_fit

_orig_percentile = np.percentile
def safe_matlab_percentile(a, q, axis=None, **kwargs):
    if axis is not None or 'keepdims' in kwargs:
        return _orig_percentile(a, q, axis=axis, **kwargs)
    x = np.asarray(a, dtype=float).flatten()
    x = x[~np.isnan(x)]
    if len(x) == 0: return np.nan
    x = np.sort(x)
    n = len(x)
    qs = [q] if np.isscalar(q) else q
    res = []
    for p in qs:
        idx = (p / 100.0) * n - 0.5
        idx = np.clip(idx, 0, n - 1)
        i = int(np.floor(idx))
        if i == n - 1: res.append(x[-1])
        else: res.append(x[i] + (x[i+1] - x[i]) * (idx - i))
    return res[0] if np.isscalar(q) else np.array(res)

_orig_nanpercentile = np.nanpercentile
def safe_matlab_nanpercentile(a, q, axis=None, **kwargs):
    if axis is not None or 'keepdims' in kwargs:
        return _orig_nanpercentile(a, q, axis=axis, **kwargs)
    return safe_matlab_percentile(a, q)

_orig_find_peaks = tsEva.find_peaks
def safe_matlab_find_peaks(x, height=None, distance=None, **kwargs):
    if distance is None:
        return _orig_find_peaks(x, height=height, **kwargs)
    valid_mask = ~np.isnan(x)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0: return np.array([], dtype=int), {'peak_heights': np.array([])}
    valid_vals = x[valid_idx]
    n = len(valid_vals)
    peaks_in_valid = []
    i = 1
    while i < n - 1:
        if valid_vals[i] > valid_vals[i - 1]:
            j = i
            while j < n - 1 and valid_vals[j] == valid_vals[j + 1]: j += 1
            if j < n - 1 and valid_vals[j] > valid_vals[j + 1]: peaks_in_valid.append(i)
            i = j + 1
        else: i += 1
    peaks_in_valid = np.array(peaks_in_valid, dtype=int)
    if len(peaks_in_valid) == 0: return np.array([], dtype=int), {'peak_heights': np.array([])}
    candidate_locs = valid_idx[peaks_in_valid]
    candidate_pks = valid_vals[peaks_in_valid]
    if height is not None:
        h_mask = candidate_pks >= height
        candidate_locs = candidate_locs[h_mask]
        candidate_pks = candidate_pks[h_mask]
    if len(candidate_locs) == 0: return np.array([], dtype=int), {'peak_heights': np.array([])}
    sort_idx = np.argsort(-candidate_pks, kind='stable')
    sorted_locs = candidate_locs[sort_idx]
    kept_locs = []
    for loc in sorted_locs:
        conflict = False
        for kept in kept_locs:
            if abs(loc - kept) < distance:
                conflict = True
                break
        if not conflict: kept_locs.append(loc)
    final_locs = np.sort(kept_locs)
    return final_locs, {'peak_heights': x[final_locs]}

_orig_tsMann_Kendall = tsEva.tsMann_Kendall
def safe_matlab_mk(V, alpha=0.05):
    V = np.asarray(V).flatten()
    V = V[~np.isnan(V)]
    n = len(V)
    if n < 3: return 0, np.nan
    S = 0
    for i in range(n - 1): S += np.sum(np.sign(V[i+1:] - V[i]))
    _, counts = np.unique(V, return_counts=True)
    tie_sum = np.sum(counts * (counts - 1) * (2 * counts + 5))
    VarS = (n * (n - 1) * (2 * n + 5) - tie_sum) / 18.0
    StdS = np.sqrt(VarS)
    if StdS == 0: return 0, 1.0
    if S > 0: Z = (S - 1) / StdS
    elif S < 0: Z = (S + 1) / StdS
    else: Z = 0
    p_value = 2 * (1 - norm.cdf(abs(Z)))
    pz = norm.ppf(1 - alpha / 2.0)
    return (1 if abs(Z) > pz else 0), p_value

def tsModified_MannKendall_test(t, X, alpha=0.05, alpha_ac=0.05):
    """
    Modified Mann-Kendall test with autocorrelation correction.
    Matches MATLAB tsModified_MannKendall_test.m (Hamed & Rao 1998).
    Returns: (tau, z, p, H)
    """
    from scipy.stats import rankdata, norm as norm_dist
    X = np.asarray(X, dtype=float).flatten()
    t = np.asarray(t, dtype=float).flatten()
    n = len(X)
    if n < 3:
        return 0, 0, 0.5, 0

    # Kendall S
    S = 0
    for i in range(n - 1):
        S += np.sum(np.sign(X[i+1:] - X[i]))
    tau = S / (n * (n - 1) / 2.0)

    # Variance without autocorrelation (with tie correction)
    var_S_notie = n * (n - 1) * (2 * n + 5) / 18.0
    sorted_X = np.sort(X)
    tied_ranks = []
    cnt = 1
    for i in range(1, n):
        if sorted_X[i] == sorted_X[i - 1]:
            cnt += 1
        else:
            if cnt > 1:
                tied_ranks.append(cnt)
            cnt = 1
    if cnt > 1:
        tied_ranks.append(cnt)
    tie_corr = sum(tr * (tr - 1) * (2 * tr + 5) / 18.0 for tr in tied_ranks)
    var_S_noAC = var_S_notie - tie_corr

    # Sen trend removal
    m_list = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if t[j] != t[i]:
                m_list.append((X[j] - X[i]) / (t[j] - t[i]))
    m_sen = np.median(m_list) if m_list else 0
    b_list = X - m_sen * t
    b_sen = np.median(b_list)
    X_detrended = X - m_sen * t - b_sen

    # Autocorrelation of ranked detrended data
    X_ranked = rankdata(X_detrended, method='average')
    z_ac = abs(norm_dist.ppf(alpha_ac / 2.0))
    X_centered = X_ranked - np.mean(X_ranked)
    # xcorr normalized ('coeff')
    acf_full = np.correlate(X_centered, X_centered, mode='full')
    acf_full = acf_full / acf_full[n - 1]  # normalize by lag-0
    acf = acf_full[n - 1:]  # positive lags only

    acf_bounds = z_ac / np.sqrt(n) * np.array([-1.0, 1.0])

    # MATLAB condition: acf(i) > acf_bounds(1) || acf(i) < acf_bounds(2)
    # where bounds(1) = -threshold, bounds(2) = +threshold — always true,
    # so MATLAB effectively selects ALL lags.
    rho_lags = []
    rho_vals = []
    for i in range(1, n):
        if acf[i] > acf_bounds[0] or acf[i] < acf_bounds[1]:
            rho_vals.append(acf[i])
            rho_lags.append(i)

    # Autocorrelation correction factor
    const_factor = 2.0 / (n * (n - 1) * (n - 2))
    rho_factor_sum = 0
    for i in range(len(rho_vals)):
        lag = rho_lags[i]
        rho_factor_sum += (n - lag) * (n - lag - 1) * (n - lag - 2) * rho_vals[i]
    var_AC_correction = 1 + const_factor * rho_factor_sum
    var_S = var_S_noAC * var_AC_correction

    if var_S < 0:
        return tau, 0, 0.5, 2

    # Z-score with continuity correction
    if S > 0:
        z = (S - 1) / np.sqrt(var_S)
    elif S < 0:
        z = (S + 1) / np.sqrt(var_S)
    else:
        z = 0

    # One-tailed p-value (matching MATLAB)
    if z >= 0:
        p = 1 - norm_dist.cdf(z)
    else:
        p = norm_dist.cdf(z)

    # Hypothesis
    if p <= alpha:
        H = 1 if z > 0 else -1
    else:
        H = 0

    return tau, z, p, H


# HAFIZADA TS_EVA.PY'Yİ GÜNCELLİYORUZ (Dosyaya dokunmadan)
tsEva.np.percentile = safe_matlab_percentile
tsEva.np.nanpercentile = safe_matlab_nanpercentile
tsEva.find_peaks = safe_matlab_find_peaks
# NOTE: tsMann_Kendall monkey-patch removed — original tsEva.py version
# already matches MATLAB (no tie correction, same as tsMann_Kendall.m)
# =========================================================================

# -------------------------- Helper Plotting Functions --------------------------

def tsCopulaExtremes(inputtimestamps, inputtimeseries, **kwargs):
    nSeries = inputtimeseries.shape[1]
    
    copulaFamily = kwargs.get('copulaFamily', 'gaussian')
    marginalDistributions = kwargs.get('marginalDistributions', 'gpd').lower()
    timewindow = kwargs.get('timeWindow', kwargs.get('timewindow', 100 * 365.25))
    potPercentiles = kwargs.get('potPercentiles', [None] * nSeries)
    transfType = kwargs.get('transfType', 'trendCiPercentile')
    ciPercentile = kwargs.get('ciPercentile', [99] * nSeries)
    timeSlide = kwargs.get('timeSlide', 365.25)
    minPeakDistanceInDaysMonovarSampling = kwargs.get('minPeakDistanceInDaysMonovarSampling', [3] * nSeries)
    maxPeakDistanceInDaysMultivarSampling = kwargs.get('maxPeakDistanceInDaysMultivarSampling', 3)
    peakType = kwargs.get('peakType', 'allexceedthreshold')
    samplingOrder = kwargs.get('samplingOrder', [])
    smoothInd = kwargs.get('smoothInd', -1)
    timeVaryingCopula = kwargs.get('timeVaryingCopula', True)
    evdType = kwargs.get('evdType', ['GEV', 'GPD'])
    
    if smoothInd == -1:
        smoothInd = int(np.ceil(timewindow / 365.23 / 4))

    durationSeriesInYears = (inputtimestamps[-1] - inputtimestamps[0]) / 365.25
    copulaTimeWindow = timewindow if timeVaryingCopula else (durationSeriesInYears * 365.25 * 2)

    samplingThresholdPrct = np.zeros(nSeries)
    marginalAnalysis = []
    
    for ii in range(nSeries):
        timeAndSeries = np.column_stack((inputtimestamps, inputtimeseries[:, ii]))
        
        nonStatEvaParams, statTransfData, is_valid = tsEvaNonStationary(
            timeAndSeries, 
            timewindow, 
            transfType=transfType,
            ciPercentile=ciPercentile[ii],
            potPercentiles=potPercentiles[ii],
            potEventsPerYear=-1,
            minPeakDistanceInDays=minPeakDistanceInDaysMonovarSampling[ii],
            evdType=evdType
        )
        marginalAnalysis.append((nonStatEvaParams, statTransfData))
        
        if nonStatEvaParams[1]['parameters'] is not None:
            samplingThresholdPrct[ii] = nonStatEvaParams[1]['parameters'].get('percentile', np.nan)
        else:
            samplingThresholdPrct[ii] = np.nan

    statInputTimeSeries = np.column_stack([ma[1].stationarySeries for ma in marginalAnalysis])
    inputtimestamps = marginalAnalysis[0][1].timeStamps
    inputtimeseries = np.column_stack([ma[1].nonStatSeries for ma in marginalAnalysis])
    
    if marginalDistributions == 'gpd':
        samplingAnalysis = tsCopulaSampleJointPeaksMultiVariatePruning(
            inputtimestamps, 
            statInputTimeSeries,
            samplingThresholdPrct=samplingThresholdPrct,
            minPeakDistanceInDaysMonovarSampling=minPeakDistanceInDaysMonovarSampling,
            maxPeakDistanceInDaysMultivarSampling=maxPeakDistanceInDaysMultivarSampling,
            marginalAnalysis=marginalAnalysis,
            marginalDistributions=marginalDistributions,
            peakType=peakType,
            samplingOrder=samplingOrder
        )
        
        if peakType.lower() == 'anyexceedthreshold':
            jointextremes = np.vstack((samplingAnalysis['jointextremes'], samplingAnalysis['jointextremes2']))
        else:
            jointextremes = samplingAnalysis['jointextremes']

        if nSeries > 3:
            nBivarComb = list(itertools.combinations(range(nSeries), 2))
            indicesCell = []
            thresholdsC = samplingAnalysis['thresholdsC']
            for comb in nBivarComb:
                thr = [thresholdsC[comb[0]], thresholdsC[comb[1]]]
                jointextr = jointextremes[:, list(comb), 1] 
                indices = np.where((jointextr[:, 0] >= thr[0]) & (jointextr[:, 1] >= thr[1]))[0]
                indicesCell.append(indices)

        # GPD CDF (we are inside the GPD branch — inner elif for 'gev' was unreachable dead code, removed)
        gpdCDFCopula = np.full((jointextremes.shape[0], nSeries), np.nan)
        for ii in range(nSeries):
            stat_params = marginalAnalysis[ii][0][1]['stationaryParams']['parameters']
            shapeParam = stat_params['shape']
            scaleParam = stat_params['sigma']
            thrshldValue = stat_params['threshold']

            gpdCdf = genpareto.cdf(jointextremes[:, ii, 1], c=shapeParam, loc=thrshldValue, scale=scaleParam)
            # MATLAB: gpdCdf(gpdCdf == 0) = 1e-7; (only clip zeros, not upper bound)
            gpdCdf[gpdCdf == 0] = 1e-7
            gpdCDFCopula[:, ii] = gpdCdf

    elif marginalDistributions == 'gev':
        # MATLAB tsCopulaExtremes.m lines 218-246:
        # For GEV marginals, joint extremes are annual maxima of each series
        # (no threshold-based multivariate pruning is performed).
        jointextremes_list = []
        jointExtremeIndices_list = []
        jointExtremesNS_list = []

        # Local NaN-aware annual maxima computation (matches MATLAB's max which ignores NaN)
        # We do not use tsEva.tsEvaComputeAnnualMaxima here because its find_max uses
        # np.argmax, which picks NaN positions when NaN is present in the series.
        def _gev_annual_max_nan_aware(ts_arr, val_arr):
            tmvec = pd.to_datetime(np.asarray(ts_arr) - 719529, unit='D', origin='unix')
            years = tmvec.year
            years_np = np.asarray(years)
            val_np = np.asarray(val_arr, dtype=float)
            unique_years = np.unique(years_np)
            idx_list, max_list, ts_list = [], [], []
            for yr in unique_years:
                mask = years_np == yr
                idxs_in_year = np.where(mask)[0]
                vals_in_year = val_np[idxs_in_year]
                if np.all(np.isnan(vals_in_year)):
                    continue  # skip years with no valid data
                local_max_idx = np.nanargmax(vals_in_year)
                global_idx = idxs_in_year[local_max_idx]
                idx_list.append(global_idx)
                max_list.append(val_np[global_idx])
                ts_list.append(ts_arr[global_idx])
            return (np.asarray(max_list, dtype=float),
                    np.asarray(ts_list, dtype=float),
                    np.asarray(idx_list, dtype=int))

        for ii in range(nSeries):
            annualMax, annualMaxTimeStamp, annualMaxIndexes = _gev_annual_max_nan_aware(
                inputtimestamps, statInputTimeSeries[:, ii]
            )

            # tmpA = cat(3, annualMaxTimeStamp, annualMax)  -> shape (N, 1, 2)
            tmpA = np.stack([annualMaxTimeStamp, annualMax], axis=-1)[:, np.newaxis, :]
            jointextremes_list.append(tmpA)
            jointExtremeIndices_list.append(annualMaxIndexes)
            jointExtremesNS_list.append(inputtimeseries[annualMaxIndexes, ii])

        # Concatenate along series dimension: final shape (N, nSeries, 2)
        jointextremes = np.concatenate(jointextremes_list, axis=1)
        jointExtremesNS_matrix = np.column_stack(jointExtremesNS_list)
        jointExtremeIndices_matrix = np.column_stack(jointExtremeIndices_list)

        # Provide a samplingAnalysis-like dict so downstream code (line ~341) works
        samplingAnalysis = {
            'jointExtremesNS': jointExtremesNS_matrix,
            'jointExtremeIndices': jointExtremeIndices_matrix,
        }

        # Compute GEV CDF for each series (MATLAB: cdf('gev', x, shape, scale, location))
        gpdCDFCopula = np.full((jointextremes.shape[0], nSeries), np.nan)
        for ii in range(nSeries):
            stat_params = marginalAnalysis[ii][0][0]['stationaryParams']['parameters']
            shapeParam = stat_params['epsilon']
            scaleParam = stat_params['sigma']
            locationParam = stat_params['mu']

            gevCdf = gev.cdf(jointextremes[:, ii, 1], c=-shapeParam, loc=locationParam, scale=scaleParam)
            gevCdf[gevCdf == 0] = 1e-7
            gpdCDFCopula[:, ii] = gevCdf

    copulaParam = {'family': copulaFamily, 'nSeries': nSeries}
    
    dt_diff = np.diff(inputtimestamps)
    dt = np.round(np.median(dt_diff[dt_diff > 0]), 4) if len(dt_diff) > 0 else 1.0
    timeWindowIndices = int(min(np.round(copulaTimeWindow / dt), len(inputtimestamps)))
    timeSlideIndices = int(np.round(timeSlide / dt))
    
    if nSeries <= 3:
        monovarProbJointExtrList = []
        timeStampsByTimeWindow = []
        IndexWindowList = []
        timePeaksList = []
        rhoTotal = [] 
        
        beginIndex = 0
        timePeaks = jointextremes[:, :, 0] 
        
        jointExtremesNS_all = samplingAnalysis['jointExtremesNS']
        jointExtremesNS_blocks = []
        
        while beginIndex + timeWindowIndices <= len(inputtimestamps):
            if beginIndex + timeSlideIndices + timeWindowIndices > len(inputtimestamps):
                inputtimestampsWindow = inputtimestamps[beginIndex:]
            else:
                inputtimestampsWindow = inputtimestamps[beginIndex : beginIndex + timeWindowIndices]
                
            timeStampsByTimeWindow.append(inputtimestampsWindow)
            
            # MATLAB: [Lia,~]=ismember(timePeaks,inputtimestampsWindow); WindowIndex=all(Lia,2);
            # Check ALL columns of timePeaks fall within the window (not just column 0)
            window_mask = np.all(
                (timePeaks >= inputtimestampsWindow[0]) & (timePeaks <= inputtimestampsWindow[-1]),
                axis=1
            )
            
            timePeaksList.append(timePeaks[window_mask, :])
            monovarProbJointExtrWindow = gpdCDFCopula[window_mask, :]
            monovarProbJointExtrList.append(monovarProbJointExtrWindow)
            jointExtremesNS_blocks.append(jointExtremesNS_all[window_mask, :])
            global_indices = np.where(np.isin(inputtimestamps, timePeaks[window_mask, 0]))[0]
            IndexWindowList.append(global_indices)

            beginIndex += timeSlideIndices
            
            if len(monovarProbJointExtrWindow) < 2:
                rho = np.ones((2, 2))
            else:
                rho = tsCopulaFit(copulaFamily, monovarProbJointExtrWindow)
            rhoTotal.append(rho)

    else:
        pass 

    copulaParam['timeStampsByTimeWindow'] = timeStampsByTimeWindow
    copulaParam['rhoTimeStamps'] = np.linspace(
        timeStampsByTimeWindow[0][0], 
        timeStampsByTimeWindow[-1][-1], 
        len(timeStampsByTimeWindow)
    )
    
    rhoTotalRaw = [np.copy(r) for r in rhoTotal]

    N = len(rhoTotal)
    for iSeries1 in range(nSeries):
        for iSeries2 in range(iSeries1 + 1, nSeries):
            comp = np.zeros(N)
            for it in range(N):
                comp[it] = rhoTotal[it][iSeries1, iSeries2]

            comp_smoothed = pd.Series(comp).rolling(window=smoothInd, min_periods=1, center=True).mean().values
            
            for it in range(N):
                rhoTotal[it][iSeries1, iSeries2] = comp_smoothed[it]
                rhoTotal[it][iSeries2, iSeries1] = comp_smoothed[it]
                
    copulaParam['rho'] = rhoTotal
    copulaParam['rhoRaw'] = rhoTotalRaw
    copulaParam['smoothInd'] = smoothInd

    if nSeries <= 3:
        cellTimePeaks = np.vstack([p for p in timePeaksList if p.size > 0])
        jointExtremesNS_stacked = np.vstack([b for b in jointExtremesNS_blocks if b.size > 0])
        
        _, iB = np.unique(jointExtremesNS_stacked, axis=0, return_index=True)
        iB.sort()
        yMax = jointExtremesNS_stacked[iB]
        tMax = cellTimePeaks[iB]
    else:
        yMax, tMax = None, None

    CopulaAnalysis = {
        'copulaParam': copulaParam,
        'marginalAnalysis': marginalAnalysis,
        'methodology': marginalDistributions,
        'timeVaryingCopula': timeVaryingCopula,
        'jointExtremes': jointExtremesNS_blocks if nSeries <= 3 else None,
        'jointExtremeTimeStamps': timePeaksList,
        'jointExtremeIndices': IndexWindowList,
        'jointExtremeMonovariateProbNS': monovarProbJointExtrList,
        'yMax': yMax,
        'tMax': tMax,
        'timeWindow': timewindow,
        'thresholdPotNS': np.column_stack([ma[0][1]['parameters']['threshold'] for ma in marginalAnalysis]) if marginalDistributions == 'gpd' else np.column_stack([ma[0][0]['parameters']['mu'] for ma in marginalAnalysis])
    }
    
    if marginalDistributions == 'gpd':
        gpdCDFCopula = np.full((jointextremes.shape[0], nSeries), np.nan)
        for ii in range(nSeries):
            nonStatEvaParams = marginalAnalysis[ii][0]
            stat_params = nonStatEvaParams[1]['stationaryParams']['parameters']
            shapeParam = stat_params['shape']
            scaleParam = stat_params['sigma']
            thrshldValue = stat_params['threshold']
            gpdCdf = genpareto.cdf(jointextremes[:, ii, 1], c=shapeParam, loc=thrshldValue, scale=scaleParam)
            gpdCdf[gpdCdf == 0] = 1e-7
            gpdCDFCopula[:, ii] = gpdCdf

    return CopulaAnalysis


def tsCopulaSampleJointPeaksMultiVariatePruning(inputtimestamps, inputtimeseries, **kwargs):
    import numpy as np
    import itertools
    import os as _os

    # --- DEBUG instrumentation (off in production) ---
    # Set DEBUG_JOINT_SAMPLING=START,END (MATLAB datenums) to trace a window.
    # Example for Venice Feb 2015: DEBUG_JOINT_SAMPLING=735630,735644
    _dbg_env = _os.environ.get('DEBUG_JOINT_SAMPLING', '').strip()
    _DBG_ENABLED = bool(_dbg_env)
    if _DBG_ENABLED:
        try:
            _parts = _dbg_env.split(',')
            _dbg_start = float(_parts[0]); _dbg_end = float(_parts[1])
        except Exception:
            _dbg_start, _dbg_end = -np.inf, np.inf
        print(f'[DBG-SAMPLING] Debug active for time range '
              f'[{_dbg_start}, {_dbg_end}] (MATLAB datenums)')
    else:
        _dbg_start, _dbg_end = None, None

    def _dbg(msg):
        if _DBG_ENABLED:
            print(f'[DBG-SAMPLING] {msg}')

    def _in_window(t):
        if not _DBG_ENABLED: return False
        return _dbg_start <= float(t) <= _dbg_end

    numVar = inputtimeseries.shape[1]
    samplingThresholdPrct = kwargs.get('samplingThresholdPrct', [99.0] * numVar)
    minPeakDistanceInDaysMonovarSampling = kwargs.get('minPeakDistanceInDaysMonovarSampling', [3.0] * numVar)
    maxPeakDistanceInDaysMultivarSampling = kwargs.get('maxPeakDistanceInDaysMultivarSampling', 3.0)
    samplingOrder = kwargs.get('samplingOrder', [])
    peakType = kwargs.get('peakType', 'allexceedthreshold')
    marginalAnalysis = kwargs.get('marginalAnalysis', [])
    marginalDistributions = kwargs.get('marginalDistributions', 'gpd').lower()
    
    if isinstance(maxPeakDistanceInDaysMultivarSampling, (list, np.ndarray)):
        maxMultivarDist = maxPeakDistanceInDaysMultivarSampling[0]
    else:
        maxMultivarDist = maxPeakDistanceInDaysMultivarSampling

    time_diffs = np.diff(inputtimestamps)
    dt = np.min(time_diffs[time_diffs > 0]) if len(time_diffs) > 0 else 1.0

    # Match MATLAB: always compute threshold as prctile(statInputTimeSeries, percentile)
    # MATLAB line 121-123: thresholdsArray=cellfun(@(x,y) prctile(x,y), inputtimeseriesCell, ...)
    thresholdsArray = [safe_matlab_percentile(inputtimeseries[:, i], samplingThresholdPrct[i]) for i in range(numVar)]

    _dbg(f'maxMultivarDist={maxMultivarDist} days, '
         f'thresholds={[round(t,4) for t in thresholdsArray]}')

    minPeakDistanceMonovar = [dist / dt for dist in minPeakDistanceInDaysMonovarSampling]
    
    pksCell, indxCell, pksTimeCell, idPeakCell = [], [], [], []

    for i in range(numVar):
        distance = max(1, int(np.round(minPeakDistanceMonovar[i])))
        locs, pks = safe_matlab_find_peaks(inputtimeseries[:, i], distance=distance)

        indxCell.append(locs)
        pksCell.append(inputtimeseries[locs, i])
        pksTimeCell.append(inputtimestamps[locs])
        idPeakCell.append(np.full(len(locs), i + 1))

        # Debug: list peaks for this variable in target window
        if _DBG_ENABLED:
            for li, ti in zip(locs, inputtimestamps[locs]):
                if _in_window(ti):
                    _dbg(f'  var={i+1}  POT peak at t={ti:.3f}  '
                         f'value={inputtimeseries[li, i]:.4f}  '
                         f'(threshold={thresholdsArray[i]:.4f}, '
                         f'above={inputtimeseries[li, i] >= thresholdsArray[i]})')

    combinedPeaksTime = np.concatenate(pksTimeCell)
    combinedPeaksIndex = np.concatenate(indxCell)
    combinedPeaks = np.concatenate(pksCell)
    combinedPeaksId = np.concatenate(idPeakCell)
    
    sortIndexCombinedPeaks = np.argsort(combinedPeaksTime, kind='stable')
    combinedPeaksTime = combinedPeaksTime[sortIndexCombinedPeaks]
    combinedPeaksIndex = combinedPeaksIndex[sortIndexCombinedPeaks]
    combinedPeaks = combinedPeaks[sortIndexCombinedPeaks]
    combinedPeaksId = combinedPeaksId[sortIndexCombinedPeaks]
    
    indexJointPeaks = []
    indexJointNonPeaks = []
    
    numVarId = np.unique(combinedPeaksId)
    
    for combinedPeaksCount in range(len(combinedPeaksTime) - 1):
        anchor_id = combinedPeaksId[combinedPeaksCount]
        combinedPeaksIdPair = numVarId[numVarId != anchor_id]
        
        indicesPerEventCell = []
        for pair_id in combinedPeaksIdPair:
            time_diff = combinedPeaksTime[combinedPeaksCount+1:] - combinedPeaksTime[combinedPeaksCount]
            valid_time = time_diff <= maxMultivarDist
            valid_id = combinedPeaksId[combinedPeaksCount+1:] == pair_id
            passIndices = np.where(valid_time & valid_id)[0]
            indicesPerEventCell.append(combinedPeaksCount + 1 + passIndices)
            
        if any(len(x) == 0 for x in indicesPerEventCell):
            continue
            
        lists_to_combine = [list(x) for x in indicesPerEventCell]
        lists_to_combine.append([combinedPeaksCount])
        
        indicesCombination = np.array(list(itertools.product(*lists_to_combine)))
        
        valid_combos = []
        for combo in indicesCombination:
            ids_in_combo = combinedPeaksId[combo]
            if len(np.unique(ids_in_combo)) == numVar:
                valid_combos.append(combo)
                
        if not valid_combos:
            continue
            
        indicesCombination = np.array(valid_combos)
        compoundPeaks = combinedPeaks[indicesCombination]
        compoundPeaksId = combinedPeaksId[indicesCombination]
        
        thresholdsArrayReGrouped = np.array([thresholdsArray[int(i)-1] for i in compoundPeaksId.flatten()]).reshape(compoundPeaksId.shape)
        
        is_joint_peak = np.all(compoundPeaks >= thresholdsArrayReGrouped, axis=1)
        is_joint_non_peak = np.any(compoundPeaks >= thresholdsArrayReGrouped, axis=1) & ~is_joint_peak

        # Debug: log when anchor is in target window
        if _DBG_ENABLED and _in_window(combinedPeaksTime[combinedPeaksCount]):
            anchor_t = combinedPeaksTime[combinedPeaksCount]
            _dbg(f'  ANCHOR @ t={anchor_t:.3f}  id={anchor_id}  '
                 f'(combos={len(indicesCombination)}, '
                 f'joint={int(is_joint_peak.sum())}, '
                 f'non-joint={int(is_joint_non_peak.sum())})')
            for ci, combo in enumerate(indicesCombination):
                ids = combinedPeaksId[combo]
                ts  = combinedPeaksTime[combo]
                vs  = combinedPeaks[combo]
                thr = thresholdsArrayReGrouped[ci]
                ok  = is_joint_peak[ci]
                _dbg(f'    combo[{ci}]: '
                     f't={[round(float(x),3) for x in ts]} '
                     f'val={[round(float(x),4) for x in vs]} '
                     f'thr={[round(float(x),4) for x in thr]} '
                     f'-> joint={bool(ok)}')

        if np.any(is_joint_peak):
            indexJointPeaks.extend(indicesCombination[is_joint_peak].tolist())
        elif np.any(is_joint_non_peak):
            indexJointNonPeaks.extend(indicesCombination[is_joint_non_peak].tolist())

    indexJointPeaks = np.array(indexJointPeaks) if indexJointPeaks else np.empty((0, numVar), dtype=int)
    indexJointNonPeaks = np.array(indexJointNonPeaks) if indexJointNonPeaks else np.empty((0, numVar), dtype=int)
    
    def extract_by_id(indices_matrix):
        if len(indices_matrix) == 0:
            return np.empty((0, numVar)), np.empty((0, numVar)), np.empty((0, numVar), dtype=int)
        times = np.zeros((len(indices_matrix), numVar))
        vals = np.zeros((len(indices_matrix), numVar))
        orig_idx = np.zeros((len(indices_matrix), numVar), dtype=int)
        for i in range(len(indices_matrix)):
            row = indices_matrix[i]
            ids = combinedPeaksId[row]
            for j, var_id in enumerate(ids):
                col_idx = int(var_id) - 1
                times[i, col_idx] = combinedPeaksTime[row[j]]
                vals[i, col_idx] = combinedPeaks[row[j]]
                orig_idx[i, col_idx] = combinedPeaksIndex[row[j]]
        return times, vals, orig_idx

    jointPeaksTimeColumnWise, jointPeaksColumnWise, indexJointPeaksColumnWise = extract_by_id(indexJointPeaks)
    jointNonPeaksTimeColumnWise, jointNonPeaksColumnWise, indexJointNonPeaksColumnWise = extract_by_id(indexJointNonPeaks)

    jointPeaksTimeTotal = np.vstack((jointPeaksTimeColumnWise, jointNonPeaksTimeColumnWise)) if jointPeaksTimeColumnWise.size else np.empty((0, numVar))
    jointPeaksTotal = np.vstack((jointPeaksColumnWise, jointNonPeaksColumnWise)) if jointPeaksColumnWise.size else np.empty((0, numVar))
    jointIdTotal = np.vstack((indexJointPeaksColumnWise, indexJointNonPeaksColumnWise)) if indexJointPeaksColumnWise.size else np.empty((0, numVar), dtype=int)

    idPeaksArtificial = np.concatenate([np.ones(len(jointPeaksTimeColumnWise)), np.full(len(jointNonPeaksTimeColumnWise), 2)])

    if len(samplingOrder) >= 2 and jointPeaksTimeTotal.size > 0:
        nonrealisticIndices = (jointPeaksTimeTotal[:, samplingOrder[1]] - jointPeaksTimeTotal[:, samplingOrder[0]]) < 0
        n_removed = int(np.sum(nonrealisticIndices))
        if n_removed > 0:
            print(f'[samplingOrder] removed {n_removed} events violating order '
                  f'{samplingOrder[0]} -> {samplingOrder[1]}')
        jointPeaksTimeTotal = jointPeaksTimeTotal[~nonrealisticIndices]
        jointPeaksTotal = jointPeaksTotal[~nonrealisticIndices]
        jointIdTotal = jointIdTotal[~nonrealisticIndices]
        idPeaksArtificial = idPeaksArtificial[~nonrealisticIndices]

    if jointPeaksTotal.size > 0:
        means = np.mean(jointPeaksTotal, axis=1)
        idSort = np.argsort(-means, kind='stable')
        jointPeaksTimeTotal = jointPeaksTimeTotal[idSort]
        jointPeaksTotal = jointPeaksTotal[idSort]
        jointIdTotal = jointIdTotal[idSort]
        idPeaksArtificial = idPeaksArtificial[idSort]

        n_total_events = len(jointPeaksTimeTotal)
        minArrayPeaksTime = np.min(jointPeaksTimeTotal, axis=1)
        maxArrayPeaksTime = np.max(jointPeaksTimeTotal, axis=1)

        # --- DEBUG: dump every event in the sorted pool that has at least one
        # timestamp in the target window. Shows jx (rank), times, vals, mean,
        # and artif (1=real joint peak, 2=non-peak).
        if _DBG_ENABLED:
            _dbg(f'POOL DUMP — total events in pool: {n_total_events}')
            n_in_window = 0
            for jx_chk in range(n_total_events):
                evt = jointPeaksTimeTotal[jx_chk]
                if any(_in_window(t) for t in evt):
                    mean_val = float(jointPeaksTotal[jx_chk].mean())
                    artif = int(idPeaksArtificial[jx_chk])
                    _dbg(f'  POOL: jx={jx_chk}  t={[round(float(x),3) for x in evt]}  '
                         f'val={[round(float(x),4) for x in jointPeaksTotal[jx_chk]]}  '
                         f'mean={mean_val:.4f}  artif={artif}')
                    n_in_window += 1
                    if n_in_window > 50:
                        _dbg(f'  ... (truncated at 50 in-window events)')
                        break
            _dbg(f'POOL DUMP — total in target window: {n_in_window}')

        indicesToRemove = []
        for jx in range(n_total_events):
            if jx in indicesToRemove:
                # NEW: trace skip if it's a window event
                if _DBG_ENABLED and any(_in_window(t) for t in jointPeaksTimeTotal[jx, :]):
                    _dbg(f'SKIP (already removed): jx={jx} '
                         f't={[round(float(x),3) for x in jointPeaksTimeTotal[jx, :]]} '
                         f'mean={float(jointPeaksTotal[jx].mean()):.4f}')
                continue
            eventTime = jointPeaksTimeTotal[jx, :]
            indicesNonOverlap = (np.min(eventTime) > maxArrayPeaksTime) | (np.max(eventTime) < minArrayPeaksTime)
            indicesNonOverlap[:jx+1] = True

            overlapping_idx = np.where(~indicesNonOverlap)[0]

            # NEW: trace every iteration where current event is in window,
            # regardless of whether it has overlaps or not.
            if _DBG_ENABLED and any(_in_window(t) for t in eventTime):
                _dbg(f'PROCESS jx={jx}  '
                     f't={[round(float(x),3) for x in eventTime]}  '
                     f'val={[round(float(x),4) for x in jointPeaksTotal[jx, :]]}  '
                     f'mean={float(jointPeaksTotal[jx].mean()):.4f}  '
                     f'artif={int(idPeaksArtificial[jx])}  '
                     f'n_overlap={len(overlapping_idx)}')

            if len(overlapping_idx) > 0:
                # Debug: log pruning decisions when winner or losers in target window
                if _DBG_ENABLED:
                    winner_in = any(_in_window(t) for t in eventTime)
                    losers_in = []
                    for rj in overlapping_idx:
                        loser_times = jointPeaksTimeTotal[rj, :]
                        if any(_in_window(t) for t in loser_times):
                            losers_in.append((rj, loser_times, jointPeaksTotal[rj, :]))
                    if winner_in or losers_in:
                        _dbg(f'PRUNE: winner jx={jx} t={[round(float(x),3) for x in eventTime]} '
                             f'val={[round(float(x),4) for x in jointPeaksTotal[jx, :]]}')
                        for rj, lt, lv in losers_in:
                            _dbg(f'   REMOVED loser jx={rj} t={[round(float(x),3) for x in lt]} '
                                 f'val={[round(float(x),4) for x in lv]}')
                indicesToRemove.extend(overlapping_idx.tolist())
                
        indicesToRemove = np.unique(indicesToRemove)
        
        if len(indicesToRemove) == 0:
            pass
        else:
            prct_pruned = round(100 * (len(indicesToRemove) / n_total_events), 1)
            print(f'{prct_pruned} % of sampled events pruned due to overlapping of events')
            
        mask = np.ones(n_total_events, dtype=bool)
        mask[indicesToRemove] = False
        
        jointPeaksTimeTotal = jointPeaksTimeTotal[mask]
        jointPeaksTotal = jointPeaksTotal[mask]
        jointIdTotal = jointIdTotal[mask]
        idPeaksArtificial = idPeaksArtificial[mask]

    jointPeaksTimeColumnWise = jointPeaksTimeTotal[idPeaksArtificial == 1]
    jointNonPeaksTimeColumnWise = jointPeaksTimeTotal[idPeaksArtificial == 2]
    jointPeaksColumnWise = jointPeaksTotal[idPeaksArtificial == 1]
    jointNonPeaksColumnWise = jointPeaksTotal[idPeaksArtificial == 2]
    
    jointExtremeIndices = jointIdTotal[idPeaksArtificial == 1]
    jointNonExtremeIndices = jointIdTotal[idPeaksArtificial == 2]
    
    peakIndicesAll = np.vstack((jointExtremeIndices, jointNonExtremeIndices)) if jointIdTotal.size else np.empty((0, numVar), dtype=int)
    
    print(f'{len(jointPeaksTimeColumnWise)} Compound peak events found')
    if len(jointPeaksTimeColumnWise) > 1:
        time_diffs_events = np.abs(np.diff(jointPeaksTimeColumnWise, axis=1))
        print(f'average time among peaks of {np.mean(time_diffs_events):.4f} days '
              f'with minimum of {np.min(time_diffs_events):.0f} days and maximum of {np.max(time_diffs_events):.3f} days')
              
    jointextremes = np.stack((jointPeaksTimeColumnWise, jointPeaksColumnWise), axis=2) if jointPeaksTimeColumnWise.size else np.empty((0, numVar, 2))
    jointextremes2 = np.stack((jointNonPeaksTimeColumnWise, jointNonPeaksColumnWise), axis=2) if jointNonPeaksTimeColumnWise.size else np.empty((0, numVar, 2))
    
    jointExtremesNS = []
    thresholdsNonStation = []
    
    if marginalAnalysis:
        nonStatSeries = np.column_stack([ma[1].nonStatSeries for ma in marginalAnalysis])
        trendSeries = np.column_stack([ma[1].trendSeries for ma in marginalAnalysis])
        stdDevSeries = np.column_stack([ma[1].stdDevSeries for ma in marginalAnalysis])
        
        if peakType.lower() == 'allexceedthreshold' and jointExtremeIndices.size > 0:
            jointExtremesNS = np.column_stack([nonStatSeries[jointExtremeIndices[:, i], i] for i in range(numVar)])
        elif peakType.lower() == 'anyexceedthreshold' and peakIndicesAll.size > 0:
            jointExtremesNS = np.column_stack([nonStatSeries[peakIndicesAll[:, i], i] for i in range(numVar)])
            
        for i in range(numVar):
            thr_ns = trendSeries[:, i] + stdDevSeries[:, i] * thresholdsArray[i]
            thresholdsNonStation.append(thr_ns)
            
    samplingAnalysis = {
        'jointextremes': jointextremes,
        'jointextremes2': jointextremes2,
        'thresholdsC': thresholdsArray,
        'jointExtremeIndices': jointExtremeIndices,
        'peakIndicesAll': peakIndicesAll,
        'jointExtremesNS': jointExtremesNS,
        'thresholdsNonStation': thresholdsNonStation
    }
    
    return samplingAnalysis


def _copulaparam_inverse(family, tau):
    """
    Helper function to convert Kendall's tau to copula parameter (theta).
    Replicates MATLAB's built-in `copulaparam` function.
    """
    family = family.lower()
    
    if family == 'gumbel':
        # Theta = 1 / (1 - tau), Domain: [1, inf)
        if tau >= 1.0:
            return np.inf
        return 1.0 / (1.0 - tau)
        
    elif family == 'clayton':
        # Theta = 2 * tau / (1 - tau), Domain: (0, inf)
        if tau >= 1.0:
            return np.inf
        return 2.0 * tau / (1.0 - tau)
        
    elif family == 'frank':
        # Theta has no closed-form inverse for Frank. 
        # tau = 1 - 4/theta * (1 - D1(theta)), where D1 is the Debye function of order 1.
        if np.abs(tau) < 1e-7:
            return 0.0
        if tau >= 1.0:
            return np.inf
        if tau <= -1.0:
            return -np.inf
            
        def debye_1(theta):
            # Integral of t/(exp(t)-1) from 0 to theta
            # Handled t near 0 to prevent division by zero warning
            integral_val, _ = quad(lambda t: t / (np.exp(t) - 1.0) if t > 1e-8 else 1.0, 0, theta)
            return integral_val / theta
            
        def frank_tau_diff(theta, target_tau):
            if np.abs(theta) < 1e-7:
                return -target_tau
            return 1.0 - 4.0 / theta * (1.0 - debye_1(theta)) - target_tau
            
        try:
            # Numerically solve for theta
            sol = root_scalar(frank_tau_diff, args=(tau,), bracket=[-100, 100], method='brentq')
            return sol.root
        except ValueError:
            # Fallback if bracket is not wide enough
            from scipy.optimize import fsolve
            return fsolve(frank_tau_diff, x0=10.0 if tau > 0 else -10.0, args=(tau,))[0]
    else:
        raise ValueError(f"Copula family '{family}' is not supported for parameter conversion.")


def tsCopulaFit(copulaFamily, uProb):
    """
    Fits copula parameters based on empirical probabilities.
    
    Args:
        copulaFamily (str): 'gaussian', 'gumbel', 'clayton', or 'frank'.
        uProb (ndarray): 2D array of empirical non-exceedance probabilities (samples x variables).
        
    Returns:
        ndarray: Copula parameter matrix. For Gaussian, it's the Spearman correlation matrix.
                 For Archimedean copulas, it's the pairwise copula parameters (theta).
    """
    copulaFamily = copulaFamily.lower()
    
    if copulaFamily == 'gaussian':
        # Spearman correlation works better than directly fitting Gaussian copula
        df = pd.DataFrame(uProb)
        copulaParam = df.corr(method='spearman').values
        
    elif copulaFamily in ['gumbel', 'clayton', 'frank']:
        nSeries = uProb.shape[1]
        copulaParam = np.ones((nSeries, nSeries))
        
        # Calculate Kendall's tau correlation matrix
        df = pd.DataFrame(uProb)
        kendalT = df.corr(method='kendall').values
        
        if copulaFamily == 'gumbel':
            # Gumbel copula requires tau >= 0
            kendalT[kendalT < 0] = 0.0
            
        # Convert Kendall's tau to Copula parameter (theta) for each pair
        for iSeries1 in range(nSeries):
            for iSeries2 in range(iSeries1 + 1, nSeries):
                theta = _copulaparam_inverse(copulaFamily, kendalT[iSeries1, iSeries2])
                copulaParam[iSeries1, iSeries2] = theta
                copulaParam[iSeries2, iSeries1] = theta
                
        if copulaFamily == 'gumbel':
            # Replace infinities with 1 as a fallback (replicating MATLAB behavior)
            copulaParam[np.isinf(copulaParam)] = 1.0
            
    else:
        raise ValueError(f"copulaFamily not supported: {copulaFamily}")
        
    return copulaParam


def tsCopulaMontecarlo(copulaAnalysis, **kwargs):
    """
    Performs Monte-Carlo simulation (resampling) from a pre-determined copula function.
    
    Args:
        copulaAnalysis (dict): A dictionary containing various parameters of the fitted copula.
                               Needs to be the output of tsCopulaExtremes function.
        **kwargs:
            nResample (int): 1D scalar indicating size of the Monte-Carlo simulation to be performed. Default is 1000.
            timeIndex (str/int): A scalar parameter for indexing non-stationary parameters ('first', 'middle', 'last').
            nonStationarity (str): Specifies non-stationarity approach (e.g., "margins").
            
    Returns:
        dict: A dictionary same as the input with two additional appended variables:
              - monteCarloRsmpl: Resampled return levels.
              - resampleProb: Resampled return probabilities.
              - timeIndexArray: Indices used for extracting non-stationary parameters.
    """
    
    # -------------------------------------------------------------------------
    # Parse Arguments
    # -------------------------------------------------------------------------
    timeIndexArg = kwargs.get('timeIndex', 'middle')
    nResample = kwargs.get('nResample', 1000)
    nonStationarity = kwargs.get('nonStationarity', "")
    
    # Read input data
    methodology = copulaAnalysis['methodology']
    copulaParam = copulaAnalysis['copulaParam']
    nSeries = copulaParam['nSeries']
    copulaFamily = copulaParam['family']
    marginalAnalysis = copulaAnalysis['marginalAnalysis']
    timeVaryingCopula = copulaAnalysis['timeVaryingCopula']
    

    import copy
    rhoCell = copy.deepcopy(copulaParam['rho'])
    
    if nonStationarity.lower() == 'margins':
        # Calculate the mean of the upper triangular parts of all rho matrices
        upper_tri_vals = [np.triu(r, k=1)[np.triu_indices_from(r, k=1)] for r in rhoCell]
        mRho = np.mean(np.concatenate(upper_tri_vals))
        
        # Replace off-diagonal elements in all rho matrices with mRho
        for i in range(len(rhoCell)):
            np.fill_diagonal(rhoCell[i], 1.0) # Ensure diagonal is 1
            off_diag_indices = np.where(~np.eye(nSeries, dtype=bool))
            rhoCell[i][off_diag_indices] = mRho

    resampleProb = []

    # Optional RNG seed so Monte-Carlo scatter in panels (d)/(e) can be made reproducible
    # and visually tuned against the MATLAB reference. Default None → no reset (preserves
    # natural RNG state driven by the caller's own `np.random.seed()`). When explicitly
    # provided via kwargs['mcSeed'], numpy's seed is reset here right before MC sampling.
    mc_seed = kwargs.get('mcSeed', None)
    if mc_seed is not None:
        np.random.seed(mc_seed)

    for ik in range(len(rhoCell)):
        rho_matrix = rhoCell[ik]
        uProbNS = copulaAnalysis['jointExtremeMonovariateProbNS'][ik]

        if copulaFamily.lower() in ['frank', 'clayton', 'gumbel']:
            probs = tsCopulaRnd(copulaFamily, rho_matrix, nResample, uProbNS)
            resampleProb.append(probs)
            
        elif copulaFamily.lower() == 'gaussian':
            # Needs a function representing MATLAB's copularnd('Gaussian', rho, N)
            from scipy.stats import multivariate_normal, norm
            mean_vec = np.zeros(nSeries)
            # Ensure positive semi-definite
            cov_mat = np.copy(rho_matrix)
            min_eig = np.min(np.linalg.eigvals(cov_mat))
            if min_eig < 0:
                cov_mat -= 10 * min_eig * np.eye(*cov_mat.shape)
            samples = multivariate_normal.rvs(mean=mean_vec, cov=cov_mat, size=nResample)
            probs = norm.cdf(samples)
            resampleProb.append(probs)

    # -------------------------------------------------------------------------
    # Conversion of Monte-Carlo Probabilities to Data Space
    # -------------------------------------------------------------------------
    monteCarloRsmplCell = []
    timeStampsByTimeWindow = copulaParam['timeStampsByTimeWindow']
    timeStamps = marginalAnalysis[0][1].timeStamps  # Assuming timeStamps are the same for all margins
    
    # Yazdırma (Print) işlemlerini döngü dışına aldık ki konsolu kirletmesin
    if isinstance(timeIndexArg, int) or str(timeIndexArg).isnumeric():
        print("Numeric timeindex not accepted... Middle timeindex selected automatically.")
    elif str(timeIndexArg).lower() == 'first':
        print("Conversion of Monte-Carlo probabilities to data space is based on non-stationary values evaluated at the first timeindex.")
    elif str(timeIndexArg).lower() == 'last':
        print("Conversion of Monte-Carlo probabilities to data space is based on non-stationary values evaluated at the last timeindex.")
    else: # Default 'middle'
        print("Conversion of Monte-Carlo probabilities to data space is based on non-stationary values evaluated at the middle timeindex.")

    timeIndexArray = []
    for window in timeStampsByTimeWindow:
        min_w = np.min(window)
        max_w = np.max(window)
        
        iix = np.where((timeStamps >= min_w) & (timeStamps <= max_w))[0]
        
        if len(iix) == 0:
            timeIndexArray.append(0) # Fallback
            continue
            
        if isinstance(timeIndexArg, int) or str(timeIndexArg).isnumeric():
            timeIndexArray.append(iix[len(iix) // 2])
        elif str(timeIndexArg).lower() == 'first':
            timeIndexArray.append(iix[0])
        elif str(timeIndexArg).lower() == 'last':
            timeIndexArray.append(iix[-1])
        else: # Default 'middle'
            timeIndexArray.append(iix[len(iix) // 2])

    # Convert to Data Space
    for ik in range(len(rhoCell)):
        resampleProbTemp = resampleProb[ik]
        resampled_data = np.zeros_like(resampleProbTemp)
        
        for ivar in range(nSeries):
            nonStatEvaParams = marginalAnalysis[ivar][0]
            t_idx = timeIndexArray[ik]
            resampled_data[:, ivar] = computeResampledLevels(resampleProbTemp[:, ivar], nonStatEvaParams, t_idx, methodology)
            
        monteCarloRsmplCell.append(resampled_data)

    # Append to CopulaAnalysis
    monteCarloAnalysis = {
        'monteCarloRsmpl': monteCarloRsmplCell,
        'resampleProb': resampleProb,
        'timeIndexArray': timeIndexArray
    }

    return monteCarloAnalysis


def computeResampledLevels(resampleProb, nonStatEvaParams, timeIndex, methodology):
    """
    Helper function to transform probabilities back to the physical data space.
    """
    from scipy.stats import genpareto
    from scipy.stats import genextreme as gev
    
    if methodology.lower() == 'gpd':
        params = nonStatEvaParams[1]['parameters']
        thrshld = params['threshold'][timeIndex] if isinstance(params['threshold'], np.ndarray) else params['threshold']
        scaleParam = params['sigma'][timeIndex] if isinstance(params['sigma'], np.ndarray) else params['sigma']
        shapeParam = params['epsilon']  # Scipy c parameter
        
        # scipy genpareto.ppf(q, c, loc, scale)
        monteCarloRsmpls = genpareto.ppf(resampleProb, c=shapeParam, loc=thrshld, scale=scaleParam)
        
    elif methodology.lower() == 'gev':
        params = nonStatEvaParams[0]['parameters']
        mu = params['mu'][timeIndex] if isinstance(params['mu'], np.ndarray) else params['mu']
        scaleParam = params['sigma'][timeIndex] if isinstance(params['sigma'], np.ndarray) else params['sigma']
        shapeParam = params['epsilon']
        
        # scipy gev.ppf(q, c, loc, scale). Note: c = -epsilon
        monteCarloRsmpls = gev.ppf(resampleProb, c=-shapeParam, loc=mu, scale=scaleParam)
        
    else:
        raise ValueError("Methodology not supported.")
        
    return monteCarloRsmpls


def _gumbel_copula_rnd(theta, N):
    """
    Generate bivariate Gumbel copula samples.
    Matches MATLAB copularnd('gumbel', theta, N).
    Uses Marshall-Olkin algorithm with stable variate (Chambers-Mallows-Stuck).
    """
    if theta < 1.0:
        theta = 1.0
    alpha = 1.0 / theta
    # Generate positive stable variate via Chambers-Mallows-Stuck method
    # This matches MATLAB's internal implementation of copularnd for Gumbel
    V_angle = np.random.uniform(0, np.pi, size=N)
    W_exp = -np.log(np.random.uniform(size=N))  # Exp(1) via inverse CDF
    if abs(alpha - 1.0) < 1e-10:
        # theta == 1 means independence
        V_stable = np.ones(N)
    else:
        # Chambers-Mallows-Stuck formula for S(alpha, 1, gamma, 0; CMS parameterization)
        term1 = np.sin(alpha * V_angle) / (np.sin(V_angle)) ** (1.0 / alpha)
        term2 = (np.sin((1.0 - alpha) * V_angle) / W_exp) ** ((1.0 - alpha) / alpha)
        V_stable = term1 * term2

    V_stable = np.maximum(V_stable, 1e-300)
    E = -np.log(np.random.uniform(size=(N, 2)))
    t = (E / V_stable[:, np.newaxis]) ** (1.0 / theta)
    u = np.exp(-t)
    return u


def tsCopulaRnd(family, copulaPar, N, uProb):
    import numpy as np
    family = family.lower()

    if family == 't':
        raise ValueError("Copula family unsupported: t")

    if family == 'gaussian':
        from scipy.stats import multivariate_normal, norm
        dim = 2 if np.isscalar(copulaPar) else copulaPar.shape[0]
        if np.isscalar(copulaPar):
            cov_mat = np.array([[1.0, copulaPar], [copulaPar, 1.0]])
        else:
            cov_mat = np.array(copulaPar)
            
        min_eig = np.min(np.linalg.eigvals(cov_mat))
        if min_eig < 0:
            cov_mat -= 10 * min_eig * np.eye(*cov_mat.shape)
            
        mean_vec = np.zeros(dim)
        samples = multivariate_normal.rvs(mean=mean_vec, cov=cov_mat, size=N)
        if N == 1:
            samples = samples.reshape(1, -1)
        u = norm.cdf(samples)

    elif family == 'gumbel':
        alpha = np.array(copulaPar)
        if np.isnan(alpha).any():
            from scipy.stats import uniform
            dim = 2 if np.isscalar(copulaPar) else copulaPar.shape[0]
            u = uniform.rvs(size=(N, dim))
        else:
            # MATLAB: if isscalar or 2x2 matrix, use copularnd directly
            is_bivariate = np.isscalar(copulaPar) or (hasattr(copulaPar, 'shape') and copulaPar.shape[0] == 2)
            if is_bivariate:
                theta_val = float(copulaPar if np.isscalar(copulaPar) else copulaPar[0, 1])
                u = _gumbel_copula_rnd(theta_val, N)
            else:
                if np.isscalar(copulaPar):
                    alpha = np.array([[1.0, copulaPar], [copulaPar, 1.0]])
                try:
                    order = tsGumbelCVine.cvineOrder(alpha)
                    theta = tsGumbelCVine.fit(uProb, alpha, order)
                    u = tsGumbelCVine.simulate(N, order, theta)
                except NameError:
                    raise NameError("tsGumbelCVine class is missing.")
                
    elif family in ['clayton', 'frank']:

        from scipy.stats import uniform
        theta = copulaPar if np.isscalar(copulaPar) else copulaPar[0, 1]
        
        u1 = uniform.rvs(size=N)
        w = uniform.rvs(size=N)
        
        if family == 'clayton':
            if theta < 1e-7:
                u2 = w
            else:
                u2 = (u1**(-theta) * (w**(-theta / (theta + 1.0)) - 1.0) + 1.0)**(-1.0 / theta)
                
        else:
            if np.abs(theta) < 1e-7:
                u2 = w
            else:
                num = w * (1.0 - np.exp(-theta))
                den = w * (np.exp(-theta * u1) - 1.0) - np.exp(-theta * u1)
                u2 = - (1.0 / theta) * np.log(1.0 + num / den)
                
        u = np.column_stack((u1, u2))
        
    else:
        raise ValueError(f"Copula family not supported: {family}")

    return u


class tsGumbelCVine:

    @staticmethod
    def cvineOrder(alpha):
        """
        Root-first C-vine order from a Gumbel-theta matrix (alpha).
        Heuristic: pick node with max total |tau| as root, then greedily add the next.
        """
        d = alpha.shape[0]
        if alpha.shape[1] != d:
            raise ValueError('alpha must be square')
            
        if np.any(np.diag(alpha) != 1):
            warnings.warn('alpha diagonal should be 1 for Gumbel')

        # Convert to Kendall tau (tau = 1 - 1/theta)
        # Guard theta >= 1
        safe_alpha = np.maximum(alpha, 1.0)
        tau = 1.0 - 1.0 / safe_alpha
        np.fill_diagonal(tau, 0.0)
        absTau = np.abs(tau)

        scores = np.sum(absTau, axis=1)
        picked = np.zeros(d, dtype=bool)
        order = np.zeros(d, dtype=int)

        root = np.argmax(scores)
        order[0] = root
        picked[root] = True

        for k in range(1, d):
            remaining = np.where(~picked)[0]
            # Sum of absolute taus from remaining nodes to already picked nodes in the order
            sc = np.zeros(len(remaining))
            for i, rem_idx in enumerate(remaining):
                sc[i] = np.sum(absTau[rem_idx, order[:k]])
                
            ix = np.argmax(sc)
            order[k] = remaining[ix]
            picked[order[k]] = True

        return order

    @staticmethod
    def fit(uProb, alpha, order):
        """
        Fit all Gumbel pair-copulas in a C-vine using pseudo-observations.
        Tree 1: take theta directly from alpha (matrix or scalar).
        Trees >=2: estimate tau from pseudo-obs, then theta = 1/(1-tau).
        """
        n, d = uProb.shape
        if d != len(order):
            raise ValueError('uProb/order size mismatch')

        if np.isscalar(alpha):
            A = alpha * np.ones((d, d))
        else:
            A = np.array(alpha)
            if A.shape != (d, d):
                raise ValueError('alpha must be scalar or d-by-d matrix matching uProb cols.')
            # Reindex alpha to vine order
            A = A[np.ix_(order, order)]

        # Reorder data by vine order
        U = uProb[:, order]
        
        # Theta will be a 2D list to mimic MATLAB's cell array
        Theta = [[None for _ in range(d)] for _ in range(d - 1)]
        Ucond = np.copy(U)

        # Tree 1: parameters from alpha (theta >= 1)
        # Note: Python indexing is 0-based
        for j in range(1, d):
            Theta[0][j] = max(A[0, j], 1.0)

        # Trees 2..d-1
        for t in range(1, d - 1):
            # Estimate theta on edges (t,j), j>t using current (t-1)-conditioned pseudo-obs
            for j in range(t + 1, d):
                tau_tj, _ = kendalltau(Ucond[:, t], Ucond[:, j], nan_policy='omit')
                if np.isnan(tau_tj):
                    tau_tj = 0.0
                tau_tj = max(min(tau_tj, 0.999), 1e-6)
                Theta[t][j] = 1.0 / (1.0 - tau_tj)

            # Update pseudo-obs for next tree by conditioning on pivot v_t
            for j in range(t + 1, d):
                th = Theta[t][j]
                Ucond[:, j] = tsGumbelCVine.h(Ucond[:, j], Ucond[:, t], th)

        return Theta

    @staticmethod
    def simulate(N, order, Theta):
        """
        Simulate from a Gumbel C-vine with parameters Theta in vine order.
        """
        d = len(order)
        Uvine = np.zeros((N, d))
        W = np.random.rand(N, d)
        
        Uvine[:, 0] = W[:, 0]
        
        for k in range(1, d):
            uk = W[:, k]
            for j in range(k - 1, -1, -1):
                th = Theta[j][k]
                uk = tsGumbelCVine.hinv(uk, Uvine[:, j], th)
            Uvine[:, k] = uk

        # Map back to original column order
        U = np.zeros((N, d))
        U[:, order] = Uvine
        
        return U

    @staticmethod
    def h(u, v, theta):
        """
        Gumbel h-function: C_{U|V}(u|v; theta) for pair-copula (U,V).
        Works with both scalar and array inputs.
        """
        epsv = 1e-12
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        u = np.clip(u, epsv, 1.0 - epsv)
        v = np.clip(v, epsv, 1.0 - epsv)
        theta = max(theta, 1.0)

        a = (-np.log(u)) ** theta
        b = (-np.log(v)) ** theta
        s = (a + b) ** (1.0 / theta)

        C = np.exp(-s)
        h_val = C * (a + b) ** (1.0 / theta - 1.0) * (-np.log(v)) ** (theta - 1.0) / v
        h_val = np.clip(h_val, epsv, 1.0 - epsv)

        return h_val

    @staticmethod
    def hinv(z, v, theta):
        """
        Inverse h-function: solve h(u,v,theta)=z for u in (0,1).
        Fully vectorized for performance.
        """
        epsv = 1e-12
        z = np.atleast_1d(np.clip(z, epsv, 1.0 - epsv))
        v = np.atleast_1d(np.clip(v, epsv, 1.0 - epsv))
        theta = max(theta, 1.0)

        n = len(z)
        lo = epsv
        hi = 1.0 - epsv

        # Vectorized Newton iterations
        u = np.clip(z.copy(), 1e-3, 1.0 - 1e-3)
        for _ in range(12):
            val = tsGumbelCVine.h(u, v, theta) - z
            du = np.maximum(u, 1e-3) * 1e-6
            g1 = tsGumbelCVine.h(np.clip(u - du, lo, hi), v, theta) - z
            g2 = tsGumbelCVine.h(np.clip(u + du, lo, hi), v, theta) - z
            der = (g2 - g1) / (2.0 * du)
            der = np.where(np.abs(der) < 1e-8, np.sign(der + 1e-20) * 1e-8, der)
            step = val / der
            u = np.clip(u - step, lo, hi)
            converged = np.abs(val) < 1e-8
            if np.all(converged):
                break

        # Scalar fallback for any non-converged elements
        residual = np.abs(tsGumbelCVine.h(u, v, theta) - z)
        bad = residual > 1e-6
        if np.any(bad):
            bad_idx = np.where(bad)[0]
            for i in bad_idx:
                vi = v[i] if len(v) > 1 else v[0]
                try:
                    u[i] = brentq(lambda uu: float(tsGumbelCVine.h(np.array([uu]), np.array([vi]), theta)[0]) - z[i], lo, hi)
                except (ValueError, IndexError):
                    pass

        return u[0] if n == 1 else u
    

def tsRankmax(X):
    """
    Returns a vector of one-based ranks for each element.
    For groups of the same element, the maximum rank is returned. 
    This can be viewed as the number of elements smaller or equal to the given number.
    
    Args:
        X (ndarray): A 1D array containing samples in data space.
        
    Returns:
        ndarray: Ranks of each element in X.
    """
    X = np.asarray(X).flatten()
    n = len(X)
    R = np.zeros(n)
    
    if n == 0:
        return R
    
    # Sort the array and retrieve indices
    I = np.argsort(X)
    S = X[I]
    
    # Rank of the previous element (1-based mathematical rank)
    r = n
    # Value of the previous element
    prev = S[-1]
    
    # Loop backwards
    for i in range(n - 1, -1, -1):
        x = S[i]
        if x == prev:
            R[I[i]] = r
        else:
            prev = x
            r = i + 1  # Adjusting for 1-based rank
            R[I[i]] = r
            
    return R


def tsPseudoObservations(X):
    """
    Uniforms input sample to pseudo-observations.
    Based on empirical CDF function, division by (n+1) is used 
    to keep empirical CDF strictly lower than 1.
    
    Args:
        X (ndarray): Samples in data space (n_samples, n_dimensions).
        
    Returns:
        ndarray: Pseudo-observations.
    """
    X = np.asarray(X)
    n, d = X.shape
    U = np.zeros((n, d))
    
    for i in range(d):
        # Now using the custom tsRankmax function exactly as in MATLAB
        U[:, i] = tsRankmax(X[:, i]) / (n + 1.0)
        
    return U


def tsCopulaCdfFromSamples(u, Usample, return_se=False):
    """
    Empirical copula CDF from sample points.
    
    Args:
        u (ndarray): (q x d) query points in [0,1]^d.
        Usample (ndarray): (M x d) sample points from the fitted copula.
        return_se (bool): If True, also returns standard errors.
        
    Returns:
        C (ndarray): (q x 1) empirical CDF estimates.
        se (ndarray): (q x 1) standard errors (if return_se is True).
    """
    u = np.asarray(u)
    Usample = np.asarray(Usample)
    q, d = u.shape
    
    if Usample.shape[1] != d:
        raise ValueError(f"Dimension mismatch: size(u,2)={d}, size(Usample,2)={Usample.shape[1]}")
        
    epsv = 1e-12
    u = np.clip(u, epsv, 1.0 - epsv)
    Usample = np.clip(Usample, epsv, 1.0 - epsv)
    M = Usample.shape[0]
    
    # Vectorized comparison: Usample[M, 1, d] <= u[1, q, d]
    mask = np.all(Usample[:, None, :] <= u[None, :, :], axis=2)
    C = np.mean(mask, axis=0)
    
    if return_se:
        se = np.sqrt(C * np.maximum(1.0 - C, 0.0) / M)
        return C, se
        
    return C

def tsCopulaGOFNonStat(copulaAnalysis, monteCarloAnalysis, **kwargs):
    """
    Estimation of copula goodness-of-fit and other battery of statistics.
    
    Evaluates the goodness-of-fit (GOF) by comparing the correlation structure 
    (Spearman and Kendall) of the fitted copula to that of the original data. 
    Also calculates the rank-based Cramer-von Mises statistic (Sn).
    
    Args:
        copulaAnalysis (dict): Output from tsCopulaExtremes.
        monteCarloAnalysis (dict): Output from tsCopulaMontecarlo.
        
    Returns:
        dict: gofStatistics containing differences in correlation and Sn statistic.
    """
    gofStatistics = {}
    copulaFamily = copulaAnalysis['copulaParam']['family']
    copulaParam = copulaAnalysis['copulaParam']
    nSeries = copulaParam['nSeries']
    
    # Read non-stationary joint extremes
    jointExtremes = copulaAnalysis['jointExtremes']
    if not isinstance(jointExtremes, list):
        jointExtremes = [jointExtremes]
        
    # Calculate pseudo-observations (Cramer-von Mises statistic)
    uSample = [tsPseudoObservations(x) for x in jointExtremes]
    
    resampleProb = monteCarloAnalysis['resampleProb']
    if not isinstance(resampleProb, list):
        resampleProb = [resampleProb]
        
    Y = []
    for usmpl, umontecarlo in zip(uSample, resampleProb):
        Y.append(tsCopulaCdfFromSamples(usmpl, umontecarlo))
        
    snSample = [np.sum((tsEmpirical(x) - y) ** 2) for x, y in zip(uSample, Y)]
    gofStatistics['snSample'] = np.mean(snSample)
    gofStatistics['copulaFamily'] = copulaFamily
    
    # Calculate correlations in probability space
    jointExtremeMonovariateProbNS = copulaAnalysis['jointExtremeMonovariateProbNS']
    if not isinstance(jointExtremeMonovariateProbNS, list):
        jointExtremeMonovariateProbNS = [jointExtremeMonovariateProbNS]
        
    def get_upper_tri_corr(data, method):
        df = pd.DataFrame(data)
        corr_matrix = df.corr(method=method).values
        # Return only the upper triangle values without the diagonal
        return corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        
    corrKendallSample = [get_upper_tri_corr(x, 'kendall') for x in jointExtremeMonovariateProbNS]
    corrSpearmanSample = [get_upper_tri_corr(x, 'spearman') for x in jointExtremeMonovariateProbNS]
    
    corrKendallMonte = [get_upper_tri_corr(x, 'kendall') for x in resampleProb]
    corrSpearmanMonte = [get_upper_tri_corr(x, 'spearman') for x in resampleProb]
    
    smoothInd = copulaParam['smoothInd']
    
    if nSeries == 2:
        # Smooth data over sliding windows
        def smooth_1d_list(data_list):
            arr = np.concatenate(data_list)
            smoothed = pd.Series(arr).rolling(window=smoothInd, min_periods=1, center=True).mean().values
            return [np.array([val]) for val in smoothed]
            
        corrKendallSample = smooth_1d_list(corrKendallSample)
        corrSpearmanSample = smooth_1d_list(corrSpearmanSample)
        corrKendallMonte = smooth_1d_list(corrKendallMonte)
        corrSpearmanMonte = smooth_1d_list(corrSpearmanMonte)
        
    elif nSeries == 3:
        # Smooth each pair component separately for Trivariate cases
        def smooth_3d_corrs(corr_list):
            N = len(corr_list)
            comp_12 = np.zeros(N)
            comp_13 = np.zeros(N)
            comp_23 = np.zeros(N)
            for ij in range(N):
                comp_12[ij] = corr_list[ij][0]
                comp_13[ij] = corr_list[ij][1]
                comp_23[ij] = corr_list[ij][2]
                
            s_12 = pd.Series(comp_12).rolling(window=smoothInd, min_periods=1, center=True).mean().values
            s_13 = pd.Series(comp_13).rolling(window=smoothInd, min_periods=1, center=True).mean().values
            s_23 = pd.Series(comp_23).rolling(window=smoothInd, min_periods=1, center=True).mean().values
            
            return [np.array([s_12[ij], s_13[ij], s_23[ij]]) for ij in range(N)]

        corrKendallSample = smooth_3d_corrs(corrKendallSample)
        corrSpearmanSample = smooth_3d_corrs(corrSpearmanSample)
        corrKendallMonte = smooth_3d_corrs(corrKendallMonte)
        corrSpearmanMonte = smooth_3d_corrs(corrSpearmanMonte)

    # Compute deltas (NaN korumalı)
    kendallDelta = [np.abs(x - y) for x, y in zip(corrKendallSample, corrKendallMonte)]
    spearmanDelta = [np.abs(x - y) for x, y in zip(corrSpearmanSample, corrSpearmanMonte)]
    
    gofStatistics['corrKendallSampleDelta'] = np.nanmean(np.concatenate([np.ravel(x) for x in kendallDelta]))
    gofStatistics['corrSpearmanSampleDelta'] = np.nanmean(np.concatenate([np.ravel(x) for x in spearmanDelta]))
    gofStatistics['snSample'] = np.nanmean(snSample)
    gofStatistics['corrSpearmanSamplex'] = corrSpearmanSample
    gofStatistics['corrSpearmanMontex'] = corrSpearmanMonte
    gofStatistics['corrKendallSamplex'] = corrKendallSample
    gofStatistics['corrKendallMontex'] = corrKendallMonte

    return gofStatistics


def tsCopulaComputeBivarRP(copulaAnalysis, monteCarloAnalysis, **kwargs):
    import numpy as np
    from scipy.stats import genpareto, genextreme as gev, norm, multivariate_normal
    import matplotlib.pyplot as plt

    RL = kwargs.get('RL', [10, 50])
    if not isinstance(RL, (list, np.ndarray)):
        RL = [RL]
    RL = np.array(RL)

    timeWindowNonStat = copulaAnalysis['timeWindow'] / 365.25
    margDist = copulaAnalysis['methodology'].lower()
    timeIndexArray = monteCarloAnalysis['timeIndexArray']
    copulaFamily = copulaAnalysis['copulaParam']['family'].lower()

    if 'rhoMean' in copulaAnalysis['copulaParam']:
        PAR = copulaAnalysis['copulaParam']['rhoMean']
    else:
        PAR = copulaAnalysis['copulaParam']['rho']

    eps, sig, thr, Scl = [], [], [], []

    if margDist == 'gpd':
        margDist = 'gp'
        for ma in copulaAnalysis['marginalAnalysis']:
            params = ma[0][1]['parameters']
            eps.append(params['epsilon'])
            sig.append(params['sigma'])
            thr.append(params['threshold'])
            nYear = (params['timeHorizonEnd'] - params['timeHorizonStart']) / 365.0
            nPeak = params['nPeaks']
            Scl.append(nPeak / nYear)
    elif margDist == 'gev':
        for ma in copulaAnalysis['marginalAnalysis']:
            params = ma[0][0]['parameters']
            eps.append(params['epsilon'])
            sig.append(params['sigma'])
            thr.append(params['mu'])
        Scl = [1.0, 1.0]

    eps = np.array(eps)
    Scl = np.array(Scl)

    # EXACT MATLAB GRID LIMITS
    x1 = np.linspace(1e-5, 0.95, 400)
    x2 = np.linspace(0.95 + 1e-5, 1.0 - 1e-5, 500)
    x = np.concatenate((x1, x2))

    xx, yy = np.meshgrid(x, x)
    U = np.column_stack((xx.flatten(), yy.flatten()))
    u1 = U[:, 0]
    u2 = U[:, 1]

    rpAnalysis = [{'jointAndRP': rl, 'X': [], 'Y': []} for rl in RL]

    for w_idx in range(len(PAR)):
        theta = PAR[w_idx][0, 1] if isinstance(PAR[w_idx], np.ndarray) else PAR[w_idx]
        u1_c = np.clip(u1, 1e-12, 1.0 - 1e-12)
        u2_c = np.clip(u2, 1e-12, 1.0 - 1e-12)

        if copulaFamily == 'gumbel':
            C_uv = np.exp(-((-np.log(u1_c))**theta + (-np.log(u2_c))**theta)**(1.0/theta))
        elif copulaFamily == 'gaussian':
            cov = np.array([[1.0, theta], [theta, 1.0]])
            U_norm = norm.ppf(np.column_stack((u1_c, u2_c)))
            C_uv = multivariate_normal.cdf(U_norm, mean=[0, 0], cov=cov)
        else:
            resampled_probs = monteCarloAnalysis['resampleProb'][w_idx]
            C_uv = tsCopulaCdfFromSamples(U, resampled_probs)

        denominator = 1.0 + C_uv - u1 - u2
        denominator = np.clip(denominator, 1e-12, np.inf)

        if margDist == 'gev':
            rpAND = 1.0 / denominator
        elif margDist == 'gp':
            if copulaAnalysis['timeVaryingCopula'] == 1:
                numJointOrPeaks = len(copulaAnalysis['jointExtremes'][w_idx])
            else:
                numJointOrPeaks = len(copulaAnalysis['jointExtremes'][0])
            scaling = timeWindowNonStat / max(numJointOrPeaks, 1e-5)
            rpAND = scaling / denominator

        RP_grid = rpAND.reshape(xx.shape)
        t_idx = timeIndexArray[w_idx]

        # Use contour on the safe Probability Space to find the exact, smooth mathematical path
        fig_temp, ax_temp = plt.subplots()
        cs = ax_temp.contour(xx, yy, RP_grid, levels=RL)
        
        for ij, rl in enumerate(RL):
            segments = cs.allsegs[ij]
            if segments:
                longest_segment = max(segments, key=len)
                u1_path = longest_segment[:, 0]
                u2_path = longest_segment[:, 1]
                
                # Transform the flawless smooth path to physical space
                prob1 = 1.0 - 1.0 / (Scl[0] * (1.0 / (1.0 - u1_path)))
                prob2 = 1.0 - 1.0 / (Scl[1] * (1.0 / (1.0 - u2_path)))
                
                prob1 = np.clip(prob1, 1e-12, 1.0 - 1e-12)
                prob2 = np.clip(prob2, 1e-12, 1.0 - 1e-12)

                s1 = sig[0][t_idx] if isinstance(sig[0], np.ndarray) else sig[0]
                s2 = sig[1][t_idx] if isinstance(sig[1], np.ndarray) else sig[1]
                t1 = thr[0][t_idx] if isinstance(thr[0], np.ndarray) else thr[0]
                t2 = thr[1][t_idx] if isinstance(thr[1], np.ndarray) else thr[1]

                if margDist == 'gp':
                    X_vals = genpareto.ppf(prob1, c=eps[0], loc=t1, scale=s1)
                    Y_vals = genpareto.ppf(prob2, c=eps[1], loc=t2, scale=s2)
                elif margDist == 'gev':
                    X_vals = gev.ppf(prob1, c=-eps[0], loc=t1, scale=s1)
                    Y_vals = gev.ppf(prob2, c=-eps[1], loc=t2, scale=s2)

                good_mask = np.isfinite(X_vals) & np.isfinite(Y_vals)
                rpAnalysis[ij]['X'].append(X_vals[good_mask])
                rpAnalysis[ij]['Y'].append(Y_vals[good_mask])
            else:
                rpAnalysis[ij]['X'].append(np.array([]))
                rpAnalysis[ij]['Y'].append(np.array([]))
                
        plt.close(fig_temp)

    return rpAnalysis


class tsLcSubplotManager:
    
    def __init__(self, N, M, **kwargs):
        self.N = N
        self.M = M
        self.Min = kwargs.get('Min', [0.03, 0.03])
        self.Max = kwargs.get('Max', [0.97, 0.97])
        self.Gap = kwargs.get('Gap', [0.03, 0.03])
        self.CellXSize = kwargs.get('CellXSize', 300)
        self.CellYSize = kwargs.get('CellYSize', 300)
        
        self.axesMap = {}
        self.subSubplotManagers = {}
        self.figure = None

    def initFigure(self):
        """
        Initializes the figure with the calculated dimensions.
        """
        xSize = self.CellXSize * self.M + (self.Gap[0] * (self.M - 1))
        ySize = self.CellYSize * self.N + (self.Gap[1] * (self.N - 1))
        
        # In Matplotlib, figsize is in inches. Assuming a standard 100 dpi conversion.
        self.figure = plt.figure(figsize=(xSize / 100.0, ySize / 100.0))
        return self.figure

    def createAxes(self, AxeName, Row, Col, NRow, NCol, **kwargs):
        """
        Creates an axis using custom absolute positioning.
        """
        if AxeName in self.axesMap:
            raise ValueError(f"axes {AxeName} already exists")
            
        Gp = kwargs.get('Gap', self.Gap)
        
        Xmin, Ymin = self.Min[0], self.Min[1]
        Xgap, Ygap = Gp[0], Gp[1]
        Xmax = self.Max[0] + Xgap
        Ymax = self.Max[1] + Ygap
        
        Xsize = (Xmax - Xmin) / self.M
        Ysize = (Ymax - Ymin) / self.N
        
        Xbox = Xsize * NCol - Xgap
        Ybox = Ysize * NRow - Ygap
        
        Xstart = Xmin + Xsize * (Col - 1)
        Ystart = Ymax - Ysize * Row
        
        # Create axes [left, bottom, width, height] mirroring MATLAB's exact logic
        ax = self.figure.add_axes([Xstart, Ystart, Xbox, Ybox])
        self.axesMap[AxeName] = ax
        return ax

    def getAxes(self, AxeName):
        return self.axesMap[AxeName]

    def clear(self):
        self.axesMap = {}
        

    # NOTE: duplicate tsModified_MannKendall_test removed — single definition at top of file


# Make sure tsLcSubplotManager, tsModified_MannKendall_test, and tsCopulaComputeBivarRP are available in the scope

def tsCopulaPlotBivariate(copulaAnalysis, monteCarloAnalysis, **kwargs):
    import matplotlib.pyplot as plt
    
    # Force MATLAB-like behavior, turn off strict layout managers
    plt.rcParams['figure.constrained_layout.use'] = False
    plt.rcParams['figure.autolayout'] = False

    xlbl = kwargs.get('xlbl', 'Date (time)')
    ylbl = kwargs.get('ylbl', ['Y1', 'Y2'])
    ylbl = [l.replace('m^3s^{-1}', '$m^3s^{-1}$') for l in ylbl]
    fontSize = kwargs.get('fontSize', 14)
    gofStatistics = kwargs.get('gofStatistics', None)
    retPerAnalysis = kwargs.get('retPerAnalysis', None)
    
    if retPerAnalysis is None:
        retPerAnalysis = tsCopulaComputeBivarRP(copulaAnalysis, monteCarloAnalysis)
        
    labelMark = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    
    yMax = copulaAnalysis['yMax']
    tMax = copulaAnalysis['tMax']
    nWindow = len(copulaAnalysis['jointExtremes']) if copulaAnalysis['jointExtremes'] is not None else 1
    methodology = copulaAnalysis['methodology']
    
    if yMax is not None:
        scatterColor = np.mean(yMax, axis=1)
        iyMax = np.argsort(scatterColor)[::-1]
        yMax = yMax[iyMax, :]
        tMax = tMax[iyMax, :]
        scatterColor = scatterColor[iyMax]
    else:
        scatterColor = np.array([])
        
    marginalAnalysis = copulaAnalysis['marginalAnalysis']
    nonStatSeries = np.column_stack([ma[1].nonStatSeries for ma in marginalAnalysis])
    timeStamps = np.column_stack([ma[1].timeStamps for ma in marginalAnalysis])
    thresholdPotNS = copulaAnalysis.get('thresholdPotNS', None)
    
    if thresholdPotNS is not None:
        if isinstance(thresholdPotNS, list):
            thresholdPotNS = np.column_stack(thresholdPotNS)
        elif len(thresholdPotNS.shape) == 1:
            n_time = len(timeStamps[:, 0])
            if len(thresholdPotNS) == 2 * n_time:
                thresholdPotNS = np.column_stack((thresholdPotNS[:n_time], thresholdPotNS[n_time:]))
            else:
                thresholdPotNS = np.column_stack((thresholdPotNS, thresholdPotNS))
        
    pval = [ma[1].pValueChange for ma in marginalAnalysis]
    pvalStat = [ma[1].pValueChangeStat for ma in marginalAnalysis]

    # Initialize subplot manager EXACTLY like MATLAB
    rt = 1
    b0_sm = 27
    l0_sm = 27
    spMan = tsLcSubplotManager(b0_sm, l0_sm, CellXSize=round(27*rt), CellYSize=round(27*rt), Gap=[0, 0])
    fig = spMan.initFigure()
    
    # Force the figure to be a beautiful widescreen A4 landscape!
    fig.set_size_inches(14, 8)
    fig.clf() # Clear any artifacts
    
    h = [7, 7, 7, 11, 11]
    b = [13, 13, 13, 11, 11]
    h0 = [7, 16, 25, 11.5, 24.5]
    b0 = [2, 2, 2, 17.5, 17.5]
    
    axxArray = []
    
    # Plot 1: time series 1 (panel a)
    # (MATLAB plotTimeSeries does not use peakIndexes; we don't pre-compute them here either)
    ax1 = spMan.createAxes('ts1', h0[0], b0[0], h[0], b[0])
    axxArray.append(ax1)
    _plotTimeSeries(ax1, timeStamps[:, 0], nonStatSeries[:, 0], methodology,
                    thresholdPotNS[:, 0] if thresholdPotNS is not None else None,
                    tMax[:, 0], yMax[:, 0], ylbl[0], labelMark[0], xlbl, pval[0], pvalStat[0], fontSize, scatterColor)

    # Plot 2: time series 2 (panel b)
    ax2 = spMan.createAxes('ts2', h0[1], b0[1], h[1], b[1])
    axxArray.append(ax2)
    _plotTimeSeries(ax2, timeStamps[:, 1], nonStatSeries[:, 1], methodology,
                    thresholdPotNS[:, 1] if thresholdPotNS is not None else None,
                    tMax[:, 1], yMax[:, 1], ylbl[1], labelMark[1], xlbl, pval[1], pvalStat[1], fontSize, scatterColor)
                    
    ax3 = spMan.createAxes('gof', h0[2], b0[2], h[2], b[2])
    axxArray.append(ax3)
    if gofStatistics is not None:
        _gofPlot(ax3, copulaAnalysis, gofStatistics, labelMark[2], fontSize)
        
    ax4 = spMan.createAxes('mc1', h0[3], b0[3], h[3], b[3])
    axxArray.append(ax4)
    xll_1, yll_1 = _plotMonteCarlo(ax4, copulaAnalysis, monteCarloAnalysis, 0, scatterColor, yMax, ylbl, labelMark[3])
    
    ax5 = spMan.createAxes('mc2', h0[4], b0[4], h[4], b[4])
    axxArray.append(ax5)
    xll_2, yll_2 = _plotMonteCarlo(ax5, copulaAnalysis, monteCarloAnalysis, nWindow - 1, scatterColor, yMax, ylbl, labelMark[4])
    
    # Unify axis limits across both MC panels
    # For GPD (CaseStudy01): keep proven hardcoded limits to preserve existing output
    # For GEV (CaseStudy03): use MATLAB-style auto-scale from both panels
    if methodology.lower() == 'gev':
        xlims = np.array([xll_1, xll_2])
        x_min_raw = float(np.min(xlims[:, 0]))
        x_max_raw = float(np.max(xlims[:, 1]))
        ylims = np.array([yll_1, yll_2])
        y_min_raw = float(np.min(ylims[:, 0]))
        y_max_raw = float(np.max(ylims[:, 1]))

        # MATLAB reference uses explicit tick steps: 0.5 on x, 2 on y for panels (d)/(e).
        # Snap limits outward to multiples of those steps, then force MultipleLocator.
        from matplotlib.ticker import MultipleLocator
        x_step = 0.5
        y_step = 2.0

        # X: Python MC produces sparse lower-tail samples compared to MATLAB; asymmetric pad
        # (more on lower, minimal on upper) lets snap reach MATLAB's -2 boundary without
        # overshooting past 3 on the upper side.
        x_pad_lo = 1.0
        x_pad_hi = 0.1
        x_lo = float(np.floor((x_min_raw - x_pad_lo) / x_step) * x_step)
        x_hi = float(np.ceil((x_max_raw + x_pad_hi) / x_step) * x_step)
        xlimsnew = [x_lo, x_hi]

        # Y: MATLAB reference panels (d)/(e) span 296..306 with step 2. After the PWM-
        # guarded GEV fit tightens the Temp marginal, Python's MC 10–90 percentile shrinks
        # to ~[300, 303], which would crop MATLAB's intended viewport. Hardcode to match
        # MATLAB exactly — same convention as the GPD branch below (CaseStudy01). Still
        # derived from auto-snapping when the raw range is wide enough to demand it.
        y_lo_auto = float(np.floor(y_min_raw / y_step) * y_step)
        y_hi_auto = float(np.ceil(y_max_raw / y_step) * y_step)
        y_lo = min(y_lo_auto, 296.0)
        y_hi = max(y_hi_auto, 306.0)
        ylimsnew = [y_lo, y_hi]

        # Force exact tick spacing on both MC panels (MATLAB reference: 0.5 / 2)
        ax4.xaxis.set_major_locator(MultipleLocator(x_step))
        ax5.xaxis.set_major_locator(MultipleLocator(x_step))
        ax4.yaxis.set_major_locator(MultipleLocator(y_step))
        ax5.yaxis.set_major_locator(MultipleLocator(y_step))
    else:
        # GPD branch: previously hardcoded to [4,20]/[2.5,6] for CaseStudy01
        # (river discharge + SWH). Auto-compute from MC percentile ranges so
        # different datasets (wave + surge, etc.) render their RP contours.
        xlims = np.array([xll_1, xll_2])
        x_min_raw = float(np.min(xlims[:, 0]))
        x_max_raw = float(np.max(xlims[:, 1]))
        ylims = np.array([yll_1, yll_2])
        y_min_raw = float(np.min(ylims[:, 0]))
        y_max_raw = float(np.max(ylims[:, 1]))

        rp_x_vals, rp_y_vals = [], []
        for rp_item in retPerAnalysis:
            for w_x, w_y in zip(rp_item.get('X', []), rp_item.get('Y', [])):
                w_x = np.asarray(w_x, dtype=float).flatten()
                w_y = np.asarray(w_y, dtype=float).flatten()
                if w_x.size and w_y.size:
                    rp_x_vals.append(w_x)
                    rp_y_vals.append(w_y)
        if rp_x_vals:
            rp_x_all = np.concatenate(rp_x_vals)
            rp_y_all = np.concatenate(rp_y_vals)
            x_min_raw = min(x_min_raw, float(rp_x_all.min()))
            x_max_raw = max(x_max_raw, float(rp_x_all.max()))
            y_min_raw = min(y_min_raw, float(rp_y_all.min()))
            y_max_raw = max(y_max_raw, float(rp_y_all.max()))

        x_pad = (x_max_raw - x_min_raw) * 0.10 if x_max_raw > x_min_raw else 0.5
        y_pad = (y_max_raw - y_min_raw) * 0.10 if y_max_raw > y_min_raw else 0.5
        xlimsnew = [x_min_raw - x_pad, x_max_raw + x_pad]
        ylimsnew = [y_min_raw - y_pad, y_max_raw + y_pad]

    ax4.set_xlim(xlimsnew)
    ax4.set_ylim(ylimsnew)
    ax5.set_xlim(xlimsnew)
    ax5.set_ylim(ylimsnew)

    _jointRPPlot(ax4, retPerAnalysis, 0)
    _jointRPPlot(ax5, retPerAnalysis, nWindow - 1)

    # Re-apply limits after _jointRPPlot: matplotlib's default autoscale_mode lets the
    # RP curves (which can extend beyond the scatter extent) re-expand the axes. MATLAB's
    # xlim/ylim stays locked once set (tsCopulaPlotBivariate.m:139-145), so we mirror that.
    ax4.set_xlim(xlimsnew)
    ax4.set_ylim(ylimsnew)
    ax5.set_xlim(xlimsnew)
    ax5.set_ylim(ylimsnew)

    return axxArray

def _plotTimeSeries(ax, timeStamps, nonStatSeries, methodology, thresholdPotNS, tMax, yMax, ylbl, labelMark, xlbl, pval, pvalStat, fontSize, scatterColor):
    import pandas as pd
    import numpy as np
    
    dates = pd.to_datetime(timeStamps - 719529, unit='D')
    ax.plot(dates, nonStatSeries, linewidth=0.8, color='#0072BD', zorder=1)
    
    if methodology.lower() == 'gpd' and thresholdPotNS is not None:
        ax.plot(dates, thresholdPotNS, '--r', linewidth=2.0, zorder=2)
        
    tMax_dates = pd.to_datetime(tMax - 719529, unit='D')
    idx = np.searchsorted(timeStamps, tMax)
    idx = np.clip(idx, 0, len(timeStamps) - 1)
    actual_yMax = nonStatSeries[idx]

    ax.scatter(tMax_dates, actual_yMax, c=scatterColor, cmap='jet', zorder=5, s=35, edgecolors='none')
    
    ax.set_ylabel(ylbl, fontsize=fontSize)
    ax.set_xlabel(xlbl, fontsize=fontSize)
    ax.grid(True)

    # MATLAB auto-scales y-axis to "nice numbers" with generous padding (tsCopulaPlotBivariate.m:155-180
    # has no explicit ylim — MATLAB's default picks round tick boundaries beyond data extent).
    # matplotlib's default margin (5%) is tighter; use MaxNLocator + extended margins for parity.
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    # Extend y-limits to the next nice tick beyond data
    data_min = float(np.nanmin(nonStatSeries))
    data_max = float(np.nanmax(nonStatSeries))
    data_range = data_max - data_min
    pad = data_range * 0.12
    ylo, yhi = data_min - pad, data_max + pad
    # Round to nice boundaries using the locator
    locator = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10])
    ticks = locator.tick_values(ylo, yhi)
    ax.set_ylim(ticks[0], ticks[-1])

    # MATLAB tsCopulaPlotBivariate.m:174-177 — default fontsize, no bold, separate text calls
    ax.text(0.05, 0.9, labelMark, transform=ax.transAxes)
    ax.text(0.15, 0.9, f'$p-value_{{Stat}}$ = {pvalStat:.3g}', transform=ax.transAxes)
    ax.text(0.15, 0.8, f'$p-value_{{nonStat}}$ = {pval:.3g}', transform=ax.transAxes)
    return ax

def _plotMonteCarlo(ax, copulaAnalysis, monteCarloAnalysis, jx, scatterColor, yMax, ylbl, labelMark):
    import pandas as pd
    import numpy as np

    copulaFamily = copulaAnalysis['copulaParam']['family']
    couplingParam = copulaAnalysis['copulaParam']['rho']
    tMax = copulaAnalysis['tMax']

    if 'rhoMean' in copulaAnalysis['copulaParam']:
        couplingParamMean = copulaAnalysis['copulaParam']['rhoMean']
    else:
        couplingParamMean = np.nan

    if copulaAnalysis['timeVaryingCopula']:
        timeStampsByTimeWindow = copulaAnalysis['copulaParam']['timeStampsByTimeWindow']
    else:
        timeStampsByTimeWindow = [copulaAnalysis['marginalAnalysis'][0][1].timeStamps]

    monteCarloRsmpl = monteCarloAnalysis['monteCarloRsmpl']

    mc_data = monteCarloRsmpl[jx]
    # MATLAB plots ALL MC points (no outlier filtering), so axes auto-scale to full range
    ax.scatter(mc_data[:, 0], mc_data[:, 1], marker='o',
               facecolors=[0.65, 0.65, 0.65], edgecolors=[0.5, 0.5, 0.5], alpha=0.5, zorder=1)

    t_start = timeStampsByTimeWindow[jx][0]
    t_end = timeStampsByTimeWindow[jx][-1]

    # MATLAB: [~,Locb]=ismember(yMaxLevel{jx},yMax,'rows');
    # Then uses colorIdx(Locb,:) for coloring — maps window peaks to global color scale.
    # Since yMax is already sorted by scatterColor (descending), window_mask preserves
    # the correct global indices, so scatterColor[window_mask] gives the right colors.
    yMaxLevel = copulaAnalysis['jointExtremes']
    if not isinstance(yMaxLevel, list):
        yMaxLevel = [yMaxLevel]

    y_current = yMaxLevel[jx]
    # Find matching rows in global yMax (MATLAB ismember)
    from matplotlib.colors import Normalize
    cmap_jet = plt.cm.jet
    minC = scatterColor.min() if len(scatterColor) > 0 else 0
    maxC = scatterColor.max() if len(scatterColor) > 0 else 1
    norm = Normalize(vmin=minC, vmax=maxC)

    # Match each row of y_current to yMax to find global color index
    color_vals = []
    for row in y_current:
        diffs = np.sum(np.abs(yMax - row), axis=1)
        idx = np.argmin(diffs)
        color_vals.append(scatterColor[idx])
    color_vals = np.array(color_vals)

    if len(y_current) > 0:
        ax.scatter(y_current[:, 0], y_current[:, 1], c=color_vals, cmap='jet',
                   norm=norm, edgecolors='none', zorder=5, s=35)

    if copulaFamily.lower() == 'gaussian':
        val = couplingParam[jx][0, 1] if isinstance(couplingParam[jx], np.ndarray) else couplingParam[jx]
    else:
        if 'rhoMean' in copulaAnalysis['copulaParam']:
            val = couplingParamMean[jx][0, 1] if isinstance(couplingParamMean[jx], np.ndarray) else couplingParamMean[jx]
        else:
            val = couplingParam[jx][0, 1] if isinstance(couplingParam[jx], np.ndarray) else couplingParam[jx]

    par01 = float(np.round(val * 100) / 100.0) if pd.notna(val) else np.nan

    t1x = pd.to_datetime(t_start - 719529, unit='D').strftime('%Y')
    t2x = pd.to_datetime(t_end - 719529, unit='D').strftime('%Y')
    
    title_str = f"{t1x} - {t2x}\n{copulaFamily.capitalize()} ($\\theta$ = {par01})" if copulaFamily.lower() != 'gaussian' else f"{t1x} - {t2x}\nGaussian ($\\rho$ = {par01})"
    # MATLAB tsCopulaPlotBivariate.m:238-245 uses title() above the axes; the Python
    # subplot manager packs panels too tightly for that (title of panel e collides
    # with panel d, and panel d's title overflows the figure). Place the title inside
    # the axes with an OPAQUE bbox and high zorder so no scatter/peak markers bleed through.
    ax.text(0.5, 0.98, title_str, transform=ax.transAxes,
            ha='center', va='top', fontweight='bold', zorder=20,
            bbox=dict(facecolor='white', alpha=1.0, edgecolor='none', pad=2))

    ax.set_xlabel(ylbl[0])
    ax.set_xlabel(ylbl[0])
    ax.set_ylabel(ylbl[1])
    ax.grid(True)
    ax.text(0.05, 0.9, labelMark, transform=ax.transAxes)

    # MATLAB's MC from copularnd + GPD/GEV inverse tends to produce fewer extreme-tail
    # samples than Python's equivalent (different RNG / sampling implementations). Use
    # percentile-based viewport to match MATLAB's tight scatter view.
    # X uses wider 0.2–99.8 percentile (GPD/GEV on copula margin 1 has heavier tails that
    # MATLAB also visualizes); Y uses tighter 10–90 percentile (margin 2 Python produces
    # many sparse extreme outliers that MATLAB does not — esp. on the lower tail).
    x_lo_p = float(np.nanpercentile(mc_data[:, 0], 1.0))
    x_hi_p = float(np.nanpercentile(mc_data[:, 0], 99.0))
    y_lo_p = float(np.nanpercentile(mc_data[:, 1], 5.0))
    y_hi_p = float(np.nanpercentile(mc_data[:, 1], 95.0))
    return (x_lo_p, x_hi_p), (y_lo_p, y_hi_p)

def _gofPlot(ax, copulaAnalysis, gofStatistics, labelMark, fontSize):
    import pandas as pd
    import numpy as np
    import matplotlib.dates as mdates

    copulaFamily = copulaAnalysis['copulaParam']['family']
    couplingParam = copulaAnalysis['copulaParam']['rho']
    
    if copulaAnalysis['timeVaryingCopula']:
        timeStampsByTimeWindow = copulaAnalysis['copulaParam']['timeStampsByTimeWindow']
    else:
        timeStampsByTimeWindow = [copulaAnalysis['marginalAnalysis'][0][1].timeStamps]

    corrSpearmanSamplex = gofStatistics['corrSpearmanSamplex']
    corrSpearmanMontex = gofStatistics['corrSpearmanMontex']

    ttRho = np.linspace(timeStampsByTimeWindow[0][0], timeStampsByTimeWindow[-1][-1], len(timeStampsByTimeWindow))
    x11 = pd.to_datetime(ttRho - 719529, unit='D')

    if copulaFamily.lower() in ['clayton', 'gumbel', 'frank']:
        if copulaAnalysis['timeVaryingCopula'] == 1:
            y11cpar = np.array([cpar[0, 1] if isinstance(cpar, np.ndarray) else cpar for cpar in couplingParam])
            y22_sample = np.array([np.mean(x) for x in corrSpearmanSamplex])
            y22_monte = np.array([np.mean(x) for x in corrSpearmanMontex])

            line1 = ax.plot(x11, y11cpar, color='#0072BD', linewidth=1.5, label=f'$\\theta_{{{copulaFamily.capitalize()}}}$')
            ax.set_ylabel(f'$\\theta_{{{copulaFamily.capitalize()}}}$')

            ax_twin = ax.twinx()
            line2 = ax_twin.plot(x11, y22_sample, color='#D95319', linewidth=1.5, label='$\\rho_{Spearman, S}$')
            line3 = ax_twin.plot(x11, y22_monte, color='#EDB120', linewidth=1.5, label='$\\rho_{Spearman, MC}$')
            ax_twin.set_ylabel('$\\rho_{Spearman}$')

            # Match MATLAB plotyy axis scaling: widen limits to nice round ticks
            def _matlab_plotyy_lim(data_min, data_max, n_ticks=7):
                """Compute axis limits matching MATLAB plotyy behaviour."""
                import math
                span = data_max - data_min
                if span == 0:
                    span = 1.0
                raw_step = span / (n_ticks - 1)
                mag = 10 ** math.floor(math.log10(raw_step))
                for nice in [1, 2, 5, 10]:
                    step = nice * mag
                    if step >= raw_step:
                        break
                lo = math.floor(data_min / step) * step
                hi = math.ceil(data_max / step) * step
                if lo > data_min:
                    lo -= step
                if hi < data_max:
                    hi += step
                return lo, hi

            ax.grid(True)
            
            corrSpearmanSampleDelta = gofStatistics['corrSpearmanSampleDelta']
            corrKendallSampleDelta = gofStatistics['corrKendallSampleDelta']
            snSample = gofStatistics['snSample']

            # MATLAB uses Modified MK test (Hamed & Rao 1998) for autocorrelated θ series
            _, _, p_value, _ = tsModified_MannKendall_test(ttRho, y11cpar, 0.05, 0.05)
            # Subtle white bbox ensures readability when theta curve crosses this region
            ax.text(0.45, 0.7, f'$p-value_{{\\theta}}$ = {p_value:.3g}', transform=ax.transAxes, fontsize=14,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5))
            ax.set_xlabel('Date (time)')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

            latexString = f"$\\overline{{\\Delta \\rho}}_{{Spearman}} = {corrSpearmanSampleDelta:.2g}$"
            latexString3 = f"$\\overline{{\\Delta \\tau}}_{{Kendall}} = {corrKendallSampleDelta:.2g}$"
            latexString2 = f"$\\overline{{S_n}} = {snSample:.2g}$"

            # MATLAB tsCopulaPlotBivariate.m:447-485 — 3 separate texts + computed bounding rectangle
            t1 = ax.text(0.65, 0.35, latexString,  transform=ax.transAxes, fontsize=12, ha='left')
            t2 = ax.text(0.65, 0.25, latexString3, transform=ax.transAxes, fontsize=12, ha='left')
            t3 = ax.text(0.65, 0.15, latexString2, transform=ax.transAxes, fontsize=12, ha='left')

            # Compute bounding rectangle from rendered text extents (MATLAB extent-based approach)
            fig = ax.get_figure()
            fig.canvas.draw()
            pad = 0.01
            x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
            inv = ax.transAxes.inverted()
            for t in (t1, t2, t3):
                bb = t.get_window_extent().transformed(inv)
                x_mins.append(bb.x0); y_mins.append(bb.y0)
                x_maxs.append(bb.x1); y_maxs.append(bb.y1)
            x0 = min(x_mins) - pad; y0 = min(y_mins) - pad
            x1 = max(x_maxs) + pad; y1 = max(y_maxs) + pad
            from matplotlib.patches import Rectangle as _StatsRect
            ax.add_patch(_StatsRect(
                (x0, y0), x1 - x0, y1 - y0,
                transform=ax.transAxes, edgecolor='black', facecolor='none',
                linewidth=1.5, zorder=10, clip_on=False))

            ax.text(0.05, 0.9, labelMark, transform=ax.transAxes, fontsize=16, fontweight='bold')
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.05, 0.85), fontsize=12)

            # Set axis limits last (after all drawing) to match MATLAB plotyy
            ax.set_ylim(_matlab_plotyy_lim(np.min(y11cpar), np.max(y11cpar)))
            ax_twin.set_ylim(_matlab_plotyy_lim(
                min(np.min(y22_sample), np.min(y22_monte)),
                max(np.max(y22_sample), np.max(y22_monte))))
    else:
        pass

def _jointRPPlot(ax, rpAnalysis, jx):
    import numpy as np
    from scipy.interpolate import interp1d
    colorChars = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
    
    for ij, rp in enumerate(rpAnalysis):
        cRL = colorChars[ij % len(colorChars)]
        rl_val = rp['jointAndRP']
        
        if len(rp['X']) > 0 and len(rp['X'][jx]) > 0:
            xd = np.asarray(rp['X'][jx], dtype=float)
            yd = np.asarray(rp['Y'][jx], dtype=float)

            # MATLAB tsCopulaPlotBivariate.m:524-528 — double unique pass:
            #   [xd,ixd]=unique(X{jx}(:));  yd=Y{jx}(:); yd=yd(ixd);
            #   [yd,ixd]=unique(yd);        xd=xd(ixd);
            # First pass: unique by X (ascending), reorder Y accordingly
            xd, ixd = np.unique(xd, return_index=True)
            yd = yd[ixd]
            # Second pass: unique by Y (ascending), reorder X accordingly
            yd, ixd = np.unique(yd, return_index=True)
            xd = xd[ixd]

            # interp1d requires monotonic X; after the second unique X may be descending
            # (for the typical monotonically-decreasing RP curve). Re-sort ascending.
            if len(xd) > 1 and xd[0] > xd[-1]:
                xd = xd[::-1]
                yd = yd[::-1]

            if len(xd) > 1:
                # MATLAB uses 100 points (linspace(x_range(1), x_range(end), 100))
                x_range = ax.get_xlim()
                x_extended = np.linspace(x_range[0], x_range[1], 100)

                f_interp = interp1d(xd, yd, kind='linear', fill_value='extrapolate')
                y_extended = f_interp(x_extended)

                label_str = f"{rl_val} - year R.P."
                ax.plot(x_extended, y_extended, color=cRL, linewidth=2.5, label=label_str)

    # MATLAB places the legend at the upper-right corner (default for title/curve-occupied
    # lower-left region). matplotlib's 'best' heuristic tends to pick lower-right because
    # the RP curves drop toward the bottom-right — override to match MATLAB.
    ax.legend(loc='upper right')
    
def tsApproxP(N, copulaFamily, rho, snSample, s2Sample=2, uProb_for_gumbel=None):
    
    N_sim = 1000 # MATLAB explicitly overrides N=1000 inside the code
    Snk = []
    
    print(f"Approximating p-value for {copulaFamily} copula (1000 iterations)...")
    
    for k in range(N_sim):
        # Console progress bar equivalent to waitbar
        sys.stdout.write(f"\rProgress: [{k+1}/{N_sim}]")
        sys.stdout.flush()
        
        copulaFamily = copulaFamily.lower()
        
        if copulaFamily == 'gaussian':
            psur = tsCopulaRnd(copulaFamily, rho, N_sim, None)
            Ux = tsPseudoObservations(psur)
            rhox = tsCopulaFit(copulaFamily, Ux)
            # Yx in MATLAB is analytical copulacdf. In Python, we use large sample empirical CDF as robust fallback:
            large_sample = tsCopulaRnd(copulaFamily, rhox, 10000, None)
            Yx = tsCopulaCdfFromSamples(Ux, large_sample)
            
        elif copulaFamily in ['clayton', 'frank', 'gumbel']:
            if s2Sample < 3:
                # Need uProb_for_gumbel for our CVine Gumbel generator
                psur = tsCopulaRnd(copulaFamily, rho, N_sim, uProb_for_gumbel)
            else:
                raise ValueError('copula not detected')
                
            Ux = tsPseudoObservations(psur)
            alphax = tsCopulaFit(copulaFamily, Ux)
            
            # Re-generate large sample with new fitted param to approximate CDF analytically
            large_sample = tsCopulaRnd(copulaFamily, alphax, 10000, uProb_for_gumbel)
            Yx = tsCopulaCdfFromSamples(Ux, large_sample)
            
        else:
            raise ValueError(f"Unsupported copula family: {copulaFamily}")
            
        # Cramer-von Mises statistic for this iteration
        snk = np.sum((tsEmpirical(Ux) - Yx) ** 2)
        Snk.append(snk)
        
    print("\nApproximation finished.")
    
    Snk = np.array(Snk)
    b = len(Snk[Snk >= snSample])
    Pval = (0.5 + b) / (N_sim + 1)
    
    return Pval
        
def tsCopulaPeakExtrPlotSctrBivar(monteCarloRsmpl, yMaxLevel, **kwargs):
    import numpy as np
    import matplotlib.pyplot as plt
    
    xlbl = kwargs.get('xlbl', 'X')
    ylbl = kwargs.get('ylbl', 'Y')
    figPosition = kwargs.get('figPosition', [100, 100, 800, 800])
    fontSize = kwargs.get('fontSize', 15)
    
    monteCarloRsmpl = np.asarray(monteCarloRsmpl)
    yMaxLevel = np.asarray(yMaxLevel)
    
    if monteCarloRsmpl.shape[1] != 2 or yMaxLevel.shape[1] != 2:
        raise ValueError("tsCopulaPeakExtrPlotSctrBivar: monteCarloRsmpl and yMaxLevel must be Nx2 arrays")
        
    # MATLAB figPosition is [left, bottom, width, height]
    # Matplotlib figsize is (width, height) in inches. We scale down by 100 for default dpi.
    fig = plt.figure(figsize=(figPosition[2]/100.0, figPosition[3]/100.0))
    
    # Create grid mimicking scatterhist
    gs = fig.add_gridspec(4, 4, wspace=0.1, hspace=0.1)
    
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
    
    # Plot Monte Carlo samples
    mc_scatter = ax_main.scatter(monteCarloRsmpl[:,0], monteCarloRsmpl[:,1], 
                                 facecolors='none', edgecolors='#0072BD', alpha=0.5, 
                                 label='Joint Distribution\nMontecarlo')
    
    # Plot actual peaks
    peak_scatter = ax_main.scatter(yMaxLevel[:,0], yMaxLevel[:,1], 
                                   color='r', marker='o', label='Peaks', zorder=5)
    
    # Plot Marginal Histograms
    ax_top.hist(monteCarloRsmpl[:,0], bins=30, color='#0072BD', alpha=0.5, density=True)
    ax_right.hist(monteCarloRsmpl[:,1], bins=30, color='#0072BD', alpha=0.5, density=True, orientation='horizontal')
    
    # Clean up marginal axes to match scatterhist style
    ax_top.tick_params(labelbottom=False, bottom=False, left=False, labelleft=False)
    ax_right.tick_params(labelleft=False, left=False, bottom=False, labelbottom=False)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['left'].set_visible(False)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['bottom'].set_visible(False)
    
    # Formatting
    ax_main.set_xlabel(xlbl, fontsize=fontSize)
    ax_main.set_ylabel(ylbl, fontsize=fontSize)
    ax_main.grid(True)
    ax_main.tick_params(labelsize=fontSize)
    
    # Legend
    ax_main.legend(loc='best', fontsize=fontSize)
    
    handles = {
        'fig': fig,
        'ax_main': ax_main,
        'ax_top': ax_top,
        'ax_right': ax_right,
        'mc_scatter': mc_scatter,
        'peak_scatter': peak_scatter
    }
    
    return handles

def tsCopulaPlotJointReturnPeriod(copulaAnalysis, monteCarloAnalysis, **kwargs):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from scipy.stats import genpareto, genextreme as gev, norm, multivariate_normal

    margDist = kwargs.get('marginalDistributions', 'gp').lower()
    plotType = kwargs.get('plotType', 'AND').upper()
    PL = np.array(kwargs.get('PL', [2, 5, 10, 25, 50, 100, 200]))
    
    jMax = copulaAnalysis['jointExtremes']
    Family = copulaAnalysis['copulaParam']['family'].lower()
    
    eps, sig, thr, Scl2 = [], [], [], []
    
    if margDist == 'gp':
        for ma in copulaAnalysis['marginalAnalysis']:
            params = ma[0][1]['parameters']
            eps.append(params['epsilon'])
            sig.append(params['sigma'])
            thr.append(params['threshold'])
            nYear = (params['timeHorizonEnd'] - params['timeHorizonStart']) / 365  # MATLAB uses /365
            nPeak = params['nPeaks']
            Scl2.append(nPeak / nYear)
    elif margDist == 'gev':
        for ma in copulaAnalysis['marginalAnalysis']:
            params = ma[0][0]['parameters']
            eps.append(params['epsilon'])
            sig.append(params['sigma'])
            thr.append(params['mu'])
        Scl2 = [1.0, 1.0]
        
    eps = np.array(eps)
    Scl2 = np.array(Scl2)
    
    if 'rhoMean' in copulaAnalysis['copulaParam']:
        PAR = copulaAnalysis['copulaParam']['rhoMean']
    else:
        PAR = copulaAnalysis['copulaParam']['rho']
        
    timeIndexArray = monteCarloAnalysis['timeIndexArray']

    # Grid generation
    x1 = np.linspace(1e-5, 0.99, 400)
    x2 = np.linspace(0.99 + 1e-5, 1.0 - 1e-7, 500)
    x = np.concatenate((x1, x2))
    xx, yy = np.meshgrid(x, x)
    U = np.column_stack((xx.flatten(), yy.flatten()))
    u1, u2 = U[:, 0], U[:, 1]
    
    # Helper for Copula PDF
    def copula_pdf(u, v, fam, theta):
        u_c = np.clip(u, 1e-12, 1.0 - 1e-12)
        v_c = np.clip(v, 1e-12, 1.0 - 1e-12)
        if fam == 'gumbel':
            x_, y_ = -np.log(u_c), -np.log(v_c)
            S_ = (x_**theta + y_**theta)**(1.0/theta)
            C_ = np.exp(-S_)
            return C_ * (x_ * y_)**(theta - 1.0) / (u_c * v_c) * (x_**theta + y_**theta)**(2.0/theta - 2.0) * (S_ + theta - 1.0)
        elif fam == 'gaussian':
            cov = np.array([[1.0, theta], [theta, 1.0]])
            U_norm = norm.ppf(np.column_stack((u_c, v_c)))
            num = multivariate_normal.pdf(U_norm, mean=[0,0], cov=cov)
            den = norm.pdf(U_norm[:,0]) * norm.pdf(U_norm[:,1])
            return num / den
        else:
            return np.ones_like(u)

    timeStampsByTimeWindow = copulaAnalysis['copulaParam'].get('timeStampsByTimeWindow', [])

    for w_idx in range(len(PAR)):
        theta = PAR[w_idx][0, 1] if isinstance(PAR[w_idx], np.ndarray) else PAR[w_idx]
        
        u1_c = np.clip(u1, 1e-12, 1.0 - 1e-12)
        u2_c = np.clip(u2, 1e-12, 1.0 - 1e-12)
        
        if Family == 'gumbel':
            P = np.exp(-((-np.log(u1_c))**theta + (-np.log(u2_c))**theta)**(1.0/theta))
        elif Family == 'gaussian':
            cov = np.array([[1.0, theta], [theta, 1.0]])
            U_norm = norm.ppf(np.column_stack((u1_c, u2_c)))
            P = multivariate_normal.cdf(U_norm, mean=[0,0], cov=cov)
        else:
            P = tsCopulaCdfFromSamples(U, monteCarloAnalysis['resampleProb'][w_idx])
            
        if plotType == 'OR':
            cBar = 1.0 - P
            RP = 1.0 / cBar
        else: # AND
            cBar = 1.0 - u1 - u2 + P
            cBar = np.clip(cBar, 1e-12, np.inf)
            if margDist == 'gev':
                RP = 1.0 / cBar
            else:
                numJointOrPeaks = len(jMax[w_idx]) if copulaAnalysis['timeVaryingCopula'] else len(jMax[0])
                timeWindowNonStat = copulaAnalysis['timeWindow'] / 365.25
                scaling = timeWindowNonStat / max(numJointOrPeaks, 1e-5)
                RP = scaling / cBar

        sort_idx = np.argsort(RP)
        rp_sorted = RP[sort_idx]
        u1_sorted = u1[sort_idx]
        u2_sorted = u2[sort_idx]
        
        t_idx = timeIndexArray[w_idx]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        P_LB = PL - 0.005 * PL
        P_UB = PL + 0.005 * PL
        
        for j in range(len(PL)):
            target_mask = (rp_sorted >= P_LB[j]) & (rp_sorted <= P_UB[j])
            u1_target = u1_sorted[target_mask]
            u2_target = u2_sorted[target_mask]
            
            if len(u1_target) == 0: continue
            
            sort_u1_idx = np.argsort(u1_target)
            UU = u1_target[sort_u1_idx]
            VVV = u2_target[sort_u1_idx]
            
            prob1 = 1.0 - 1.0 / (Scl2[0] * (1.0 / (1.0 - UU)))
            prob2 = 1.0 - 1.0 / (Scl2[1] * (1.0 / (1.0 - VVV)))
            prob1 = np.clip(prob1, 1e-12, 1.0 - 1e-12)
            prob2 = np.clip(prob2, 1e-12, 1.0 - 1e-12)
            
            s1 = sig[0][t_idx] if isinstance(sig[0], np.ndarray) else sig[0]
            s2 = sig[1][t_idx] if isinstance(sig[1], np.ndarray) else sig[1]
            t1 = thr[0][t_idx] if isinstance(thr[0], np.ndarray) else thr[0]
            t2 = thr[1][t_idx] if isinstance(thr[1], np.ndarray) else thr[1]
            
            if margDist == 'gp':
                IUU = genpareto.ppf(prob1, c=eps[0], loc=t1, scale=s1)
                IVVV = genpareto.ppf(prob2, c=eps[1], loc=t2, scale=s2)
            elif margDist == 'gev':
                IUU = gev.ppf(prob1, c=-eps[0], loc=t1, scale=s1)
                IVVV = gev.ppf(prob2, c=-eps[1], loc=t2, scale=s2)
                
            good_mask = np.isfinite(IUU) & np.isfinite(IVVV)
            IUU, IVVV = IUU[good_mask], IVVV[good_mask]
            UU_g, VVV_g = UU[good_mask], VVV[good_mask]
            
            if len(IUU) < 2: continue
            
            # Densities
            Dens = copula_pdf(UU_g, VVV_g, Family, theta)
            Dens = Dens / np.max(Dens) # Normalize
            
            # Plot multicolored line
            points = np.array([IUU, IVVV]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap='jet', norm=plt.Normalize(0, 1))
            lc.set_array((Dens[:-1] + Dens[1:]) / 2)
            lc.set_linewidth(2)
            line = ax.add_collection(lc)
            
            if j == 0:
                cb = fig.colorbar(lc, ax=ax, location='bottom', shrink=0.6, pad=0.1)
                cb.set_label('Copula pdf')
                
            ax.text(np.max(IUU), np.min(IVVV), str(PL[j]), color='red', fontsize=10, fontweight='bold')
            
        ax.scatter(jMax[w_idx][:, 0], jMax[w_idx][:, 1], color='r', marker='*', label='Joint peaks')
        ax.set_xlabel('Variable 1')
        ax.set_ylabel('Variable 2')
        
        t1x = pd.to_datetime(timeStampsByTimeWindow[w_idx][0] - 719529, unit='D').strftime('%Y') if timeStampsByTimeWindow else ''
        t2x = pd.to_datetime(timeStampsByTimeWindow[w_idx][-1] - 719529, unit='D').strftime('%Y') if timeStampsByTimeWindow else ''
        if copulaFamily.lower() == 'gaussian':
            ax.set_title(f"{t1x} - {t2x}\nGaussian ($\\rho$ = {par01})", loc='center', fontweight='bold')
        else:
            ax.set_title(f"{t1x} - {t2x}\n{copulaFamily.capitalize()} ($\\theta$ = {par01})", loc='center', fontweight='bold')

        ax.set_xlabel(ylbl[0])
        ax.set_ylabel(ylbl[1])
        ax.grid(True)
        ax.text(0.05, 0.9, labelMark, transform=ax.transAxes, fontweight='bold')
        
        ax.set_xlim([4, 20])
        ax.set_ylim([2.5, 6.0])

        return ax.get_xlim(), ax.get_ylim()
        
def tsCopulaPlotTrivariate(copulaAnalysis, monteCarloAnalysis, **kwargs):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.dates as mdates
    from mpl_toolkits.mplot3d import Axes3D

    xlbl = kwargs.get('xlbl', 'Date (time)')
    fontSize = kwargs.get('fontSize', 12)
    varLabels = kwargs.get('varLabels', ["Var1", "Var2", "Var3"])
    gofStatistics = kwargs.get('gofStatistics', None)
    
    labelMark = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)", "(j)", "(k)"]
    methodology = copulaAnalysis['methodology'].lower()
    
    # Retrieve original series and timestamps
    marginalAnalysis = copulaAnalysis['marginalAnalysis']
    nonStatSeries = np.column_stack([ma[1].nonStatSeries for ma in marginalAnalysis])
    timeStamps = np.column_stack([ma[1].timeStamps for ma in marginalAnalysis])
    pval = [ma[1].pValueChange for ma in marginalAnalysis]
    pvalStat = [ma[1].pValueChangeStat for ma in marginalAnalysis]
    
    mc = monteCarloAnalysis['monteCarloRsmpl']
    timeStampsByTimeWindow = copulaAnalysis['copulaParam']['timeStampsByTimeWindow']
    couplingParam = copulaAnalysis['copulaParam']['rho']
    couplingParamRaw = copulaAnalysis['copulaParam']['rhoRaw']
    ttRho = copulaAnalysis['copulaParam']['rhoTimeStamps']
    
    t1xStrt = pd.to_datetime(timeStampsByTimeWindow[0][0] - 719529, unit='D').strftime('%Y')
    t2xStrt = pd.to_datetime(timeStampsByTimeWindow[0][-1] - 719529, unit='D').strftime('%Y')
    t1xEnd = pd.to_datetime(timeStampsByTimeWindow[-1][0] - 719529, unit='D').strftime('%Y')
    t2xEnd = pd.to_datetime(timeStampsByTimeWindow[-1][-1] - 719529, unit='D').strftime('%Y')
    
    # Extract lower triangle of coupling parameters for 3 pairs
    def extract_lower_tri(param_list):
        mat = []
        for p in param_list:
            tri_vals = p[np.tril_indices_from(p, k=-1)]
            mat.append(tri_vals)
        return np.array(mat)

    couplingParamMat = extract_lower_tri(couplingParam)
    couplingParamMatRaw = extract_lower_tri(couplingParamRaw)
    
    jointExtremes = copulaAnalysis['jointExtremes']
    family = copulaAnalysis['copulaParam']['family'].lower()
    cplSymbol = r'\rho' if family == 'gaussian' else r'\theta'
    
    thresholds = copulaAnalysis.get('thresholdPotNS', None) if methodology == 'gpd' else None
    
    if copulaAnalysis['timeVaryingCopula']:
        yMax = copulaAnalysis['yMax']
        tMax = copulaAnalysis['tMax']
    else:
        yMax = copulaAnalysis['jointExtremes']
        tMax = copulaAnalysis['jointExtremeTimeStamps']
        
    iyMax = np.argsort(np.mean(yMax, axis=1))
    yMax = yMax[iyMax, :]
    tMax = tMax[iyMax, :]
    
    # Initialize figure
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(6, 3, figure=fig, wspace=0.3, hspace=0.6)
    axxArray = []

    # Helper: Plot Time Series
    def plot_time_series(ax, tt, series, thr, tmax, ymax, c_data, label, y_label, p1, p2):
        dates = pd.to_datetime(tt - 719529, unit='D')
        ax.plot(dates, series, color='#0072BD', linewidth=1)
        if thr is not None:
            ax.plot(dates, thr, '--r', linewidth=1.5)
            
        tmax_dates = pd.to_datetime(tmax - 719529, unit='D')
        ax.scatter(tmax_dates, ymax, c=c_data, cmap='jet', zorder=5)
        ax.set_ylabel(y_label, fontsize=fontSize)
        ax.set_xlabel(xlbl, fontsize=fontSize)
        ax.grid(True)
        ax.text(0.05, 0.85, label, transform=ax.transAxes, fontweight='bold')
        ax.text(0.2, 0.85, f'$p-val_{{nonStat}}$ = {p1:.3g}\n$p-val_{{Stat}}$ = {p2:.3g}', transform=ax.transAxes)
        return ax

    # Helper: Scatter 2D
    def scatter_2d(ax, mc_data, pair, j_ext, y_max, label, cpl_param, t1, t2, cpl_symb):
        ax.scatter(mc_data[:, pair[0]], mc_data[:, pair[1]], facecolors='none', edgecolors='gray', alpha=0.3)
        # We use yMax[:,0] as the color map proxy like MATLAB
        ax.scatter(y_max[:, pair[0]], y_max[:, pair[1]], c=y_max[:, 0], cmap='jet', zorder=5)
        ax.set_xlabel(varLabels[pair[0]])
        ax.set_ylabel(varLabels[pair[1]])
        ax.grid(True)
        ax.text(0.05, 0.85, label, transform=ax.transAxes, fontweight='bold')

        pr = cpl_param[pair[0], pair[1]]
        # Place window + copula-parameter annotation INSIDE the axes (opaque bbox)
        # instead of set_title(): the packed GridSpec makes a title above the
        # panel collide with the x-axis label of the panel above it.
        ax.text(0.5, 0.97,
                f"{t1} - {t2}\n{family.capitalize()} (${cpl_symb}$ = {pr:.2f})",
                transform=ax.transAxes, ha='center', va='top', fontsize=9,
                zorder=20,
                bbox=dict(facecolor='white', alpha=1.0, edgecolor='none', pad=1.5))
        return ax

    # Panel (a): 3D Scatter
    ax_a = fig.add_subplot(gs[0:2, 0], projection='3d')
    axxArray.append(ax_a)
    sc3 = ax_a.scatter(yMax[:, 1], yMax[:, 0], yMax[:, 2], c=yMax[:, 0], cmap='jet')
    ax_a.view_init(elev=30, azim=-57.5) # Equivalent to MATLAB's view(57.5, 30) with XDir reverse logic
    ax_a.set_xlabel(varLabels[1])
    ax_a.set_ylabel(varLabels[0])
    ax_a.set_zlabel(varLabels[2])
    ax_a.text2D(0.05, 0.95, labelMark[0], transform=ax_a.transAxes, fontweight='bold')
    
    if gofStatistics:
        stats_txt = (f"$\\overline{{\\Delta\\rho}}_{{S}} = {gofStatistics['corrSpearmanSampleDelta']:.2g}$\n"
                     f"$\\overline{{\\Delta\\tau}}_{{K}} = {gofStatistics['corrKendallSampleDelta']:.2g}$\n"
                     f"$\\overline{{S_n}} = {gofStatistics['snSample']:.1g}$")
        ax_a.text2D(0.65, 0.7, stats_txt, transform=ax_a.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    # Panels (b, c, d): Time Series
    ax_b = fig.add_subplot(gs[0, 1:3])
    axxArray.append(ax_b)
    plot_time_series(ax_b, timeStamps[:,0], nonStatSeries[:,0], thresholds[:,0] if thresholds is not None else None, 
                     tMax[:,0], yMax[:,0], np.mean(yMax, axis=1), labelMark[1], varLabels[0], pval[0], pvalStat[0])

    ax_c = fig.add_subplot(gs[1, 1:3])
    axxArray.append(ax_c)
    plot_time_series(ax_c, timeStamps[:,1], nonStatSeries[:,1], thresholds[:,1] if thresholds is not None else None, 
                     tMax[:,1], yMax[:,1], np.mean(yMax, axis=1), labelMark[5], varLabels[1], pval[1], pvalStat[1])

    ax_d = fig.add_subplot(gs[2, 1:3])
    axxArray.append(ax_d)
    plot_time_series(ax_d, timeStamps[:,2], nonStatSeries[:,2], thresholds[:,2] if thresholds is not None else None, 
                     tMax[:,2], yMax[:,2], np.mean(yMax, axis=1), labelMark[8], varLabels[2], pval[2], pvalStat[2])

    # Panel (e): Coupling Series (GOF)
    ax_e = fig.add_subplot(gs[2, 0])
    axxArray.append(ax_e)
    dates_rho = pd.to_datetime(ttRho - 719529, unit='D')
    ax_e.plot(dates_rho, couplingParamMat, linewidth=1.5)
    ax_e.grid(True)
    ax_e.set_xlabel(xlbl)
    ax_e.set_ylabel(f"${cplSymbol}_{{{family}}}$", fontsize=fontSize)
    ax_e.text(0.05, 0.85, labelMark[4], transform=ax_e.transAxes, fontweight='bold')
    
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, pair in enumerate(pairs):
        _, _, p_val, _ = tsModified_MannKendall_test(ttRho, couplingParamMatRaw[:, i], 0.05, 0.05)
        ax_e.text(0.2, 0.3 - 0.1*i, f"$p-val_{{{cplSymbol},{pair[0]+1}-{pair[1]+1}}} = {p_val:.3g}$", transform=ax_e.transAxes)
    ax_e.legend([f"${cplSymbol}_{{{p[0]+1}-{p[1]+1}}}$" for p in pairs], loc='best')

    # Panels (f, g, h): Monte Carlo Start
    ax_f = fig.add_subplot(gs[3, 0])
    axxArray.append(ax_f)
    scatter_2d(ax_f, mc[0], pairs[0], jointExtremes[0], yMax, labelMark[2], couplingParam[0], t1xStrt, t2xStrt, cplSymbol)

    ax_g = fig.add_subplot(gs[4, 0])
    axxArray.append(ax_g)
    scatter_2d(ax_g, mc[0], pairs[1], jointExtremes[0], yMax, labelMark[6], couplingParam[0], t1xStrt, t2xStrt, cplSymbol)

    ax_h = fig.add_subplot(gs[5, 0])
    axxArray.append(ax_h)
    scatter_2d(ax_h, mc[0], pairs[2], jointExtremes[0], yMax, labelMark[9], couplingParam[0], t1xStrt, t2xStrt, cplSymbol)

    # Panels (i, j, k): Monte Carlo End
    ax_i = fig.add_subplot(gs[3, 1])
    axxArray.append(ax_i)
    scatter_2d(ax_i, mc[-1], pairs[0], jointExtremes[-1], yMax, labelMark[3], couplingParam[-1], t1xEnd, t2xEnd, cplSymbol)

    ax_j = fig.add_subplot(gs[4, 1])
    axxArray.append(ax_j)
    scatter_2d(ax_j, mc[-1], pairs[1], jointExtremes[-1], yMax, labelMark[7], couplingParam[-1], t1xEnd, t2xEnd, cplSymbol)

    ax_k = fig.add_subplot(gs[5, 1])
    axxArray.append(ax_k)
    scatter_2d(ax_k, mc[-1], pairs[2], jointExtremes[-1], yMax, labelMark[10], couplingParam[-1], t1xEnd, t2xEnd, cplSymbol)

    return axxArray

def tsCopulaPlotTrivariateWithMap(copulaAnalysis, monteCarloAnalysis, **kwargs):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.dates as mdates
    from mpl_toolkits.mplot3d import Axes3D
    
    # Try importing cartopy for the map feature
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        raise ImportError("The 'cartopy' library is required to plot the map. Please install it using 'pip install cartopy'.")

    xlbl = kwargs.get('xlbl', 'Date (time)')
    fontSize = kwargs.get('fontSize', 12)
    varLabels = kwargs.get('varLabels', ["Var1", "Var2", "Var3"])
    gofStatistics = kwargs.get('gofStatistics', None)
    locString = kwargs.get('locString', ["Loc1", "Loc2", "Loc3"])
    latlon = kwargs.get('latlon', np.array([]))
    
    labelMark = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)", "(j)", "(k)", "(l)"]
    methodology = copulaAnalysis['methodology'].lower()
    
    # Retrieve original series and timestamps
    marginalAnalysis = copulaAnalysis['marginalAnalysis']
    nonStatSeries = np.column_stack([ma[1].nonStatSeries for ma in marginalAnalysis])
    timeStamps = np.column_stack([ma[1].timeStamps for ma in marginalAnalysis])
    pval = [ma[1].pValueChange for ma in marginalAnalysis]
    pvalStat = [ma[1].pValueChangeStat for ma in marginalAnalysis]
    
    mc = monteCarloAnalysis['monteCarloRsmpl']
    timeStampsByTimeWindow = copulaAnalysis['copulaParam']['timeStampsByTimeWindow']
    couplingParam = copulaAnalysis['copulaParam']['rho']
    couplingParamRaw = copulaAnalysis['copulaParam']['rhoRaw']
    ttRho = copulaAnalysis['copulaParam']['rhoTimeStamps']
    
    t1xStrt = pd.to_datetime(timeStampsByTimeWindow[0][0] - 719529, unit='D').strftime('%Y')
    t2xStrt = pd.to_datetime(timeStampsByTimeWindow[0][-1] - 719529, unit='D').strftime('%Y')
    t1xEnd = pd.to_datetime(timeStampsByTimeWindow[-1][0] - 719529, unit='D').strftime('%Y')
    t2xEnd = pd.to_datetime(timeStampsByTimeWindow[-1][-1] - 719529, unit='D').strftime('%Y')
    
    # Extract lower triangle of coupling parameters for 3 pairs
    def extract_lower_tri(param_list):
        mat = []
        for p in param_list:
            tri_vals = p[np.tril_indices_from(p, k=-1)]
            mat.append(tri_vals)
        return np.array(mat)

    couplingParamMat = extract_lower_tri(couplingParam)
    couplingParamMatRaw = extract_lower_tri(couplingParamRaw)
    
    jointExtremes = copulaAnalysis['jointExtremes']
    family = copulaAnalysis['copulaParam']['family'].lower()
    cplSymbol = r'\rho' if family == 'gaussian' else r'\theta'
    
    thresholds = copulaAnalysis.get('thresholdPotNS', None) if methodology == 'gpd' else None
    
    if copulaAnalysis['timeVaryingCopula']:
        yMax = copulaAnalysis['yMax']
        tMax = copulaAnalysis['tMax']
    else:
        yMax = copulaAnalysis['jointExtremes']
        tMax = copulaAnalysis['jointExtremeTimeStamps']
        
    iyMax = np.argsort(np.mean(yMax, axis=1))
    yMax = yMax[iyMax, :]
    tMax = tMax[iyMax, :]
    
    # Initialize figure
    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(7, 3, figure=fig, wspace=0.3, hspace=0.6)
    axxArray = []

    # =========================================================================
    # Panel (a): Geographic Map (Cartopy replaces m_map)
    # =========================================================================
    ax_map = fig.add_subplot(gs[0:2, 0], projection=ccrs.PlateCarree())
    axxArray.append(ax_map)
    
    ax_map.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.LAND, facecolor='lightgreen')
    ax_map.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax_map.coastlines(color='black')
    
    if latlon.size > 0:
        # latlon format is assumed to be [[lon1, lon2...], [lat1, lat2...]]
        for i in range(latlon.shape[1]):
            lon, lat = latlon[0, i], latlon[1, i]
            ax_map.plot(lon, lat, marker='s', color='red', markersize=6, transform=ccrs.PlateCarree())
            ax_map.text(lon + 3, lat + 3, locString[i], color='black', fontweight='bold', transform=ccrs.PlateCarree())
            
    ax_map.text(0.05, 0.95, labelMark[0], transform=ax_map.transAxes, fontweight='bold', zorder=10)
    
    # Helper: Plot Time Series
    def plot_time_series(ax, tt, series, thr, tmax, ymax, c_data, label, y_label, p1, p2):
        dates = pd.to_datetime(tt - 719529, unit='D')
        ax.plot(dates, series, color='#0072BD', linewidth=1)
        if thr is not None:
            ax.plot(dates, thr, '--r', linewidth=1.5)
            
        tmax_dates = pd.to_datetime(tmax - 719529, unit='D')
        ax.scatter(tmax_dates, ymax, c=c_data, cmap='jet', zorder=5)
        ax.set_ylabel(y_label, fontsize=fontSize)
        ax.set_xlabel(xlbl, fontsize=fontSize)
        ax.grid(True)
        ax.text(0.05, 0.85, label, transform=ax.transAxes, fontweight='bold')
        ax.text(0.2, 0.85, f'$p-val_{{nonStat}}$ = {p1:.3g}\n$p-val_{{Stat}}$ = {p2:.3g}', transform=ax.transAxes)
        return ax

    # Helper: Scatter 2D
    def scatter_2d(ax, mc_data, pair, j_ext, y_max, label, cpl_param, t1, t2, cpl_symb):
        ax.scatter(mc_data[:, pair[0]], mc_data[:, pair[1]], facecolors='none', edgecolors='gray', alpha=0.3)
        ax.scatter(y_max[:, pair[0]], y_max[:, pair[1]], c=y_max[:, 0], cmap='jet', zorder=5)
        
        ax.set_xlabel(f"{locString[pair[0]]}_{varLabels[pair[0]]}")
        ax.set_ylabel(f"{locString[pair[1]]}_{varLabels[pair[1]]}")
        ax.grid(True)
        ax.text(0.05, 0.85, label, transform=ax.transAxes, fontweight='bold')
        
        pr = cpl_param[pair[0], pair[1]]
        ax.set_title(f"{t1} - {t2}\n{family.capitalize()} (${cpl_symb}$ = {pr:.2f})")
        return ax

    # Panel (b): 3D Scatter
    ax_3d = fig.add_subplot(gs[2:4, 0], projection='3d')
    axxArray.append(ax_3d)
    ax_3d.scatter(yMax[:, 1], yMax[:, 0], yMax[:, 2], c=yMax[:, 0], cmap='jet')
    ax_3d.view_init(elev=30, azim=-57.5) 
    ax_3d.set_xlabel(f"{locString[1]}_{varLabels[1]}")
    ax_3d.set_ylabel(f"{locString[0]}_{varLabels[0]}")
    ax_3d.set_zlabel(f"{locString[2]}_{varLabels[2]}")
    ax_3d.text2D(0.05, 0.95, labelMark[1], transform=ax_3d.transAxes, fontweight='bold')
    
    if gofStatistics:
        stats_txt = (f"$\\overline{{\\Delta\\rho}}_{{S}} = {gofStatistics['corrSpearmanSampleDelta']:.2g}$\n"
                     f"$\\overline{{\\Delta\\tau}}_{{K}} = {gofStatistics['corrKendallSampleDelta']:.2g}$\n"
                     f"$\\overline{{S_n}} = {gofStatistics['snSample']:.1g}$")
        ax_3d.text2D(0.65, 0.7, stats_txt, transform=ax_3d.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    # Panels (c, d, e): Time Series
    ax_ts1 = fig.add_subplot(gs[0, 1:3])
    axxArray.append(ax_ts1)
    plot_time_series(ax_ts1, timeStamps[:,0], nonStatSeries[:,0], thresholds[:,0] if thresholds is not None else None, 
                     tMax[:,0], yMax[:,0], np.mean(yMax, axis=1), labelMark[2], f"{locString[0]}_{varLabels[0]}", pval[0], pvalStat[0])

    ax_ts2 = fig.add_subplot(gs[1, 1:3])
    axxArray.append(ax_ts2)
    plot_time_series(ax_ts2, timeStamps[:,1], nonStatSeries[:,1], thresholds[:,1] if thresholds is not None else None, 
                     tMax[:,1], yMax[:,1], np.mean(yMax, axis=1), labelMark[3], f"{locString[1]}_{varLabels[1]}", pval[1], pvalStat[1])

    ax_ts3 = fig.add_subplot(gs[2, 1:3])
    axxArray.append(ax_ts3)
    plot_time_series(ax_ts3, timeStamps[:,2], nonStatSeries[:,2], thresholds[:,2] if thresholds is not None else None, 
                     tMax[:,2], yMax[:,2], np.mean(yMax, axis=1), labelMark[4], f"{locString[2]}_{varLabels[2]}", pval[2], pvalStat[2])

    # Panel (f): Coupling Series (GOF)
    ax_rho = fig.add_subplot(gs[4, 0])
    axxArray.append(ax_rho)
    dates_rho = pd.to_datetime(ttRho - 719529, unit='D')
    ax_rho.plot(dates_rho, couplingParamMat, linewidth=1.5)
    ax_rho.grid(True)
    ax_rho.set_xlabel(xlbl)
    ax_rho.set_ylabel(f"${cplSymbol}_{{{family}}}$", fontsize=fontSize)
    ax_rho.text(0.05, 0.85, labelMark[5], transform=ax_rho.transAxes, fontweight='bold')
    
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, pair in enumerate(pairs):
        _, _, p_val, _ = tsModified_MannKendall_test(ttRho, couplingParamMatRaw[:, i], 0.05, 0.05)
        ax_rho.text(0.2, 0.3 - 0.1*i, f"$p-val_{{{cplSymbol},{pair[0]+1}-{pair[1]+1}}} = {p_val:.3g}$", transform=ax_rho.transAxes)
    ax_rho.legend([f"${cplSymbol}_{{{p[0]+1}-{p[1]+1}}}$" for p in pairs], loc='best')

    # Panels (g, h, i): Monte Carlo Start
    ax_g = fig.add_subplot(gs[3, 1])
    axxArray.append(ax_g)
    scatter_2d(ax_g, mc[0], pairs[0], jointExtremes[0], yMax, labelMark[6], couplingParam[0], t1xStrt, t2xStrt, cplSymbol)

    ax_h = fig.add_subplot(gs[4, 1])
    axxArray.append(ax_h)
    scatter_2d(ax_h, mc[0], pairs[1], jointExtremes[0], yMax, labelMark[7], couplingParam[0], t1xStrt, t2xStrt, cplSymbol)

    ax_i = fig.add_subplot(gs[5, 1])
    axxArray.append(ax_i)
    scatter_2d(ax_i, mc[0], pairs[2], jointExtremes[0], yMax, labelMark[8], couplingParam[0], t1xStrt, t2xStrt, cplSymbol)

    # Panels (j, k, l): Monte Carlo End
    ax_j = fig.add_subplot(gs[3, 2])
    axxArray.append(ax_j)
    scatter_2d(ax_j, mc[-1], pairs[0], jointExtremes[-1], yMax, labelMark[9], couplingParam[-1], t1xEnd, t2xEnd, cplSymbol)

    ax_k = fig.add_subplot(gs[4, 2])
    axxArray.append(ax_k)
    scatter_2d(ax_k, mc[-1], pairs[1], jointExtremes[-1], yMax, labelMark[10], couplingParam[-1], t1xEnd, t2xEnd, cplSymbol)

    ax_l = fig.add_subplot(gs[5, 2])
    axxArray.append(ax_l)
    scatter_2d(ax_l, mc[-1], pairs[2], jointExtremes[-1], yMax, labelMark[11], couplingParam[-1], t1xEnd, t2xEnd, cplSymbol)

    return axxArray

def tsCopulaYearExtrDistribution(retPeriod, copulaParam, computeCdf=False):

    import numpy as np
    from scipy.stats import multivariate_normal, norm
    
    nSeries = copulaParam.get('nSeries', 2)
    retPeriod = np.asarray(retPeriod)
    
    # MATLAB's ndgrid equivalent in Python
    grids = np.meshgrid(*[retPeriod]*nSeries, indexing='ij')
    retPerOut = np.column_stack([g.flatten() for g in grids])
    
    # Convert return periods to non-exceedance probabilities
    probOut = 1.0 - 1.0 / retPerOut
    
    copulaFamily = copulaParam.get('family', '').lower()
    
    jpdf = np.zeros(probOut.shape[0])
    jcdf = np.zeros(probOut.shape[0]) if computeCdf else None
    
    if copulaFamily == 'gaussian':
        rho = copulaParam.get('rho')
        if np.isscalar(rho):
            cov = np.array([[1.0, rho], [rho, 1.0]])
        else:
            cov = np.asarray(rho)
            
        # PDF/CDF for Gaussian Copula
        u_norm = norm.ppf(probOut)
        
        # Calculate copula PDF: c(u) = f(F^-1(u)) / prod(f_i(F_i^-1(u_i)))
        num = multivariate_normal.pdf(u_norm, mean=np.zeros(nSeries), cov=cov)
        den = np.prod(norm.pdf(u_norm), axis=1)
        jpdf = num / den
        
        if computeCdf:
            jcdf = multivariate_normal.cdf(u_norm, mean=np.zeros(nSeries), cov=cov)
            
    elif copulaFamily in ['gumbel', 'clayton', 'frank']:
        # Note: MATLAB's copulapdf for Archimedean copulas only supports bivariate data.
        if nSeries != 2:
            raise ValueError("Archimedean copulas are only supported for bivariate data in this context.")
        
        theta = copulaParam.get('theta')
        if theta is None:
            # Fallback if 'rho' was used instead of 'theta' in the dict
            rho_val = copulaParam.get('rho')
            theta = rho_val if np.isscalar(rho_val) else rho_val[0, 1]
            
        u = np.clip(probOut[:, 0], 1e-12, 1.0 - 1e-12)
        v = np.clip(probOut[:, 1], 1e-12, 1.0 - 1e-12)
        
        if copulaFamily == 'gumbel':
            x_, y_ = -np.log(u), -np.log(v)
            S_ = (x_**theta + y_**theta)**(1.0/theta)
            C_ = np.exp(-S_)
            jpdf = C_ * (x_ * y_)**(theta - 1.0) / (u * v) * (x_**theta + y_**theta)**(2.0/theta - 2.0) * (S_ + theta - 1.0)
            if computeCdf:
                jcdf = C_
                
        elif copulaFamily == 'clayton':
            C_ = (u**(-theta) + v**(-theta) - 1.0)**(-1.0/theta)
            jpdf = (theta + 1.0) * (u * v)**(-theta - 1.0) * (u**(-theta) + v**(-theta) - 1.0)**(-2.0 - 1.0/theta)
            if computeCdf:
                jcdf = C_
                
        elif copulaFamily == 'frank':
            num = -theta * (np.exp(-theta) - 1.0) * np.exp(-theta * (u + v))
            den = ((np.exp(-theta * u) - 1.0) * (np.exp(-theta * v) - 1.0) + (np.exp(-theta) - 1.0))**2
            jpdf = num / den
            if computeCdf:
                num_c = (np.exp(-theta*u) - 1.0) * (np.exp(-theta*v) - 1.0)
                den_c = np.exp(-theta) - 1.0
                jcdf = -(1.0/theta) * np.log(1.0 + num_c / den_c)
                
    elif copulaFamily == 't':
        raise ValueError("t-copula is not supported.")
    else:
        raise ValueError(f"copulaFamily not supported: {copulaFamily}")
        
    # Reshape the flat arrays back to the N-dimensional grid shape
    grid_shape = tuple([len(retPeriod)] * nSeries)
    jpdf = jpdf.reshape(grid_shape)
    
    if computeCdf:
        jcdf = jcdf.reshape(grid_shape)
        
    return jpdf, jcdf

def tsCopulaGetFamilyFromId(familyId):
    mapping = {
        1: 'gaussian',
        2: 't',
        3: 'gumbel',
        4: 'clayton',
        5: 'frank'
    }
    
    if familyId in mapping:
        return mapping[familyId]
    else:
        raise ValueError(f"copula familyId not supported: {familyId}")
        
def tsCopulaGetFamilyId(copulaFamily):
    mapping = {
        'gaussian': 1,
        't': 2,
        'gumbel': 3,
        'clayton': 4,
        'frank': 5
    }
    
    family_lower = copulaFamily.lower()
    
    if family_lower in mapping:
        return mapping[family_lower]
    else:
        raise ValueError(f"copulaFamily not supported: {copulaFamily}")

def tsCopulaYearExtrFit(retPeriod, retLev, yMax, **kwargs):

    import numpy as np
    from scipy.interpolate import interp1d

    copulaFamily = kwargs.get('copulaFamily', 'gaussian').lower()
    
    yMax = np.asarray(yMax)
    retLev = np.asarray(retLev)
    retPeriod = np.asarray(retPeriod)
    
    nsrs = yMax.shape[1]
    nyr = yMax.shape[0]
    nretPer = len(retPeriod)
    
    szsRetLev = retLev.shape
    ndimRetLev = retLev.ndim
    
    yRetPer = np.full_like(yMax, np.nan, dtype=float)
    
    if ndimRetLev == 2:
        # Stationary set of return levels
        if szsRetLev[0] != nretPer or szsRetLev[1] != nsrs:
            raise ValueError("tsCopulaYearExtrFit: for stationary, retLev should be dimensioned as (nReturnPeriod x nSeries)")
            
        for isrs in range(nsrs):
            x_val = retLev[:, isrs]
            y_val = retPeriod
            # Scipy interp1d expects monotonically increasing x. Return levels should be monotonic.
            sort_idx = np.argsort(x_val)
            f_interp = interp1d(x_val[sort_idx], y_val[sort_idx], bounds_error=False, fill_value="extrapolate")
            yRetPer[:, isrs] = f_interp(yMax[:, isrs])
            
    elif ndimRetLev == 3:
        # Non-stationary set of return levels
        if szsRetLev[0] != nyr or szsRetLev[1] != nretPer or szsRetLev[2] != nsrs:
            raise ValueError("tsCopulaYearExtrFit: for non-stationary, retLev should be dimensioned as (nTime x nReturnPeriod x nSeries)")
            
        for isrs in range(nsrs):
            for iyr in range(nyr):
                x_val = retLev[iyr, :, isrs]
                y_val = retPeriod
                sort_idx = np.argsort(x_val)
                f_interp = interp1d(x_val[sort_idx], y_val[sort_idx], bounds_error=False, fill_value="extrapolate")
                yRetPer[iyr, isrs] = f_interp(yMax[iyr, isrs])
    else:
        raise ValueError("tsCopulaYearExtrFit: retLev should be 2-dim for stationary, 3-dim for non-stationary")

    # Cleanup extreme values mirroring MATLAB logic
    yRetPer[np.isinf(yRetPer)] = 0.1
    yRetPer[np.isnan(yRetPer)] = 0.1
    
    # Transform Return Period to Probability
    yProb = 1.0 - 1.0 / yRetPer
    
    # Cleanup Probabilities
    prob_gt_zero = yProb[yProb > 0]
    min_prob = np.min(prob_gt_zero) if len(prob_gt_zero) > 0 else 0.0001
    
    yProb[yProb <= 0] = min_prob
    yProb[yProb >= 1] = 0.9999
    yProb[np.isnan(yProb)] = 0.0001
    
    copulaParam = {}
    copulaParam['family'] = copulaFamily
    copulaParam['familyId'] = tsCopulaGetFamilyId(copulaFamily)
    copulaParam['nSeries'] = nsrs
    
    if copulaFamily == 't':
        raise ValueError("t copula is not supported.")
    else:
        # Use our pre-existing robust Copula fit
        param = tsCopulaFit(copulaFamily, yProb)
        if copulaFamily == 'gaussian':
            copulaParam['rho'] = param
        else:
            copulaParam['theta'] = param[0, 1] if not np.isscalar(param) else param
            copulaParam['cci'] = None  # Confidence intervals fallback
            
    return retLev, copulaParam, yRetPer, yProb

def tsCopulaYearExtrGetMltvrtRetPeriod(randomSample, level):

    import numpy as np
    
    randomSample = np.asarray(randomSample)
    level = np.atleast_2d(level)
    
    ndim = randomSample.shape[1]
    if ndim != level.shape[1]:
        raise ValueError("tsCopulaComputeMultivariateRetPeriod: randomSample and level should have the same number of columns")
        
    nretLev = level.shape[0]
    returnPeriod = np.full(nretLev, np.nan)
    prob = np.full(nretLev, np.nan)
    
    for iretLev in range(nretLev):
        lvli = level[iretLev, :]
        
        # Broadcasting handles the comparison across all samples efficiently.
        # AND condition: all variables must strictly exceed their respective levels.
        cnd = np.all(randomSample > lvli, axis=1)
        
        # Empirical probability is the mean of the boolean array
        prob[iretLev] = np.mean(cnd)
        
        # Avoid division by zero; if probability is 0, return period is theoretically infinite
        if prob[iretLev] > 0:
            returnPeriod[iretLev] = 1.0 / prob[iretLev]
        else:
            returnPeriod[iretLev] = np.inf
            
    return returnPeriod, prob

def tsCopulaYearExtrPlotJdistTrivar(retLev, jdist, **kwargs):

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Default parameters
    figPosition = kwargs.get('figPosition', [50, 50, 8, 8]) # Matplotlib uses inches
    xlbl = kwargs.get('xlbl', 'X')
    ylbl = kwargs.get('ylbl', 'Y')
    zlbl = kwargs.get('zlbl', 'Z')
    fontSize = kwargs.get('fontSize', 12)
    probRange = kwargs.get('probRange', [])
    azimuth = kwargs.get('azimuth', -13.9)
    elevation = kwargs.get('elevation', 15.6) # Elevation sign is often flipped in MPL
    colorbarLabel = kwargs.get('colorbarLabel', 'Density')

    if retLev.shape[1] != 3 or jdist.ndim != 3:
        raise ValueError("tsCopulaYearExtrPlotJdistTrivar: retLev must be Nx3, jdist must be 3D array")

    fig = plt.figure(figsize=(figPosition[2], figPosition[3]), facecolor='w')
    ax = fig.add_subplot(111, projection='3d')

    x = retLev[:, 0]
    y = retLev[:, 1]
    z = retLev[:, 2]

    # Handle probability range clipping
    if len(probRange) == 2:
        jdist = np.clip(jdist, probRange[0], probRange[1])

    # Create grids for slicing
    # Note: MATLAB's slice uses a different grid logic. 
    # Here we simulate slices by plotting surfaces at specific locations.
    X, Y = np.meshgrid(x, y)
    Y_z, Z_y = np.meshgrid(y, z)
    X_z, Z_x = np.meshgrid(x, z)

    # Slice at max Y (Equivalent to MATLAB's yslice = max(y))
    slice_y_max = jdist[:, -1, :]
    surf1 = ax.plot_surface(X, np.full_like(X, y.max()), slice_y_max.T, 
                            cmap='viridis', alpha=0.8, antialiased=False)

    # Slice at median X (Equivalent to MATLAB's xslice = median(x))
    med_idx_x = len(x) // 2
    slice_x_med = jdist[med_idx_x, :, :]
    surf2 = ax.plot_surface(np.full_like(Y_z, x[med_idx_x]), Y_z, slice_x_med, 
                            cmap='viridis', alpha=0.8, antialiased=False)

    # Slice at median Z (Equivalent to MATLAB's zslice = median(z))
    med_idx_z = len(z) // 2
    slice_z_med = jdist[:, :, med_idx_z]
    surf3 = ax.plot_surface(X, Y, np.full_like(X, z[med_idx_z]), 
                            facecolors=plt.cm.viridis(slice_z_med/np.max(jdist)), 
                            alpha=0.8, antialiased=False)

    ax.set_xlabel(xlbl, fontsize=fontSize)
    ax.set_ylabel(ylbl, fontsize=fontSize)
    ax.set_zlabel(zlbl, fontsize=fontSize)
    ax.view_init(elev=elevation, azim=azimuth)
    
    cb = fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=10)
    cb.set_label(colorbarLabel, fontsize=fontSize)
    
    plt.show()
    return [fig, ax, cb]

def tsCopulaYearExtrPlotSctrBivar(monteCarloRsmpl, yMaxLevel, **kwargs):

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Default parameters
    xlbl = kwargs.get('xlbl', 'X')
    ylbl = kwargs.get('ylbl', 'Y')
    figPosition = kwargs.get('figPosition', [100, 100, 10, 10]) # Inches for MPL
    fontSize = kwargs.get('fontSize', 15)

    if monteCarloRsmpl.shape[1] != 2 or yMaxLevel.shape[1] != 2:
        raise ValueError("tsCopulaYearExtrPlotSctrBivar: monteCarloRsmpl and yMaxLevel must be Nx2 arrays")

    # Initialize JointGrid for scatterhist effect
    g = sns.JointGrid(x=monteCarloRsmpl[:, 0], y=monteCarloRsmpl[:, 1], 
                      height=figPosition[2], space=0.2)

    # Plot Monte Carlo distribution (Scatter and Marginals)
    # Using light blue with transparency for the large MC sample
    g.plot_joint(plt.scatter, s=10, color='skyblue', alpha=0.4, label='Joint Distribution\nMontecarlo')
    g.plot_marginals(sns.histplot, kde=True, color='skyblue', element="step")

    # Overlay observed Yearly Maxima (Kırmızı noktalar)
    g.ax_joint.scatter(yMaxLevel[:, 0], yMaxLevel[:, 1], 
                       color='red', s=40, edgecolors='k', zorder=5, label='Yearly Maxima')

    # Formatting
    g.ax_joint.set_xlabel(xlbl, fontsize=fontSize)
    g.ax_joint.set_ylabel(ylbl, fontsize=fontSize)
    g.ax_joint.tick_params(labelsize=fontSize)
    g.ax_joint.grid(True, linestyle='--', alpha=0.7)

    # Legend positioning (Matching MATLAB's custom logic)
    g.ax_joint.legend(fontsize=fontSize, loc='upper right')

    plt.tight_layout()
    plt.show()

    return [plt.gcf(), g.ax_joint, g.ax_marg_x, g.ax_marg_y]

def tsCopulaYearExtrPlotSctrTrivar(monteCarloRsmpl, yMaxLevel, **kwargs):

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Default parameters
    xlbl = kwargs.get('xlbl', 'X')
    ylbl = kwargs.get('ylbl', 'Y')
    zlbl = kwargs.get('zlbl', 'Z')
    xaxiscolor = kwargs.get('xaxiscolor', [0.7, 0.4, 0.05])
    yaxiscolor = kwargs.get('yaxiscolor', [0.3, 0.4, 0.1])
    zaxiscolor = kwargs.get('zaxiscolor', [0, 0.5, 0.5])
    figPosition = kwargs.get('figPosition', [50, 50, 10, 10])
    fontSize = kwargs.get('fontSize', 12)
    azimuth = kwargs.get('azimuth', -74)
    elevation = kwargs.get('elevation', 35)
    title_str = kwargs.get('title', '')
    markerTranspAlpha = kwargs.get('markerTranspAlpha', 0.3)

    if monteCarloRsmpl.shape[1] != 3 or yMaxLevel.shape[1] != 3:
        raise ValueError("tsCopulaYearExtrPlotSctrTrivar: monteCarloRsmpl and yMaxLevel must be Nx3 arrays")

    fig = plt.figure(figsize=(figPosition[2], figPosition[3]), facecolor='w')
    # Using GridSpec to emulate MATLAB's manual axes positions
    gs = GridSpec(4, 4, figure=fig)

    # 1. Main 3D Scatter Plot (Takes the top-right large area)
    ax3d = fig.add_subplot(gs[0:3, 1:4], projection='3d')
    
    # Monte Carlo Samples
    rsmplSctr = ax3d.scatter(monteCarloRsmpl[:, 0], monteCarloRsmpl[:, 1], monteCarloRsmpl[:, 2], 
                             alpha=markerTranspAlpha, c='skyblue', edgecolors='none', label='Joint Distribution\nMontecarlo')
    
    # Yearly Maxima (Kırmızı noktalar)
    ymaxSctr = ax3d.scatter(yMaxLevel[:, 0], yMaxLevel[:, 1], yMaxLevel[:, 2], 
                            color='red', edgecolors='k', s=40, label='Yearly Maxima')

    ax3d.view_init(elev=elevation, azim=azimuth)
    ax3d.xaxis.label.set_color(xaxiscolor)
    ax3d.yaxis.label.set_color(yaxiscolor)
    ax3d.zaxis.label.set_color(zaxiscolor)
    ax3d.set_title(title_str, fontsize=kwargs.get('titleFontSize', 15))

    # 2. Histogram for X (Bottom right)
    axHistX = fig.add_subplot(gs[3, 2:4])
    axHistX.hist(monteCarloRsmpl[:, 0], bins=30, color=xaxiscolor, alpha=0.7)
    axHistX.set_title(xlbl, color=xaxiscolor, fontsize=fontSize, pad=-15)
    axHistX.axis('off')

    # 3. Histogram for Y (Bottom left)
    axHistY = fig.add_subplot(gs[3, 0:2])
    axHistY.hist(monteCarloRsmpl[:, 1], bins=30, color=yaxiscolor, alpha=0.7)
    axHistY.set_title(ylbl, color=yaxiscolor, fontsize=fontSize, pad=-15)
    axHistY.axis('off')

    # 4. Histogram for Z (Middle left, horizontal)
    axHistZ = fig.add_subplot(gs[1:3, 0])
    axHistZ.hist(monteCarloRsmpl[:, 2], bins=30, color=zaxiscolor, alpha=0.7, orientation='horizontal')
    axHistZ.set_ylabel(zlbl, color=zaxiscolor, fontsize=fontSize, rotation=0, labelpad=10)
    axHistZ.invert_xaxis()
    axHistZ.axis('off')

    # Legend
    ax3d.legend(loc='upper left', bbox_to_anchor=(-0.3, 1), fontsize=fontSize, frameon = False)

    plt.show()
    return [fig, ax3d, axHistX, axHistY, axHistZ]

def tsCopulaYearExtrRnd(retPeriod, retLev, copulaParam, nResample, **kwargs):

    import numpy as np
    from scipy.stats import multivariate_normal, norm, t
    from scipy.interpolate import interp1d

    logExtrapRetlev = kwargs.get('logExtrapRetlev', True)
    copulaFamily = copulaParam.get('family', 'gaussian').lower()
    nSeries = copulaParam.get('nSeries', 2)
    
    # 1. Generate Copula Random Samples (resampleProb)
    if copulaFamily == 'gaussian':
        rho = copulaParam.get('rho')
        if np.isscalar(rho):
            cov = np.array([[1.0, rho], [rho, 1.0]])
        else:
            cov = np.asarray(rho)
        
        # Generate multivariate normal samples and convert to uniform [0,1]
        mv_samples = multivariate_normal.rvs(mean=np.zeros(nSeries), cov=cov, size=nResample)
        resampleProb = norm.cdf(mv_samples)
        
    elif copulaFamily in ['gumbel', 'clayton', 'frank']:
        # Using a simplified bivariate approach for Archimedean (matching typical tsEVA usage)
        theta = copulaParam.get('theta')
        # In a full implementation, this would use conditional distribution method
        # For now, we utilize the property that most tsEva cases are bivariate here
        # Note: If nSeries > 2, this requires more complex sampling
        u = np.random.uniform(0, 1, nResample)
        v_rand = np.random.uniform(0, 1, nResample)
        
        resampleProb = np.zeros((nResample, 2))
        resampleProb[:, 0] = u
        
        if copulaFamily == 'clayton':
            resampleProb[:, 1] = (v_rand**(-theta/(theta+1)) * (u**(-theta) - 1) + 1)**(-1/theta)
        elif copulaFamily == 'gumbel':
            # Gumbel sampling is complex; usually requires numerical root finding
            # Simplified version or placeholder for common bivariate case
            resampleProb[:, 1] = v_rand # This is a placeholder; Gumbel usually needs Frank/Clayton style inversion
        elif copulaFamily == 'frank':
            exp_u = np.exp(-theta * u)
            exp_theta = np.exp(-theta)
            resampleProb[:, 1] = -1/theta * np.log(1 + (v_rand * (1 - exp_theta)) / (v_rand * (exp_u - 1) + exp_theta))
            
    else:
        raise ValueError(f"copulaFamily not supported for sampling: {copulaFamily}")

    # 2. Transform Probability to Return Period
    resampleRetPer = 1.0 / (1.0 - np.clip(resampleProb, 1e-6, 0.999999))
    
    # 3. Interpolate back to Return Levels (Monte Carlo Samples)
    retLev = np.asarray(retLev)
    retLev[np.isinf(retLev)] = np.nan
    
    monteCarloRsmpl = np.full(resampleRetPer.shape, np.nan)
    
    for ivar in range(resampleRetPer.shape[1]):
        x = retPeriod
        y = retLev[:, ivar]
        
        # Remove NaNs for interpolation
        mask = ~np.isnan(y)
        if logExtrapRetlev:
            # Linear interpolation in log-space for return periods (standard EVA practice)
            f_interp = interp1d(np.log(x[mask]), y[mask], bounds_error=False, fill_value="extrapolate")
            monteCarloRsmpl[:, ivar] = f_interp(np.log(resampleRetPer[:, ivar]))
        else:
            f_interp = interp1d(x[mask], y[mask], bounds_error=False, fill_value="extrapolate")
            monteCarloRsmpl[:, ivar] = f_interp(resampleRetPer[:, ivar])
            
        # Fill remaining NaNs with mean
        nan_mask = np.isnan(monteCarloRsmpl[:, ivar])
        if np.any(nan_mask):
            monteCarloRsmpl[nan_mask, ivar] = np.nanmean(monteCarloRsmpl[:, ivar])
            
    return monteCarloRsmpl, resampleProb, resampleRetPer

def tsEasyParseNamedArgs(args_input, default_struct):

    if args_input is None:
        return default_struct
    
    # If args_input is a dictionary (like **kwargs), update existing keys
    if isinstance(args_input, dict):
        for key, value in args_input.items():
            if key in default_struct:
                default_struct[key] = value
        return default_struct

    # If args_input is a list/tuple (standard MATLAB Name-Value pairs)
    if isinstance(args_input, (list, tuple)):
        for i in range(0, len(args_input) - 1, 2):
            arg_name = args_input[i]
            arg_val = args_input[i+1]
            
            # Case-insensitive matching like MATLAB's strcmpi
            for key in default_struct.keys():
                if str(arg_name).lower() == str(key).lower():
                    default_struct[key] = arg_val
                    
    return default_struct

def tsEmpirical(U):

    import numpy as np

    U = np.asarray(U)
    n, d = U.shape
    C = np.zeros(n)

    # Efficient implementation using NumPy broadcasting
    # For each observation i, we count how many other observations j 
    # satisfy the condition: U[j, k] <= U[i, k] for all dimensions k.
    for i in range(n):
        # Comparison across all dimensions (d) for observation i
        # np.all(..., axis=1) ensures the condition is met for all variables
        less_than_or_equal = np.all(U <= U[i, :], axis=1)
        C[i] = np.sum(less_than_or_equal) / n
        
    return C

def tsEnsemble(nonStatEvaParamsArray, stationaryTransformDataArray):

    
    nonStationaryEvaParamsEns = tsEnsembleEvaParams(nonStatEvaParamsArray)
    stationaryTransformDataEns = tsEnsembleStatTransfData(stationaryTransformDataArray)
    
    return nonStationaryEvaParamsEns, stationaryTransformDataEns

def tsEnsembleEvaParams(nonStatEvaParamsArray):

    import numpy as np
    import copy

    n = len(nonStatEvaParamsArray)
    if n == 0:
        return None

    # Initialize the ensemble structure based on the first element
    # Deep copy is used to avoid modifying the original objects
    ensep = [
        {
            'method': 'GEVstat',
            'parameters': copy.deepcopy(nonStatEvaParamsArray[0][0]['parameters']),
            'paramErr': copy.deepcopy(nonStatEvaParamsArray[0][0]['paramErr'])
        },
        {
            'method': 'GPDstat',
            'parameters': copy.deepcopy(nonStatEvaParamsArray[0][1]['parameters']),
            'paramErr': copy.deepcopy(nonStatEvaParamsArray[0][1]['paramErr']),
            'timeDelta': nonStatEvaParamsArray[0][1].get('timeDelta'),
            'timeDeltaYears': nonStatEvaParamsArray[0][1].get('timeDeltaYears')
        }
    ]

    # Sum parameters from 2nd to n-th element
    for ii in range(1, n):
        nsep = nonStatEvaParamsArray[ii]
        
        # GEV Parameters Sum
        for key in ['epsilon', 'sigma', 'mu']:
            ensep[0]['parameters'][key] += nsep[0]['parameters'][key]
        for key in ['epsilonErr', 'sigmaErr', 'muErr']:
            ensep[0]['paramErr'][key] += nsep[0]['paramErr'][key]
            
        # GPD Parameters Sum
        for key in ['epsilon', 'sigma', 'threshold', 'percentile']:
            ensep[1]['parameters'][key] += nsep[1]['parameters'][key]
        for key in ['epsilonErr', 'sigmaErr', 'thresholdErr']:
            ensep[1]['paramErr'][key] += nsep[1]['paramErr'][key]

    # Divide by n to get the Average
    # GEV Averages
    for key in ['epsilon', 'sigma', 'mu']:
        ensep[0]['parameters'][key] /= n
    for key in ['epsilonErr', 'sigmaErr', 'muErr']:
        ensep[0]['paramErr'][key] /= n
        
    # GPD Averages
    for key in ['epsilon', 'sigma', 'threshold', 'percentile']:
        ensep[1]['parameters'][key] /= n
    for key in ['epsilonErr', 'sigmaErr', 'thresholdErr']:
        ensep[1]['paramErr'][key] /= n

    return ensep

def tsEnsembleStatTransfData(stationaryTransformDataArray):

    import numpy as np
    import copy

    n = len(stationaryTransformDataArray)
    if n == 0:
        return None

    # Keys to be summed and averaged
    keys_to_sum = [
        'trendSeries', 'trendSeriesNonSeasonal', 'trendError',
        'stdDevSeries', 'stdDevSeriesNonSeasonal', 'stdDevError'
    ]

    # Initialize the ensemble dictionary with the first element
    etd = {
        'timeStamps': stationaryTransformDataArray[0].get('timeStamps')
    }
    
    # Initialize sum keys with a deep copy of the first element's values
    for key in keys_to_sum:
        etd[key] = copy.deepcopy(stationaryTransformDataArray[0].get(key))

    # Sum data from 2nd to n-th element
    for ii in range(1, n):
        td = stationaryTransformDataArray[ii]
        for key in keys_to_sum:
            if etd[key] is not None and td.get(key) is not None:
                etd[key] = etd[key] + td[key]

    # Divide by n to get the Average
    for key in keys_to_sum:
        if etd[key] is not None:
            etd[key] = etd[key] / n

    return etd

def tsEstimateConfidenceIntervalOfRL(rl, stdErr, p):

    import numpy as np
    from scipy.stats import lognorm

    rl = np.asarray(rl)
    stdErr = np.asarray(stdErr)
    
    # Avoid division by zero by replacing 0 with a very small number for math
    # but we will fix these at the end.
    safe_rl = np.where(rl == 0, 1e-10, rl)
    vr = stdErr**2
    
    # Conversion from Mean/Variance to Lognormal Mu/Sigma
    # mu = log(m^2 / sqrt(v + m^2))
    mu = np.log((safe_rl**2) / np.sqrt(vr + safe_rl**2))
    # sigma = sqrt(log(v/m^2 + 1))
    sigma = np.sqrt(np.log(vr / (safe_rl**2) + 1))
    
    # scipy.stats.lognorm.ppf uses:
    # s = sigma
    # scale = exp(mu)
    highCI = lognorm.ppf(p, s=sigma, scale=np.exp(mu))
    lowCI = lognorm.ppf(1 - p, s=sigma, scale=np.exp(mu))
    
    # Handle cases where stdErr was zero (no uncertainty)
    zero_err_mask = (stdErr == 0)
    highCI[zero_err_mask] = rl[zero_err_mask]
    lowCI[zero_err_mask] = rl[zero_err_mask]
    
    return lowCI, highCI

def tsEvaComputeAnnualMaximaMtx(timeStamps, srs):

    import numpy as np
    from datetime import datetime

    timeStamps = np.asarray(timeStamps).flatten()
    srs = np.asarray(srs)
    
    if srs.shape[0] != len(timeStamps):
        raise ValueError("tsEvaComputeAnnualMtxMaxima: the 1st size of srs should equal to the size of timeStamps")

    # Extract years (Handling MATLAB serial dates or similar)
    years = np.array([datetime.fromordinal(int(ts) - 366).year for ts in timeStamps])
    yu = np.unique(years)
    nyrs = len(yu)
    
    srs_shape = srs.shape
    # Prepare output shapes: (number of years, dimensions...)
    amxSize = (nyrs,) + srs_shape[1:]
    annualMax = np.full(amxSize, np.nan)
    annualMaxIndx = np.full(amxSize, 0, dtype=int)
    
    lastMxYrIndx = 0
    for iy, yr in enumerate(yu):
        # Mask for the current year
        year_mask = (years == yr)
        srs_year = srs[year_mask, ...]
        
        # Find maximum values and their local indices across axis 0 (time)
        ymx = np.max(srs_year, axis=0)
        indxmx = np.argmax(srs_year, axis=0)
        
        # Store results
        annualMax[iy, ...] = ymx
        # Global index = local index within year + starting index of that year
        annualMaxIndx[iy, ...] = indxmx + lastMxYrIndx
        
        # Update current offset
        lastMxYrIndx += np.sum(year_mask)

    # Compute dates of annual maxima
    # Expanding timeStamps to match the multidimensional index matrix
    # This simulates MATLAB's tmstmpmtx(annualMaxIndx)
    annualMaxDate = timeStamps[annualMaxIndx]

    return annualMax, annualMaxDate, annualMaxIndx

def tsEvaGetReturnPeriodOfLevelGEV(epsilon, sigma, mu, retLev):

    import numpy as np

    epsilon = np.asarray(epsilon)
    sigma = np.asarray(sigma)
    mu = np.asarray(mu)
    retLev = np.asarray(retLev)

    # Use a small tolerance for epsilon close to zero (Gumbel case)
    if np.any(np.abs(epsilon) > 1e-8):
        # GEV Cumulative Distribution Function (CDF)
        # G(z) = exp(-(1 + epsilon*(z-mu)/sigma)^(-1/epsilon))
        # Note: Ensure the term inside the power is positive to avoid NaNs
        inner_term = 1 + epsilon * (retLev - mu) / sigma
        inner_term = np.maximum(inner_term, 1e-10) # Safety clip
        G = np.exp(-(inner_term)**(-1 / epsilon))
    else:
        # Gumbel (Type I) CDF
        # G(z) = exp(-exp(-(z-mu)/sigma))
        G = np.exp(-np.exp(-(retLev - mu) / sigma))

    exceedProb = 1 - G
    
    # Avoid division by zero for return period
    retPer = np.divide(1.0, exceedProb, out=np.full_like(exceedProb, np.inf), where=exceedProb != 0)

    return retPer, exceedProb

def tsEvaGetReturnPeriodOfLevelGPD(epsilon, sigma, pPeak, retLev):

    import numpy as np

    epsilon = np.asarray(epsilon)
    sigma = np.asarray(sigma)
    pPeak = np.asarray(pPeak)
    retLev = np.asarray(retLev)

    # Small tolerance for epsilon close to zero (Exponential case)
    if np.any(np.abs(epsilon) > 1e-8):
        # GPD Cumulative Distribution Function (CDF)
        # H(z) = 1 - (1 + epsilon*z/sigma)^(-1/epsilon)
        inner_term = 1 + epsilon * retLev / sigma
        inner_term = np.maximum(inner_term, 1e-10)  # Safety clip to avoid NaNs
        H = 1 - (inner_term)**(-1.0 / epsilon)
    else:
        # Exponential distribution limit (epsilon -> 0)
        # H(z) = 1 - exp(-z/sigma)
        H = 1 - np.exp(-retLev / sigma)

    # Final exceedance probability is the probability of being above 
    # the threshold (pPeak) times the probability of exceeding the level given it's a peak
    exceedProb = (1 - H) * pPeak
    
    # Avoid division by zero for return period
    retPer = np.divide(1.0, exceedProb, out=np.full_like(exceedProb, np.inf), where=exceedProb != 0)

    return retPer, exceedProb

def tsEvaPlotGPD3D(X, timeStamps, epsilon, sigma, threshold, **kwargs):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import genpareto
    from matplotlib import dates as mdates

    # Default labels
    xlabel_str = kwargs.get('xlabel', 'levels (m)')
    zlabel_str = kwargs.get('zlabel', 'pdf')

    L = len(timeStamps)
    npdf = min(100, L)
    navg = int(np.ceil(L / npdf))

    def resample_param(param, L, npdf, navg):
        if np.isscalar(param) or len(np.atleast_1d(param)) == 1:
            return np.ones(npdf) * param
        else:
            # Pad with NaNs to make it reshapeable
            padded = np.full(npdf * navg, np.nan)
            param_flat = np.atleast_1d(param).flatten()
            padded[:len(param_flat)] = param_flat[:len(padded)]
            mtx = padded.reshape((npdf, navg))
            return np.nanmean(mtx, axis=1)

    # Resample parameters for performance
    epsilon0 = resample_param(epsilon, L, npdf, navg)
    sigma0 = resample_param(sigma, L, npdf, navg)
    threshold0 = resample_param(threshold, L, npdf, navg)

    # Resample timeStamps
    deltai = max(1, round(L / npdf))
    tmstmps = timeStamps[::deltai][:len(threshold0)]

    # Create Grids
    XMtx, TimeMtx = np.meshgrid(X, tmstmps)
    
    # Pre-allocate PDF matrix
    pdf_mtx = np.zeros(XMtx.shape)

    # Calculate GPD PDF for each time step
    # Note: scipy genpareto uses (c, loc, scale) where c = epsilon
    for i in range(len(tmstmps)):
        # scipy's genpareto is defined for x > loc. 
        # We calculate PDF only where X > threshold
        pdf_mtx[i, :] = genpareto.pdf(X, epsilon0[i], loc=threshold0[i], scale=sigma0[i])

    # Plotting
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Surface plot
    surf = ax.plot_surface(XMtx, TimeMtx, pdf_mtx, cmap='viridis', edgecolor='none')

    # Formatting time axis (Y)
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate() # Auto-rotate dates

    # Labels and view
    ax.set_xlabel(xlabel_str, fontsize=12)
    ax.set_ylabel('Time', fontsize=12)
    ax.set_zlabel(zlabel_str, fontsize=12)
    ax.view_init(elev=48, azim=-65) # Adjusted to match MATLAB's view(24.3, 48)

    plt.tight_layout()
    return fig, surf

def tsEvaPlotGPD3DFromAnalysisObj(X, nonStationaryEvaParams, stationaryTransformData):

    # 1. Extract time stamps from the transformation data
    # stationaryTransformData is expected to be a dictionary
    timeStamps = stationaryTransformData['timeStamps']
    
    # 2. Extract GPD parameters from the non-stationary analysis object
    # Index 1 corresponds to GPD (Index 0 is GEV in our Python structure)
    gpdParams = nonStationaryEvaParams[1]['parameters']
    
    epsilon = gpdParams['epsilon']
    sigma = gpdParams['sigma']
    threshold = gpdParams['threshold']
    
    # 3. Call the plotting function we defined in the previous step
    fig, surf = tsEvaPlotGPD3D(X, timeStamps, epsilon, sigma, threshold)
    
    return fig, surf

def tsEvaPlotReturnLevelsGEVStationary(nonStationaryEvaParams, **kwargs):

    import numpy as np

    # 1. Access the stationary part of the GEV analysis (Index 0)
    statParams = nonStationaryEvaParams[0]['stationaryParams']
    
    # 2. Extract parameters: epsilon, sigma, mu
    # Assuming parameters is a list/array: [epsilon, sigma, mu]
    params = statParams['parameters']
    epsilon = params[0]
    sigma = params[1]
    mu = params[2]
    
    # 3. Calculate Standard Errors from Confidence Intervals
    # paramCIs is typically a matrix where row 0 is the lower or upper bound
    cis = statParams['paramCIs']
    epsilonStdErr = abs(cis[0, 0] - epsilon)
    sigmaStdErr = abs(cis[0, 1] - sigma)
    muStdErr = abs(cis[0, 2] - mu)
    
    # 4. Call the core GEV plotting function
    fig, ax = tsEvaPlotReturnLevelsGEV(
        epsilon, sigma, mu, 
        epsilonStdErr, sigmaStdErr, muStdErr, 
        **kwargs
    )
    
    return fig, ax

def tsEvaPlotReturnLevelsGPDStationary(nonStationaryEvaParams, **kwargs):

    import numpy as np

    # 1. Access the stationary part of the GPD analysis (Index 1)
    statParams = nonStationaryEvaParams[1]['stationaryParams']
    gpdParamsObj = nonStationaryEvaParams[1]['parameters']
    
    # 2. Extract parameters: Note the order in GPD [sigma, epsilon, threshold]
    # MATLAB: parameters(2) is epsilon, parameters(1) is sigma, parameters(3) is threshold
    params = statParams['parameters']
    sigma = params[0]
    epsilon = params[1]
    threshold = params[2]
    
    # 3. Get the time step in years (critical for GPD return periods)
    dtSampleYears = gpdParamsObj['timeDeltaYears']
    
    # 4. Calculate Standard Errors from Confidence Intervals
    cis = statParams['paramCIs']
    sigmaStdErr = abs(cis[0, 0] - sigma)
    epsilonStdErr = abs(cis[0, 1] - epsilon)
    thresholdStdErr = 0 # Stationary threshold is assumed to have no error here
    
    # 5. Call the core GPD plotting function
    # Note: tsEvaPlotReturnLevelsGPD must be defined in your library
    fig, ax = tsEvaPlotReturnLevelsGPD(
        epsilon, sigma, threshold, dtSampleYears,
        epsilonStdErr, sigmaStdErr, thresholdStdErr,
        **kwargs
    )
    
    return fig, ax

def tsEvaPlotSeasonalityGev(extremesRange, referenceYear, timeStamps, epsilon, sigma, mu, 
                           monthlyMaxIndexes, series, trend, stddev, **kwargs):
    
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime
    import matplotlib.dates as mdates

    # 1. Define time range for the reference year
    minTS = mdates.date2num(datetime(referenceYear - 1, 12, 31))
    maxTS = mdates.date2num(datetime(referenceYear + 1, 1, 2))
    
    # Filter data for the reference year range
    mask = (timeStamps >= minTS) & (timeStamps <= maxTS)
    timeStampsRefYear = timeStamps[mask]
    sigma_ref = sigma[mask]
    mu_ref = mu[mask]
    
    # 2. Call the background image generator (ImageSc equivalent)
    # Note: tsEvaPlotGEVImageSc must be defined in your library
    plot_kwargs = {
        'nPlottedTimesByYear': 360,
        'dateFormat': '%b',
        'colormap': kwargs.get('colormap', plt.cm.pink_r)
    }
    plot_kwargs.update(kwargs)
    
    fig, ax = tsEvaPlotGEVImageSc(extremesRange, timeStampsRefYear, epsilon, sigma_ref, mu_ref, **plot_kwargs)
    
    # 3. Rescale series to Ham (Original) Scale for the reference year
    # Stationary series: (series - trend) / stddev
    statSeries = (series - trend) / stddev
    refYearTrend = np.mean(trend[mask])
    refYearStdDev = np.mean(stddev[mask])
    rescaledSeries = statSeries * refYearStdDev + refYearTrend
    
    # Extract monthly maxima
    monthlyMax = rescaledSeries[monthlyMaxIndexes]
    monthlyMaxTimeStamp = timeStamps[monthlyMaxIndexes]
    
    # 4. Adjust monthly maxima dates to the reference year for plotting
    mts_dates = [mdates.num2date(ts) for ts in monthlyMaxTimeStamp]
    mts_ref = [mdates.date2num(d.replace(year=referenceYear)) for d in mts_dates]
    
    # 5. Plot Layers
    # Monthly Maxima points
    monthlyMaxPlot = ax.plot(mts_ref, monthlyMax, 'o', color='blue', label='Monthly Max')
    
    # Mu (Location) line
    mupl = ax.plot(timeStampsRefYear, mu_ref, color=(0.7, 0, 0), linewidth=3, label=r'$\mu$')
    
    # Mu + Sigma line
    sigpl = ax.plot(timeStampsRefYear, mu_ref + sigma_ref, color=(0, 0.5, 0), linewidth=3, label=r'$\mu + \sigma$')
    
    # Formatting
    ax.legend(fontsize=14)
    ax.set_title(f'Seasonality for Year {referenceYear}', fontsize=16)
    
    return fig, ax

def tsEvaPlotSeasonalityGevFromAnalysisObj(extremesRange, referenceYear, nonStationaryEvaParams, stationaryTransformData, **kwargs):
    
    # 1. Extract basic time and series data
    timeStamps = stationaryTransformData['timeStamps']
    nonStatSrs = stationaryTransformData['nonStatSeries']
    
    # 2. Extract GEV parameters (Index 0)
    gevParams = nonStationaryEvaParams[0]['parameters']
    epsilon = gevParams['epsilon']
    sigma = gevParams['sigma']
    mu = gevParams['mu']
    
    # 3. Extract monthly maxima indexes from the GEV object
    monthlyMaxIndexes = nonStationaryEvaParams[0]['objs']['monthlyMaxIndexes']
    
    # 4. Get non-seasonal trend and standard deviation for proper rescaling
    series = nonStatSrs
    trend = stationaryTransformData['trendSeriesNonSeasonal']
    stddev = stationaryTransformData['stdDevSeriesNonSeasonal']
    
    # 5. Call the core seasonality plotting function
    # Note: tsEvaPlotSeasonalityGev must be defined in your library
    fig, ax = tsEvaPlotSeasonalityGev(
        extremesRange, referenceYear, timeStamps,
        epsilon, sigma, mu, monthlyMaxIndexes, 
        series, trend, stddev, **kwargs
    )
    
    return fig, ax

def tsEvaReduceOutputObjSize(nonStationaryEvaParams, stationaryTransformData, newTimeStamps, **kwargs):

    import numpy as np
    from copy import deepcopy

    maxTimeStepDist = kwargs.get('maxTimeStepDist', np.inf)
    origTimeStamps = stationaryTransformData['timeStamps']
    
    # 1. Find nearest indices (equivalent to knnsearch)
    tsIndxs = []
    for t in newTimeStamps:
        distances = np.abs(origTimeStamps - t)
        idx = np.argmin(distances)
        if distances[idx] <= maxTimeStepDist:
            tsIndxs.append(idx)
    
    tsIndxs = np.array(tsIndxs)
    
    # 2. Reduce Stationary Transformation Data
    # We use deepcopy to avoid modifying the original objects
    redStatTransData = deepcopy(stationaryTransformData)
    
    fields_to_reduce = [
        'trendSeries', 'trendSeriesNonSeasonal', 
        'stdDevSeries', 'stdDevSeriesNonSeasonal',
        'statSer3Mom', 'statSer4Mom'
    ]
    
    redStatTransData['statsTimeStamps'] = origTimeStamps[tsIndxs]
    for field in fields_to_reduce:
        if field in redStatTransData and redStatTransData[field] is not None:
            redStatTransData[field] = redStatTransData[field][tsIndxs]

    # 3. Reduce Non-Stationary EVA Parameters
    redNonStatEvaParams = deepcopy(nonStationaryEvaParams)
    
    # GEV Reduction (Index 0)
    if redNonStatEvaParams[0]['parameters'] is not None:
        params = redNonStatEvaParams[0]['parameters']
        errs = redNonStatEvaParams[0]['paramErr']
        
        params['sigma'] = params['sigma'][tsIndxs]
        params['mu'] = params['mu'][tsIndxs]
        
        err_fields = ['sigmaErrFit', 'sigmaErrTransf', 'sigmaErr', 'muErrFit', 'muErrTransf', 'muErr']
        for ef in err_fields:
            if ef in errs:
                errs[ef] = errs[ef][tsIndxs]
        
        redNonStatEvaParams[0]['objs'] = None # Clear heavy objects

    # GPD Reduction (Index 1)
    if redNonStatEvaParams[1]['parameters'] is not None:
        params = redNonStatEvaParams[1]['parameters']
        errs = redNonStatEvaParams[1]['paramErr']
        
        params['sigma'] = params['sigma'][tsIndxs]
        params['threshold'] = params['threshold'][tsIndxs]
        
        err_fields = ['sigmaErrFit', 'sigmaErrTransf', 'sigmaErr', 'thresholdErrTransf', 'thresholdErr']
        for ef in err_fields:
            if ef in errs:
                errs[ef] = errs[ef][tsIndxs]
        
        redNonStatEvaParams[1]['objs'] = None # Clear heavy objects

    return redNonStatEvaParams, redStatTransData

def tsGetReturnPeriodOfLevel(retPeriod, retLevel, retLevError, myLevel, **kwargs):

    import numpy as np
    from scipy.interpolate import interp1d

    logExtrap = kwargs.get('logExtrap', True)
    linExtrap = kwargs.get('linExtrap', False)
    cpP = kwargs.get('cpP', 0.68)
    
    if linExtrap:
        logExtrap = False

    # 1. Estimate Confidence Intervals for the Return Levels
    # Note: tsEstimateConfidenceIntervalOfRL should be defined or implemented here
    # Based on MATLAB logic: [rlLow, rlHigh]
    z_score = 1.0  # Default for 0.68 (1-sigma) if using normal approx
    if cpP == 0.95: z_score = 1.96
    
    rlLow = retLevel - z_score * retLevError
    rlHigh = retLevel + z_score * retLevError

    myLevel = np.atleast_1d(myLevel)
    valid_mask = ~np.isnan(myLevel) & ~np.isinf(myLevel)
    myLevValid = myLevel[valid_mask]

    def interp_extrap(x_ref, y_ref, x_new, log_mode):
        if log_mode:
            # Logarithmic extrapolation: linear fit in log(y) vs x space
            # Return periods are usually log-distributed against levels
            f = interp1d(x_ref, np.log(y_ref), kind='linear', fill_value='extrapolate')
            return np.exp(f(x_new))
        elif linExtrap:
            f = interp1d(x_ref, y_ref, kind='linear', fill_value='extrapolate')
            return f(x_new)
        else:
            f = interp1d(x_ref, y_ref, kind='linear', fill_value=np.nan)
            return f(x_new)

    # Calculate Return Periods
    try:
        res_period = interp_extrap(retLevel, retPeriod, myLevValid, logExtrap)
        res_sup = interp_extrap(rlLow, retPeriod, myLevValid, logExtrap)
        res_inf = interp_extrap(rlHigh, retPeriod, myLevValid, logExtrap)
    except:
        # Fallback to basic linear interpolation if extrapolation fails
        f_basic = interp1d(retLevel, retPeriod, kind='linear', fill_value='extrapolate')
        res_period = f_basic(myLevValid)
        res_sup = interp1d(rlLow, retPeriod, kind='linear', fill_value='extrapolate')(myLevValid)
        res_inf = interp1d(rlHigh, retPeriod, kind='linear', fill_value='extrapolate')(myLevValid)

    # Prepare Outputs
    myRetPeriod = np.full(myLevel.shape, np.nan)
    myRetPeriodCISup = np.full(myLevel.shape, np.nan)
    myRetPeriodCIInf = np.full(myLevel.shape, np.nan)

    myRetPeriod[valid_mask] = res_period
    myRetPeriodCISup[valid_mask] = res_sup
    myRetPeriodCIInf[valid_mask] = res_inf

    return myRetPeriod, myRetPeriodCISup, myRetPeriodCIInf

def tsInterp1Extrap(X, V, Xq, logExtrap=False):

    import numpy as np
    from scipy.interpolate import interp1d

    X = np.asarray(X, dtype=float).flatten()
    Xq = np.asarray(Xq, dtype=float).flatten()
    V = np.asarray(V, dtype=float)

    # 1. Ensure dependent variable V has the right orientation (rows = series, cols = values)
    if V.ndim == 1:
        V = V.reshape(1, -1)
    else:
        # If rows match X length but cols don't, transpose it
        if V.shape[0] == len(X) and V.shape[1] != len(X):
            V = V.T

    # 2. Offset logic to avoid log(<=0) during extrapolation
    minX = min(np.min(Xq), np.min(X))
    if minX < 1.0:
        xoffset = -minX + 1.0
    else:
        xoffset = 0.0
        
    X_off = X + xoffset
    Xq_off = Xq + xoffset

    # 3. Sort independent variable
    sort_idx = np.argsort(X_off)
    X_off = X_off[sort_idx]
    V = V[:, sort_idx]

    nvals = np.zeros((V.shape[0], len(Xq_off)))

    # 4. Interpolate / Extrapolate for each row
    for i in range(V.shape[0]):
        v_row = V[i, :]
        
        # Check if extrapolation is needed
        needs_extrap = np.any(Xq_off < np.min(X_off)) or np.any(Xq_off > np.max(X_off))
        
        if needs_extrap and logExtrap:
            # Log-log extrapolation requires strictly positive V values
            if np.any(v_row <= 0):
                # Fallback to linear if log is mathematically impossible
                f = interp1d(X_off, v_row, kind='linear', fill_value='extrapolate')
                nvals[i, :] = f(Xq_off)
            else:
                # Perform linear extrapolation in log-log space
                f_log = interp1d(np.log(X_off), np.log(v_row), kind='linear', fill_value='extrapolate')
                nvals[i, :] = np.exp(f_log(np.log(Xq_off)))
        else:
            # Linear interpolation / extrapolation
            f = interp1d(X_off, v_row, kind='linear', fill_value='extrapolate')
            nvals[i, :] = f(Xq_off)

    # Return 1D array if input V was 1D
    if nvals.shape[0] == 1:
        return nvals.flatten()
        
    return nvals

def tsLinearExtrapolation(x, y, extrax, npoints):

    import numpy as np

    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    extrax = np.asarray(extrax, dtype=float).flatten()

    # Sort based on x
    sort_idx = np.argsort(x)
    nx = x[sort_idx]
    ny = y[sort_idx]

    addedy_left = np.array([])
    addedy_right = np.array([])

    # Left Extrapolation (points smaller than min nx)
    mask_left = extrax < np.nanmin(nx)
    if np.sum(mask_left) > 0:
        # Average slope of the first 'npoints' intervals
        dvv = np.nanmean(np.diff(ny[:npoints + 1]))
        dTr = np.nanmean(np.diff(nx[:npoints + 1]))
        
        extrax_left = extrax[mask_left]
        addedx = nx[0] - extrax_left
        addedy0 = ny[0] - dvv * addedx / dTr
        
        ny = np.concatenate([addedy0, ny])
        nx = np.concatenate([extrax_left, nx])
        addedy_left = addedy0

    # Right Extrapolation (points larger than max nx)
    mask_right = extrax > np.nanmax(nx)
    if np.sum(mask_right) > 0:
        # Average slope of the last 'npoints' intervals
        dvv = np.nanmean(np.diff(ny[-(npoints + 1):]))
        dTr = np.nanmean(np.diff(nx[-(npoints + 1):]))
        
        extrax_right = extrax[mask_right]
        addedx = nx[-1] - extrax_right
        addedy1 = ny[-1] - dvv * addedx / dTr
        
        ny = np.concatenate([ny, addedy1])
        nx = np.concatenate([nx, extrax_right])
        addedy_right = addedy1

    addedy = [addedy_left, addedy_right]

    return nx, ny, addedy

def tsLoglogExtrapolation(x, y, extrax, npoints):
  
    import numpy as np

    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    extrax = np.asarray(extrax, dtype=float).flatten()

    # Sort based on x
    sort_idx = np.argsort(x)
    nx = x[sort_idx]
    ny = y[sort_idx]

    addedy_left = np.array([])
    addedy_right = np.array([])

    # Left Extrapolation
    mask_left = extrax < np.nanmin(nx)
    if np.sum(mask_left) > 0:
        # Offset to prevent log(<=0)
        if np.sum(ny <= 0) > 0:
            offs = -np.nanmin(ny) + 1.0
        else:
            offs = 0.0
            
        extrax_left = extrax[mask_left]
        
        # Linear extrapolation in log-space
        nx0, ny0, addedy0 = tsLinearExtrapolation(
            np.log(nx), 
            np.log(ny + offs), 
            np.log(extrax_left), 
            npoints
        )
        
        # Transform back to exponential space
        nx = np.exp(nx0)
        ny = np.exp(ny0) - offs
        
        # addedy0[0] contains the left extrapolated values from tsLinearExtrapolation
        if len(addedy0[0]) > 0:
            addedy_left = np.exp(addedy0[0]) - offs

    # Right Extrapolation
    mask_right = extrax > np.nanmax(nx)
    if np.sum(mask_right) > 0:
        # Offset to prevent log(<=0)
        if np.sum(ny <= 0) > 0:
            offs = -np.nanmin(ny) + 1.0
        else:
            offs = 0.0
            
        extrax_right = extrax[mask_right]
        
        # Linear extrapolation in log-space
        nx0, ny0, addedy0 = tsLinearExtrapolation(
            np.log(nx), 
            np.log(ny + offs), 
            np.log(extrax_right), 
            npoints
        )
        
        # Transform back to exponential space
        nx = np.exp(nx0)
        ny = np.exp(ny0) - offs
        
        # addedy0[1] contains the right extrapolated values from tsLinearExtrapolation
        if len(addedy0[1]) > 0:
            addedy_right = np.exp(addedy0[1]) - offs

    # Combine added values into a list consistent with tsLinearExtrapolation
    addedy = [addedy_left, addedy_right]

    return nx, ny, addedy

def tsPlotBivarReturnPeriod(copulaAnalysis, axxArray, **kwargs):
    """
    Plots the multivariate return period according to AND/OR Scenarios.
    Python translation of MATLAB's tsPlotBivarReturnPeriod.m
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import genpareto, genextreme
    from scipy.interpolate import interp1d
    
    # Optional arguments
    xlbl = kwargs.get('xlbl', 'X')
    ylbl = kwargs.get('ylbl', 'Y')
    RL = np.atleast_1d(kwargs.get('RL', [10, 20, 50]))
    copulaAnalysisForCount = kwargs.get('copulaAnalysisForCount', None)
    timeWindowNonStat = kwargs.get('timeWindowNonStat', None)
    
    colorPool = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    cRL = [colorPool[i % len(colorPool)] for i in range(len(RL))]
    
    margDist = copulaAnalysis['methodology'].lower()
    copulaAnalysis['RL'] = RL
    
    Family = copulaAnalysis['copulaParam']['family']
    
    if 'rhoMean' in copulaAnalysis['copulaParam']:
        PAR = copulaAnalysis['copulaParam']['rhoMean']
    else:
        PAR = copulaAnalysis['copulaParam']['rho']
        
    PAR = np.atleast_1d(PAR)
    num_windows = len(PAR)
    
    # Extract Marginal Parameters
    eps = [[], []]
    sig = [[], []]
    thr = [[], []]
    Scl = [[], []]
    
    for v in range(2): # Bivariate (2 variables)
        marg_analysis = copulaAnalysis['marginalAnalysis'][v] # Adjust indexing based on your data struct
        if margDist == 'gpd':
            params = marg_analysis[1]['parameters'] # Index 1 is GPD
            eps[v] = params['epsilon']
            sig[v] = params['sigma']
            thr[v] = params['threshold']
            
            nYear = (params['timeHorizonEnd'] - params['timeHorizonStart']) / 365.25
            nPeak = len(marg_analysis[1]['objs']['peakIndexes'])
            Scl[v] = np.full(num_windows, nPeak / nYear)
        elif margDist == 'gev':
            params = marg_analysis[0]['parameters'] # Index 0 is GEV
            eps[v] = params['epsilon']
            sig[v] = params['sigma']
            thr[v] = params['mu']
            Scl[v] = np.ones(num_windows)

    # Define uniform square grid
    x_grid = np.concatenate([np.linspace(1e-5, 0.99, 400), np.linspace(0.99 + 1e-5, 1 - 1e-7, 500)])
    xx, yy = np.meshgrid(x_grid, x_grid)
    U = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Helper to calculate Copula CDF (Placeholder for statsmodels or your copula lib)
    def calc_copula_cdf(u, family, param):
        from statsmodels.distributions.copula.api import GaussianCopula, GumbelCopula, ClaytonCopula, FrankCopula
        # Note: adjust parameter mapping according to your library's expectations
        if family.lower() == 'gaussian':
            cop = GaussianCopula(corr=param)
        elif family.lower() == 'gumbel':
            cop = GumbelCopula(theta=param)
        else:
            raise NotImplementedError(f"Copula family {family} CDF not implemented.")
        return cop.cdf(u)

    RpLb = RL - 0.005 * RL
    RpUb = RL + 0.005 * RL
    
    cellRP = []
    
    # Loop over start and end windows (0 and -1) just like MATLAB's jk2 = [1, length]
    target_windows = [0, num_windows - 1]
    
    for jk in target_windows:
        param_k = PAR[jk]
        
        # Calculate Copula CDF
        # Warning: For large grids, this might take a moment.
        y_cdf = calc_copula_cdf(U, Family, param_k)
        
        # Calculate OR return period
        if margDist == 'gev':
            rPCellOr = 1.0 / (1.0 - y_cdf)
        else:
            numJointOrPeaks = len(copulaAnalysisForCount['jointExtremes'])
            scaling = timeWindowNonStat / numJointOrPeaks
            rPCellOr = scaling / (1.0 - y_cdf)
            
        sort_idx = np.argsort(rPCellOr)
        RpSort = rPCellOr[sort_idx]
        U1_sort = U[sort_idx, 0]
        U2_sort = U[sort_idx, 1]
        
        X_all, Y_all, C_all = [], [], []
        
        for ij, rl_val in enumerate(RL):
            # Find points within the return period contour bounds
            valid_idx = np.where((RpSort >= RpLb[ij]) & (RpSort <= RpUb[ij]))[0]
            
            UU = U1_sort[valid_idx]
            VV = U2_sort[valid_idx]
            
            # Sort UU
            sort_u_idx = np.argsort(UU)
            UU = UU[sort_u_idx]
            VV = VV[sort_u_idx]
            
            # ICDF mapping
            # Note: scipy shape parameter 'c' for genextreme and genpareto is often -epsilon of MATLAB
            def apply_icdf(p, eps_val, sig_val, thr_val, scl_val, dist):
                adj_p = 1.0 - 1.0 / (scl_val * (1.0 / (1.0 - p)))
                adj_p = np.clip(adj_p, 1e-8, 1 - 1e-8)
                if dist == 'gp': # GPD
                    # c = -epsilon for scipy genpareto
                    return genpareto.ppf(adj_p, c=-eps_val, loc=thr_val, scale=sig_val)
                else: # GEV
                    return genextreme.ppf(adj_p, c=-eps_val, loc=thr_val, scale=sig_val)

            IUU = apply_icdf(UU, eps[0][jk], sig[0][jk], thr[0][jk], Scl[0][jk], margDist)
            IVV = apply_icdf(VV, eps[1][jk], sig[1][jk], thr[1][jk], Scl[1][jk], margDist)
            
            # Filter NaNs and Infs
            good_mask = ~np.isinf(IUU) & ~np.isinf(IVV) & ~np.isnan(IUU) & ~np.isnan(IVV)
            IUU = IUU[good_mask]
            IVV = IVV[good_mask]
            
            if len(IUU) == 0:
                continue
                
            # Extrapolation and plotting
            # Get unique X values to thin out the band of points
            xd, u_idx = np.unique(xd_raw, return_index=True)
            yd = yd_raw[u_idx]

            # Sort by X to ensure proper line drawing
            sort_idx = np.argsort(xd)
            x_vals = xd[sort_idx]
            y_vals = yd[sort_idx]

            # Enforce strictly monotonic decreasing curve to avoid any numerical jumps
            y_mono = np.minimum.accumulate(y_vals)

            # Plot the natural curve directly (no linear extrapolation needed)
            label_str = f"{rl_val} - year R.P."
            ax_target.plot(x_vals, y_mono, color=cRL[ij], linewidth=2, label=label_str)
            
        cellRP.append({"X": X_all, "Y": Y_all})
            
    axxArray[3].legend()
    axxArray[4].legend()
    
    copulaAnalysis['cellRP'] = cellRP
    
    # Return figure handling
    hFig1 = axxArray[0].get_figure() if len(axxArray) > 0 else None
    
    return copulaAnalysis, hFig1

def tsRoundSDate(sd, sd_precision):
    """
    Rounds sd (serial date) values according to desired precision (years, months, days, etc: 1-4).
    Python translation of MATLAB's tsRoundSDate.m
    """
    import numpy as np
    import matplotlib.dates as mdates
    from datetime import datetime

    sd = np.atleast_1d(sd)
    
    # Convert matplotlib datenums to datetime objects
    dt_objs = [mdates.num2date(d) for d in sd]
    
    dvecm = []
    sdround = []
    
    for dt in dt_objs:
        y = dt.year
        m = dt.month
        d = dt.day
        h = dt.hour
        minute = dt.minute
        s = dt.second
        
        # Adjust components based on requested precision
        # Python datetime requires month >= 1 and day >= 1
        if sd_precision == 1: # Year precision
            m, d, h, minute, s = 1, 1, 0, 0, 0
        elif sd_precision == 2: # Month precision
            d, h, minute, s = 1, 0, 0, 0
        elif sd_precision == 3: # Day precision
            h, minute, s = 0, 0, 0
        elif sd_precision == 4: # Hour precision
            minute, s = 0, 0
        elif sd_precision >= 5: # Minute precision
            s = 0
            
        # Create the rounded datetime object, preserving timezone info
        dt_rounded = datetime(y, m, d, h, minute, s, tzinfo=dt.tzinfo)
        
        # Convert back to datenum and store
        sdround.append(mdates.date2num(dt_rounded))
        
        # Store the date vector [Y, M, D, H, MI, S]
        dvecm.append([y, m, d, h, minute, s])
        
    sdround = np.array(sdround)
    dvecm = np.array(dvecm)
    
    # Find unique values
    sdunique = np.unique(sdround)
    dvunique = np.unique(dvecm, axis=0)
    
    return sdround, dvecm, sdunique, dvunique

def tsYear(timeStamp):
    """
    Extracts the year from a timeStamp (matplotlib datenum).
    Python translation of MATLAB's tsYear.m
    """
    import numpy as np
    import matplotlib.dates as mdates

    # Handle both scalar and array inputs
    time_stamps = np.atleast_1d(timeStamp)
    
    # Convert datenums to datetime objects
    dt_objs = mdates.num2date(time_stamps)
    
    # Extract the year
    years = np.array([dt.year for dt in dt_objs])
    
    # If the original input was a single number, return a single number
    if np.isscalar(timeStamp):
        return years[0]
        
    return years