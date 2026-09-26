# Large-Scale Monovariate tsEVA — Python Guidelines

## 1. Output organization (large-scale monovariate tsEVA)

Large-scale monovariate tsEVA runs (gridded or many stations) should produce outputs that are:
- **map-friendly** (easy to reshape to lon×lat),
- **subset-friendly** (easy to extract a given return period and/or year),
- **self-consistent** (fixed dimensions, missing values handled uniformly),
- **self-describing** (axes and metadata stored with the product).

The guiding principle is that return levels are a function of:
- **space** (grid cell / station / point index),
- **return period** (years),
- **evaluation time** (only for non-stationary runs).

---

### 1.1 Axes and tensor layout

For each point `p`, the non-stationary return level can be written as:

`RL(p, T, t)`

where:
- `T` is the return period in years (`returnPeriodsInYears`)
- `t` is the evaluation time (`retLevTimeStamps`, often ≤ 1 timestamp per year)

Recommended storage layouts (numpy arrays):

- **Unstructured point set** (most general):  
  `RL(npt, nRP, nEval)` — shape `(n_points, n_return_periods, n_eval_times)`

- **Gridded domain** (if you store lon/lat explicitly):  
  `RL(nLon, nLat, nRP, nEval)`  
  (or store flattened `npt = nLon*nLat` and reshape later)

with:
- `npt = number of points`
- `nRP = len(returnPeriodsInYears)`
- `nEval = len(retLevTimeStamps)` (or `len(retLevYears)`)

---

### 1.2 Evaluation-time axis: `retLevYears` → `retLevTimeStamps`

In the TS (Transformed-Stationary) framework, the fitted distribution evolves smoothly. For large-scale products, it is usually sufficient to store return levels at **≤ 1 timestamp per year**.

A common convention is to evaluate at **January 1st** of each year:

```python
import numpy as np
from datetime import datetime

retLevYears = np.arange(1970, 2101)  # example
# Convert to datenum format (ordinal day + fractional day + 366 offset)
# or use pandas timestamps depending on your pipeline
```

Notes:
- `retLevYears` defines the **output time grid** (storage + temporal resolution).
- If stationary, you can keep compatibility by setting `nEval = 1` (single timestamp).

---

### 1.3 Return-period axis: `returnPeriodsInYears`

Choose return periods based on the intended application (risk mapping, design levels, adaptation planning). Store them as a 1D array:

```python
returnPeriodsInYears = np.array([1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 70, 100, 
                                  150, 250, 350, 500, 700, 1000, 1500, 2000, 5000, 10000])
```

Guidelines:
- Keep them **sorted ascending**.
- Very large return periods (e.g., 5000–10000y) are possible but represent strong extrapolation; they should always be accompanied by uncertainty layers.

---

### 1.4 Stationary vs non-stationary products

Typically, large-scale applications target the **non-stationary** product. If support for the **stationary** analysis is needed, keep the output structure consistent.

**Non-stationary run**
- Return levels depend on evaluation time:
  - `RL(…, nRP, nEval)`
  - `RLerr(…, nRP, nEval)` (recommended)

**Stationary run**
- Return levels do not depend on time.
- For interoperability with non-stationary outputs, prefer a unified structure:
  - use `nEval = 1` (single evaluation timestamp) so arrays remain `(..., nRP, nEval)`.

---

### 1.5 Preallocation of output arrays (mandatory for large scale)

Before entering the main loop, preallocate all output arrays at their final size using numpy:

```python
npt   = len(points)          # number of stations/grid cells
nRP   = len(returnPeriodsInYears)
nEval = len(retLevTimeStamps)

# Return levels and uncertainties: (point × returnPeriod × evalTime)
retLevGEV    = np.full((npt, nRP, nEval), np.nan)
retLevErrGEV = np.full((npt, nRP, nEval), np.nan)

retLevGPD    = np.full((npt, nRP, nEval), np.nan)
retLevErrGPD = np.full((npt, nRP, nEval), np.nan)

# Parameters:
# - time-invariant (shape): (point × 1)
shapeGEV    = np.full(npt, np.nan)
shapeGEVErr = np.full(npt, np.nan)
shapeGPD    = np.full(npt, np.nan)
shapeGPDErr = np.full(npt, np.nan)

# - time-varying (evaluated at each timestamp): (point × evalTime)
scaleGEV    = np.full((npt, nEval), np.nan)
scaleGEVErr = np.full((npt, nEval), np.nan)
locGEV      = np.full((npt, nEval), np.nan)
locGEVErr   = np.full((npt, nEval), np.nan)

scaleGPD        = np.full((npt, nEval), np.nan)
scaleGPDErr     = np.full((npt, nEval), np.nan)
thresholdGPD    = np.full((npt, nEval), np.nan)
thresholdGPDErr = np.full((npt, nEval), np.nan)

# Validity flag
fit_ok = np.zeros(npt, dtype=bool)
```

Recommendations:
- Keep naming explicit.
- If a point is invalid (too many missing values, too few extremes, failed fit), leave outputs as `NaN` and set `fit_ok[ipt] = False`.

---

### 1.6 Missing values and masks

- Use `np.nan` for missing/invalid values in numpy arrays.
- In NetCDF output, use a consistent `_FillValue` (typically `-9999` or `np.nan`) and propagate the mask via `xarray` or `netCDF4`.
- A point that fails should be represented by all-`NaN` outputs (plus `fit_ok = False`).

---

## 2. Large-scale execution loop (parallel over points) and per-point products

This section describes the recommended *execution pattern* for large-scale monovariate tsEVA: a parallel loop over points, where each worker reads one time series, runs tsEVA, computes return levels on the requested axes, and writes results into preallocated arrays.

> **Separation of roles**
> - **tsEVA core steps (documented workflow):** build `timeAndSeries`, run `tsEvaNonStationary` (or `tsEvaStationary`), compute return levels from the analysis object, extract parameters.
> - **Project utilities (I/O helpers):** functions for reading/writing data are part of the *large-scale pipeline*, not the scientific core.

---

### 2.1 Why parallelize over points

Large-scale EVA is "embarrassingly parallel": each grid cell/station can be processed independently. The recommended pattern in Python:

- **parallelize over points** using `multiprocessing.Pool` or `concurrent.futures.ProcessPoolExecutor`
- keep the **time dimension internal** to the point's tsEVA call
- write into **preallocated** arrays using sliced indexing (`retLevGEV[ipt, :, :] = ...`)

This is the most robust approach for performance and memory control.

**Debug vs production.** It is recommended to debug with a standard `for` loop (deterministic behavior, easier breakpoints, simpler logs). For production runs, switching to multiprocessing is typically much faster. In this workflow the two are intentionally interchangeable: you can switch between sequential and parallel by changing one line, provided that outputs are **preallocated** and assignments use **sliced indexing**.

---

### 2.2 Per-point workflow inside the loop

Each iteration follows the same steps:

1) **Identify point location** (lon/lat or station id)  
2) **Read the time series** for that point (project-specific I/O helper)  
3) **Apply optional time horizon subsetting**  
4) **Apply quality filters** (remove invalid values; skip empty/constant series)  
5) **Run EVA**
   - `tsEvaNonStationary(timeAndSeries, timeWindow, ...)` or  
   - `tsEvaStationary(timeAndSeries, ...)`
6) **Skip invalid fits** (`if not is_valid: continue`)
7) **Compute return levels** for both GEV and GPD from the analysis object
8) **Store results** into preallocated output arrays

---

### 2.3 Computing return levels (GEV and GPD) on the requested axes

After running EVA, return levels are computed for each return period in `returnPeriodsInYears`:

```python
# GEV return levels — returns (returnLevels, returnLevelsErr) as 2-tuple
RLgev, RLgevErr = tsEvaComputeReturnLevelsGEVFromAnalysisObj(
    nonStationaryEvaParams, returnPeriodsInYears)

retLevGEV[ipt, :, :]     = RLgev.T      # transpose to (nRP × nEval) if needed
retLevErrGEV[ipt, :, :]  = RLgevErr.T

# GPD return levels — same pattern
RLgpd, RLgpdErr = tsEvaComputeReturnLevelsGPDFromAnalysisObj(
    nonStationaryEvaParams, returnPeriodsInYears)

retLevGPD[ipt, :, :]     = RLgpd.T
retLevErrGPD[ipt, :, :]  = RLgpdErr.T
```

**Shape convention**
- `returnLevels` are typically returned as `(nEval × nRP)` and transposed into `(nRP × nEval)` to match the preallocated array layout.

> **Note:** In tsEva-py, return level functions return a 2-tuple `(returnLevels, returnLevelsErr)`. The ErrFit/ErrTrans separation available in MATLAB is commented out in Python — only total uncertainty is computed.

---

### 2.4 Storing parameters (shape vs time-varying parameters)

The large-scale product often stores:
- a **single** shape parameter per point (e.g., `shapeGEV[ipt]`), and
- time-varying parameters per point and evaluation time (e.g., `scaleGEV[ipt, :]`).

Example extraction from the analysis object:

```python
# GEV parameters — nonStationaryEvaParams[0] is the GEV entry
shapeGEV[ipt]    = nonStationaryEvaParams[0]['parameters']['epsilon']
shapeGEVErr[ipt] = nonStationaryEvaParams[0]['paramErr']['epsilonErr']

scaleGEV[ipt, :]    = np.array(nonStationaryEvaParams[0]['parameters']['sigma'])
scaleGEVErr[ipt, :] = np.array(nonStationaryEvaParams[0]['paramErr']['sigmaErr'])

locGEV[ipt, :]    = np.array(nonStationaryEvaParams[0]['parameters']['mu'])
locGEVErr[ipt, :] = np.array(nonStationaryEvaParams[0]['paramErr']['muErr'])

# GPD parameters — nonStationaryEvaParams[1] is the GPD entry
shapeGPD[ipt]    = nonStationaryEvaParams[1]['parameters']['epsilon']
shapeGPDErr[ipt] = nonStationaryEvaParams[1]['paramErr']['epsilonErr']

scaleGPD[ipt, :]    = np.array(nonStationaryEvaParams[1]['parameters']['sigma'])
scaleGPDErr[ipt, :] = np.array(nonStationaryEvaParams[1]['paramErr']['sigmaErr'])

thresholdGPD[ipt, :]    = np.array(nonStationaryEvaParams[1]['parameters']['threshold'])
thresholdGPDErr[ipt, :] = np.array(nonStationaryEvaParams[1]['paramErr']['thresholdErr'])
```

> **GEV sign convention:** `epsilon` here follows MATLAB convention (shape parameter ε). scipy.stats.genextreme uses c = -ε. When comparing with scipy fits or reading parameters from other tools, remember the sign flip.

---

### 2.5 Optional per-point saving of analysis details (debug/QC)

To support future checking without rerunning the full analysis, store the analysis objects per point:

```python
import pickle

if save_analysis_details:
    eva_path = os.path.join(eva_dir, f'point_{ipt}.pkl')
    with open(eva_path, 'wb') as f:
        pickle.dump({
            'nonStationaryEvaParams': nonStationaryEvaParams,
            'stationaryTransformData': stationaryTransformData,
        }, f)
```

Notes:
- Prefer **one file per point** (or per chunk of points) to prevent write contention and allow selective post-mortem inspection.
- For large-scale runs where disk space matters, consider storing only the reduced parameters (see §2.6).

---

### 2.6 Reduced analysis object for scalable diagnostics

For large-scale runs where storing full analysis objects per point is impractical, store a **reduced version**:

```python
def reduce_analysis_object(nonStationaryEvaParams, stationaryTransformData):
    """Reduce analysis object for large-scale storage (follows MATLAB tsEvaReduceOutputObjSize convention).

    Keep: BOTH original and transformed time series intact — they are data, not working artifacts.
    Keep: fitted parameters at 1 value/year resolution, thresholdError, validity flags.
    Drop: intermediate computation artifacts not needed for return level recomputation or debugging.
    """
    reduced = {
        'gev': {
            'epsilon': nonStationaryEvaParams[0]['parameters']['epsilon'],
            'sigma_annual': np.array(nonStationaryEvaParams[0]['parameters']['sigma']),  # already ≤1/year
            'mu_annual': np.array(nonStationaryEvaParams[0]['parameters']['mu']),
            'paramErr': nonStationaryEvaParams[0]['paramErr'],
        },
        'gpd': {
            'epsilon': nonStationaryEvaParams[1]['parameters']['epsilon'],
            'sigma_annual': np.array(nonStationaryEvaParams[1]['parameters']['sigma']),
            'threshold_annual': np.array(nonStationaryEvaParams[1]['parameters']['threshold']),
            'paramErr': nonStationaryEvaParams[1]['paramErr'],
            'thresholdError': nonStationaryEvaParams[1].get('thresholdError', None),
        },
    }
    return reduced
```

**What to keep:**
- Both original and transformed time series (data, not working artifacts)
- Fitted parameters (time-varying ones at 1 value/year resolution — already the convention)
- `thresholdError` for GPD uncertainty propagation
- Validity flags (`is_valid`)

**What to drop:**
- Intermediate computation artifacts not needed for return level recomputation or debugging

This is lighter than MATLAB's `tsEvaReduceOutputObjSize` approach and sufficient for return level recomputation + diagnostics.

---

### 2.X Parallel pool lifecycle and safe cleanup (Python equivalent)

Large-scale runs often create a process pool explicitly to control how many workers are used and ensure proper shutdown even if an error occurs:

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

def run_large_scale_eva(points, n_workers=4):
    results = {}
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_point, pt): pt for pt in points}
        
        try:
            for future in as_completed(futures):
                pt = futures[future]
                try:
                    results[pt] = future.result()
                except Exception as e:
                    print(f"Point {pt} failed: {e}")
                    traceback.print_exc()
                    # Leave outputs as NaN — don't crash the whole run
        
        except KeyboardInterrupt:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
    
    return results

# Normal completion: pool is automatically shut down by context manager
```

**Why this pattern:**
- **Control:** `max_workers` enforces the requested worker count.
- **Resource hygiene:** the context manager ensures cleanup even on exceptions.
- **Non-fatal failures:** individual point errors don't crash the entire run — they're logged and skipped.

---

### 2.7 Practical runtime hygiene inside parallel workers

- Minimize heavy console I/O (`print`) inside worker processes (it can become a bottleneck).
- Keep point-level failures non-fatal: if a fit fails, `continue` and leave outputs as `NaN`.
- Use logging module instead of print for structured diagnostics.

---

## 3. Saving the large-scale results (NetCDF product)

After the parallel loop completes, the analysis results exist as **preallocated numpy arrays** (mostly `NaN`-filled, with valid points populated). Section 3 defines a **portable, self-describing output format** and writes the product to disk. The chosen format is **NetCDF** because it is:
- language-agnostic (MATLAB/Python/R),
- efficient for large multidimensional arrays,
- standard in geosciences and easy to post-process and map.

---

### 3.1 Output file handling (overwrite safely)

```python
import os

print('Parallel loop complete. Saving return levels to output NetCDF')
print(retLevNcOutFilePath)

if os.path.exists(retLevNcOutFilePath):
    os.remove(retLevNcOutFilePath)
```

Guideline:
- Always **overwrite** the target file explicitly to avoid mixing old and new dimensions/variables.

---

### 3.2 Define axis sizes once

The product axes are:
- `pt` (point index): `npt`
- `return_period`: `nretper`
- `rlyear` (evaluation year): `nRlTime`

```python
nRlTime = len(retLevYears)
nretper = len(returnPeriodsInYears)
```

These sizes must match the preallocated arrays from Section 1.

---

### 3.3 Write model outputs as NetCDF variables

Use `xarray` or `netCDF4` to create a self-describing file with explicit dimensions:

```python
import xarray as xr
import numpy as np

ds = xr.Dataset()

# Coordinates
ds['pt'] = ('pt', np.arange(npt))
ds['return_period'] = ('return_period', returnPeriodsInYears)
ds['rlyear'] = ('rlyear', retLevYears)

# Return levels and uncertainty (GEV) — dimensions: (pt, return_period, rlyear)
ds['returnlevelGEV'] = ('(pt, return_period, rlyear)', retLevGEV)
ds['returnlevelErrorGEV'] = ('(pt, return_period, rlyear)', retLevErrGEV)

# Return levels and uncertainty (GPD)
ds['returnlevelGPD'] = ('(pt, return_period, rlyear)', retLevGPD)
ds['returnlevelErrorGPD'] = ('(pt, return_period, rlyear)', retLevErrGPD)

# Parameters — time-invariant (shape): dimensions: (pt,)
ds['shapeGEV'] = ('pt', shapeGEV)
ds['shapeGEVErr'] = ('pt', shapeGEVErr)
ds['shapeGPD'] = ('pt', shapeGPD)
ds['shapeGPDErr'] = ('pt', shapeGPDErr)

# Parameters — time-varying: dimensions: (pt, rlyear)
ds['scaleGEV'] = ('(pt, rlyear)', scaleGEV)
ds['locGEV'] = ('(pt, rlyear)', locGEV)
ds['scaleGPD'] = ('(pt, rlyear)', scaleGPD)
ds['thresholdGPD'] = ('(pt, rlyear)', thresholdGPD)

# Validity flag
ds['fit_ok'] = ('pt', fit_ok.astype(np.int8))

# Attributes for self-description
ds.attrs['framework'] = 'tsEVA 2.0 (Transformed-Stationary)'
ds.attrs['methodology'] = 'Non-stationary EVA via TS approach'
ds.attrs['return_periods_years'] = str(list(returnPeriodsInYears))
ds.attrs['evaluation_times'] = f'{nRlTime} timestamps, typically Jan 1st of each year'

# Save with compression
ds.to_netcdf(retLevNcOutFilePath, encoding={var: {'zlib': True, 'complevel': 4} for var in ds.data_vars})
```

Guidelines:
- Always include **attributes** describing the methodology, axes conventions, and any special handling.
- Use compression (`zlib`) for large arrays — it reduces file size significantly with minimal read penalty.
- Store `fit_ok` as a separate variable so users can easily mask invalid points.

---

## 4. Memory management tips for Python

Unlike MATLAB's automatic memory management, Python requires explicit attention to avoid excessive RAM usage during large-scale runs:

1. **Process one point at a time** in the worker — don't load all series into memory simultaneously.
2. **Delete intermediate objects** after extracting parameters and return levels:
   ```python
   del nonStationaryEvaParams, stationaryTransformData
   import gc; gc.collect()  # optional, helps with large arrays
   ```
3. **Use numpy's `del` or reassignment** to free memory before processing the next point.
4. **Monitor peak memory** during debugging — if it grows unbounded, you're leaking references somewhere.

These practices are especially important when running on HPC clusters with limited per-node RAM.
