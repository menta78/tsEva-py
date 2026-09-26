# tsEva-py — Code Agent Instructions

## Tessa P — EVA Analysis Assistant (read first)

This repository contains **Tessa P**, the expert assistant for tsEVA extreme value analysis. Before performing any EVA-related task, read `agent_tessa/0_README.md` and follow its routing to load the relevant reference files. Only use functions/workflows explicitly documented there — never invent function names.

## Project Overview

Python port of the MATLAB tsEva framework for extreme value analysis (EVA). Supports both monovariate and multivariate (copula-based) non-stationary extreme event analysis. The reference implementation is in MATLAB at `D:\src\git\tsEva`.

## Architecture

Two core modules:

- **`tsEva.py`** — Monovariate EVA. GEV/GPD fitting, non-stationary transformations (trend linear, seasonal), POT sampling, return level computation, plotting functions. Contains `_gev_pwm_init` and `_gev_mle_with_pwm_guard` for robust GEV MLE initialization using Hosking's PWM/L-moments.

- **`tsEvaMultivariate.py`** — Multivariate joint extremes via copulas (Gaussian, Gumbel, Clayton, Frank). Imports from `tsEva.py`, not vice versa. Contains `tsCopulaExtremes`, `tsCopulaSampleJointPeaksMultiVariatePruning`, `tsCopulaFit`, `tsCopulaMontecarlo`, `tsCopulaRnd`, `tsGumbelCVine` (trivariate C-vine), `tsRankmax`, `tsPseudoObservations`, `tsCopulaCdfFromSamples`, `tsCopulaGOFNonStat`, `tsCopulaComputeBivarRP`.

## Test Structure

- **`test_monovariate/`** — Example scripts demonstrating monovariate EVA workflows. Each script loads data from `./data/*.csv` relative to its own directory.
- **`test_multivariate/`** — Case studies (01: river discharge + SWH GPD, 02: trivariate SWH Gumbel, 03: SPEI + temperature GEV). Data in `./data/caseStudy*_data.mat`.

## Running Tests

Scripts must be run from their own directory due to relative data paths:

```powershell
Push-Location d:/src/git/tsEva-py/test_monovariate
$env:MPLBACKEND='Agg'  # non-blocking backend for headless execution
& D:\Programs\Miniforge3-25.3.1\envs\Default\python.exe exampleGenerateSeriesEVAGraphs_ciPercentile.py
Pop-Location
```

For multivariate:

```powershell
Push-Location d:/src/git/tsEva-py/test_multivariate
$env:MPLBACKEND='Agg'
& D:\Programs\Miniforge3-25.3.1\envs\Default\python.exe caseStudy01.py
Pop-Location
```

## Key Conventions

### Timestamps
MATLAB datenum format: ordinal day + fractional day + 366 offset. Conversion functions: `datetime_to_datenum(dt)` for single datetime, `tsEvaPandasDate2DateNum(dates)` for pandas Series/Index.

### Distribution Parameter Sign Convention
- **MATLAB GEV**: shape parameter ε (epsilon)
- **scipy.stats.genextreme**: c = -ε (negative of MATLAB convention)

When reading/writing parameters, always check which convention is in use. The codebase uses `c_scipy = -epsilon_MATLAB` consistently.

### GPD Parameters
GPD uses threshold-based parameterization: shape (ξ), scale (σ), threshold (u). scipy's `genpareto` has loc=threshold by default — no sign flip needed unlike GEV.

## Dependencies

- numpy, pandas, scipy (stats, optimize, integrate, signal)
- matplotlib for plotting
- Python 3.x with Miniforge environment at `D:\Programs\Miniforge3-25.3.1\envs\Default`

## Common Pitfalls

1. **Working directory matters** — test scripts use relative paths to data files. Always run from the script's own directory.
2. **GEV sign convention** — forgetting that scipy uses c = -ε will produce incorrect fits and return levels.
3. **Bootstrap CI computation** — `tsEva.py` uses percentile-based CIs (32nd/68th percentiles of bootstrap samples), not standard error propagation. Don't confuse with parametric methods.

## Reference Implementation

MATLAB code at `D:\src\git\tsEva` is the ground truth for algorithmic behavior. When in doubt about numerical results, compare against MATLAB output using identical inputs and parameters.
