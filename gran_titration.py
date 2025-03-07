
"""
gran_titration.py

A Python library for performing Gran titration analysis on geochemical titration data stored in an xarray.Dataset.
The dataset is expected to have the following variables and dimensions:

Dimensions:
    sample       (e.g., 59 samples)
    measurement  (e.g., 53 titration points per sample)

Variables:
    pH(sample, measurement)           - measured pH values.
    moles_acid(sample, measurement)     - titrant added at each step (not cumulative).
    Sample volume(sample)               - initial sample volume (units stored in attrs, e.g., mL).
    Note(sample)                        - any notes per sample.

The Gran calculation uses the transformation:
    Y = (Sample volume + cumulative titrant) * 10^(-pH)

Near the equivalence point, the Y vs. cumulative titrant data are assumed linear.
By fitting (by default) the last n data points with a linear regression,
the equivalence point is estimated as the x-intercept (i.e., where Y = 0):
    V_e = -intercept / slope

This version adds an option to optimize n_points based on minimizing the 95% confidence
interval (CI) width on the estimated equivalence point. The output dictionary now includes the
optimized n_points ("n_points_opt") and the CI is computed using that optimal value.

The library provides:
  • gran_calculation_for_sample: Computes the equivalence point (with CI) for one sample,
     with an option to optimize the number of points.
  • calculate_equivalence_points: Applies that calculation to all samples in the dataset.
  • analyze_dataset: A top-level function that takes the dataset as an argument and returns an updated dataset.
  • plot_gran_analysis_ds: A function that accepts the dataset and one or more sample IDs and plots the Gran analysis for each.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import warnings
from typing import Tuple, Dict, Any
import scipy.stats as stats

def compute_Y(pH: np.ndarray, cumulative_titrant: np.ndarray, sample_volume: float) -> np.ndarray:
    """
    Compute the transformed variable for the Gran plot:
    
      Y = (sample_volume + cumulative_titrant) * 10^(-pH)
    
    Parameters:
        pH                 : 1D numpy array of pH values.
        cumulative_titrant : 1D numpy array of the cumulative titrant (moles) at each measurement.
        sample_volume      : Scalar initial sample volume.
        
    Returns:
        Y : 1D numpy array of transformed values.
    """
    return (sample_volume + cumulative_titrant) * np.power(10, -pH)

def fit_linear_region(x: np.ndarray, y: np.ndarray, n_points: int = 5) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a linear model (y = a*x + b) to the last n_points of the data arrays,
    after filtering out non-finite (NaN or Inf) values.
    
    Parameters:
        x        : 1D array of cumulative titrant values.
        y        : 1D array of transformed Y values.
        n_points : Number of points from the end of the valid data to use.
        
    Returns:
        slope      : Slope (a) of the fitted line.
        intercept  : Intercept (b) of the fitted line.
        x_fit      : x values used for the fit.
        y_fit_line : Fitted y values on the selected region.
        cov        : Covariance matrix (2x2) from the regression.
    """
    valid_mask = np.isfinite(x) & np.isfinite(y)
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    if len(x_valid) < n_points:
        raise ValueError("Not enough valid data points to fit: found {}, need {}".format(len(x_valid), n_points))
    x_fit = x_valid[-n_points:]
    y_fit = y_valid[-n_points:]
    coeffs, cov = np.polyfit(x_fit, y_fit, 1, cov=True)
    slope, intercept = coeffs
    y_fit_line = np.polyval(coeffs, x_fit)
    return slope, intercept, x_fit, y_fit_line, cov

def optimize_n_points_for_sample(pH: np.ndarray, titrant: np.ndarray, sample_volume: float, min_n: int = 3) -> Tuple[int, Dict[str, Any]]:
    """
    Optimize the number of points (n_points) used for regression by minimizing the 95% CI width
    on the equivalence point estimate.
    
    Parameters:
        pH            : 1D numpy array of pH values for the sample.
        titrant       : 1D numpy array of titrant added at each step.
        sample_volume : Scalar sample volume.
        min_n         : Minimum number of points to use (must be at least 3).
        
    Returns:
        best_n       : The optimal number of points.
        best_result  : A dictionary containing regression details for the optimal n_points.
    """
    cumulative_titrant = np.cumsum(titrant)
    Y = compute_Y(pH, cumulative_titrant, sample_volume)
    valid_mask = np.isfinite(cumulative_titrant) & np.isfinite(Y)
    x_valid = cumulative_titrant[valid_mask]
    y_valid = Y[valid_mask]
    if len(x_valid) < min_n:
        raise ValueError("Not enough valid data points to optimize: found {}, need at least {}".format(len(x_valid), min_n))
    
    best_n = None
    best_CI_width = np.inf
    best_result = None
    # Try from min_n up to the total number of valid points.
    for n in range(min_n, len(x_valid)+1):
        try:
            slope, intercept, x_fit, y_fit_line, cov = fit_linear_region(cumulative_titrant, Y, n_points=n)
            eq_point = -intercept / slope
            # Propagate errors: x_e = -b/a, so:
            var_eq = ((intercept / slope**2)**2 * cov[0, 0] +
                      (1 / slope)**2 * cov[1, 1] -
                      2 * (intercept / slope**3) * cov[0, 1])
            se_eq = np.sqrt(var_eq) if var_eq >= 0 else np.nan
            df = n - 2 if n > 2 else 1
            t_val = stats.t.ppf(0.975, df=df)
            half_width = t_val * se_eq
            CI_width = 2 * half_width
        except Exception as e:
            CI_width = np.inf
        if CI_width < best_CI_width:
            best_CI_width = CI_width
            best_n = n
            best_result = {
                "n_points_opt": n,
                "equivalence_point": eq_point,
                "CI_half_width": half_width,
                "CI_width": CI_width,
                "slope": slope,
                "intercept": intercept,
                "x_fit": x_fit,
                "y_fit_line": y_fit_line,
                "cov": cov,
            }
    return best_n, best_result

def gran_calculation_for_sample(pH: np.ndarray, titrant: np.ndarray, sample_volume: float, n_points: int = 5, optimize: bool = False) -> Dict[str, Any]:
    """
    Perform the Gran titration calculation for a single sample.
    
    Parameters:
        pH            : 1D numpy array of pH values for the sample.
        titrant       : 1D numpy array of titrant amounts added at each step (not cumulative).
        sample_volume : Scalar sample volume.
        n_points      : Number of points to use for linear regression (if not optimizing).
        optimize      : If True, optimize the number of points to minimize the CI width.
        
    Returns a dictionary with:
        'equivalence_point'    : The estimated equivalence point (cumulative titrant where Y=0).
        'equivalence_point_ci' : A 2-element array with lower and upper bounds of the 95% confidence interval.
        'slope'                : Slope of the fitted line.
        'intercept'            : Intercept of the fitted line.
        'x_fit'                : x values used for the fit.
        'y_fit_line'           : Fitted y values on the regression region.
        'cumulative_titrant'   : The cumulative sum of titrant added at each measurement.
        'Y'                    : Transformed Y array.
        'n_points_used'        : The number of points used (optimized if optimize=True).
    """
    cumulative_titrant = np.cumsum(titrant)
    Y = compute_Y(pH, cumulative_titrant, sample_volume)
    if optimize:
        best_n, best_result = optimize_n_points_for_sample(pH, titrant, sample_volume, min_n=3)
        result = best_result
    else:
        try:
            slope, intercept, x_fit, y_fit_line, cov = fit_linear_region(cumulative_titrant, Y, n_points=n_points)
            eq_point = -intercept / slope
            var_eq = ((intercept / slope**2)**2 * cov[0, 0] +
                      (1 / slope)**2 * cov[1, 1] -
                      2 * (intercept / slope**3) * cov[0, 1])
            se_eq = np.sqrt(var_eq) if var_eq >= 0 else np.nan
            df = n_points - 2 if n_points > 2 else 1
            t_val = stats.t.ppf(0.975, df=df)
            half_width = t_val * se_eq
            eq_ci = np.array([eq_point - half_width, eq_point + half_width])
            result = {
                "equivalence_point": eq_point,
                "equivalence_point_ci": eq_ci,
                "slope": slope,
                "intercept": intercept,
                "x_fit": x_fit,
                "y_fit_line": y_fit_line,
                "cov": cov,
                "n_points_opt": n_points
            }
        except Exception as e:
            result = {
                "equivalence_point": np.nan,
                "equivalence_point_ci": np.array([np.nan, np.nan]),
                "slope": np.nan,
                "intercept": np.nan,
                "x_fit": np.array([]),
                "y_fit_line": np.array([]),
                "cov": np.array([]),
                "n_points_opt": np.nan
            }
            warnings.warn(f"Error in sample calculation: {e}")
    # Compute CI from the result (if not already computed)
    if "equivalence_point_ci" not in result:
        half_width = result.get("CI_half_width", np.nan)
        eq_point = result.get("equivalence_point", np.nan)
        result["equivalence_point_ci"] = np.array([eq_point - half_width, eq_point + half_width])
    result["cumulative_titrant"] = cumulative_titrant
    result["Y"] = Y
    return result

def calculate_equivalence_points(ds: xr.Dataset, n_points: int = 5, optimize: bool = False) -> xr.Dataset:
    """
    Apply the Gran titration calculation to every sample in the Dataset.
    
    The Dataset must have the following variables:
      - "pH"           with dimensions (sample, measurement)
      - "moles_acid"   with dimensions (sample, measurement)
      - "Sample volume" with dimension (sample)
      
    Returns:
        A new Dataset with added variables:
         - "equivalence_point" (dimension sample)
         - "equivalence_point_ci" (dimensions sample x ci_bound, where ci_bound has size 2)
         - "n_points_used" (dimension sample)
        containing the calculated equivalence point, its confidence interval, and the number of points used for each sample.
    """
    eq_points = []
    eq_cis = []
    n_points_used = []
    for sample in ds.sample.values:
        pH_sample = ds["pH"].sel(sample=sample).values
        titrant_sample = ds["moles_acid"].sel(sample=sample).values
        sample_vol = ds["Sample volume"].sel(sample=sample).values.item()
        result = gran_calculation_for_sample(pH_sample, titrant_sample, sample_vol, n_points=n_points, optimize=optimize)
        eq_points.append(result["equivalence_point"])
        eq_cis.append(result["equivalence_point_ci"])
        n_points_used.append(result["n_points_opt"])
    
    eq_da = xr.DataArray(eq_points, coords={"sample": ds.sample.values}, dims=["sample"], name="equivalence_point")
    eq_da.attrs["units"] = "same as moles_acid (cumulative)"
    
    eq_ci_da = xr.DataArray(np.array(eq_cis), coords={"sample": ds.sample.values, "ci_bound": ["lower", "upper"]}, dims=["sample", "ci_bound"], name="equivalence_point_ci")
    eq_ci_da.attrs["units"] = "same as moles_acid (cumulative)"
    
    n_points_da = xr.DataArray(n_points_used, coords={"sample": ds.sample.values}, dims=["sample"], name="n_points_used")
    
    ds_out = ds.copy()
    ds_out["equivalence_point"] = eq_da
    ds_out["equivalence_point_ci"] = eq_ci_da
    ds_out["n_points_used"] = n_points_da
    return ds_out

def analyze_dataset(ds: xr.Dataset, n_points: int = 5, optimize: bool = False) -> xr.Dataset:
    """
    Top-level function to perform the Gran titration analysis on the entire dataset.
    
    Parameters:
        ds       : xarray.Dataset containing the titration data.
        n_points : Number of points to use for the linear regression if not optimizing (default is 5).
        optimize : If True, the algorithm will search over possible n_points to minimize the CI width.
        
    Returns:
        An updated xarray.Dataset with added variables "equivalence_point", "equivalence_point_ci", and "n_points_used" for each sample.
    """
    return calculate_equivalence_points(ds, n_points=n_points, optimize=optimize)

def plot_gran_analysis_ds(ds: xr.Dataset, sample_ids, n_points: int = 5, optimize: bool = False):
    """
    Plot the Gran titration analysis for one or more samples from the dataset.
    
    Parameters:
        ds         : xarray.Dataset containing the titration data.
        sample_ids : A single sample ID (string) or a list of sample IDs to plot.
        n_points   : Number of points from the end to use for linear regression (if not optimizing).
        optimize   : If True, the plot uses the optimized n_points.
        
    For each sample, the function creates a plot showing:
        - Scatter plot of cumulative titrant vs. transformed Y = (Sample volume + cumulative titrant) * 10^(-pH)
        - The linear regression fit (using the last n_points or optimized n_points)
        - A vertical dashed line marking the equivalence point with its confidence interval shown as a shaded area.
    """
    if isinstance(sample_ids, str):
        sample_ids = [sample_ids]
    
    num_samples = len(sample_ids)
    fig, axs = plt.subplots(num_samples, 1, figsize=(8, 6 * num_samples)) if num_samples > 1 else (plt.gcf(), [plt.gca()])
    
    for i, sample in enumerate(sample_ids):
        pH_vals = ds["pH"].sel(sample=sample).values
        titrant_vals = ds["moles_acid"].sel(sample=sample).values
        sample_vol = ds["Sample volume"].sel(sample=sample).values.item()
        result = gran_calculation_for_sample(pH_vals, titrant_vals, sample_vol, n_points=n_points, optimize=optimize)
        eq_point = result["equivalence_point"]
        eq_ci = result["equivalence_point_ci"]
        Y = result["Y"]
        cumulative_titrant = result["cumulative_titrant"]
        
        ax = axs[i]
        ax.scatter(cumulative_titrant, Y, label="Transformed Data", color="blue")
        if result["x_fit"].size > 0:
            ax.plot(result["x_fit"], result["y_fit_line"], label=f"Linear Fit (last {result['n_points_opt']} pts)", color="red")
        ax.axvline(x=eq_point, linestyle="--", color="green", label=f"Equivalence: {eq_point:.4f}")
        ax.fill_betweenx(ax.get_ylim(), eq_ci[0], eq_ci[1], color="green", alpha=0.2, label="95% CI")
        ax.set_xlabel("Cumulative Titrant (moles)")
        ax.set_ylabel("Transformed Y = (Sample volume + cumulative titrant) * 10^(-pH)")
        ax.set_title(f"Gran Plot Analysis for Sample {sample}")
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()