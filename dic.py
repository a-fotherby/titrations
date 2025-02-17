import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import xarray as xr

# ---------------------------------------------------
# Logistic Model Functions
# ---------------------------------------------------

def decreasing_logistic(x, A, K, B, x0):
    """
    Single decreasing logistic function.
    
    pH(x) = K + (A - K) / (1 + exp(-B*(x - x0)))
    
    Parameters:
      x : array_like
          Independent variable.
      A : float
          Lower plateau value.
      K : float
          Upper plateau value.
      B : float
          Steepness of the transition.
      x0 : float
          Inflection point.
    
    Returns:
      pH values computed from the logistic model.
    """
    return K + (A - K) / (1 + np.exp(-B * (x - x0)))


def double_decreasing_logistic_reparam(x, K, A1, A2, B1, B2, x01, dx):
    """
    Reparameterized double logistic function.
    
    Models two transitions with:
      - First inflection point: x01
      - Second inflection point: x02 = x01 + dx
    
    pH(x) = K + (A1 - K)/(1 + exp(-B1*(x - x01))) +
                 (A2 - A1)/(1 + exp(-B2*(x - (x01+dx))))
    
    Parameters:
      x : array_like
          Independent variable.
      K, A1, A2 : floats
          Plateau parameters.
      B1, B2 : floats
          Steepness parameters.
      x01 : float
          First inflection point.
      dx : float
          Difference such that x02 = x01 + dx.
    
    Returns:
      pH values computed from the double logistic model.
    """
    x02 = x01 + dx
    term1 = (A1 - K) / (1 + np.exp(-B1 * (x - x01)))
    term2 = (A2 - A1) / (1 + np.exp(-B2 * (x - x02)))
    return K + term1 + term2

# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------

def get_valid_data(x_data, y_data):
    """
    Extracts the valid (non-NaN) portion of the data, removing any leading or trailing NaNs.
    
    Parameters:
      x_data : array_like
          Raw x data.
      y_data : array_like
          Raw y data.
          
    Returns:
      x_valid, y_valid, x_min, x_max, x_range
      or (None, None, None, None, None) if no valid data exists.
    """
    valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
    valid_indices = np.where(valid_mask)[0]
    if valid_indices.size == 0:
        return None, None, None, None, None
    first_valid = valid_indices[0]
    last_valid = valid_indices[-1]
    x_valid = x_data[first_valid:last_valid+1]
    y_valid = y_data[first_valid:last_valid+1]
    x_min = np.min(x_valid)
    x_max = np.max(x_valid)
    x_range = x_max - x_min
    return x_valid, y_valid, x_min, x_max, x_range


def fit_single_logistic(x_valid, y_valid, x_min, x_max, x_range):
    """
    Fits the single logistic model to the valid data.
    
    Parameters:
      x_valid, y_valid : array_like
          Clean data used for fitting.
      x_min, x_max, x_range : floats
          Minimum, maximum, and range of x_valid.
    
    Returns:
      popt, pcov: Optimal parameters and covariance matrix.
    """
    A0 = np.min(y_valid)
    K0 = np.max(y_valid)
    x0_initial = (x_min + x_max) / 2.0
    B0 = 4.0 / x_range if x_range > 0 else 1.0
    p0 = [A0, K0, B0, x0_initial]
    lower_bounds = [A0 - 1, K0 - 1, 0.1 * B0, x_min - x_range]
    upper_bounds = [A0 + 1, K0 + 1, 10 * B0, x_max + x_range]
    
    popt, pcov = curve_fit(decreasing_logistic, x_valid, y_valid, p0=p0,
                           bounds=(lower_bounds, upper_bounds))
    return popt, pcov


def fit_double_logistic(x_valid, y_valid, x_min, x_max, x_range):
    """
    Fits the reparameterized double logistic model to the valid data.
    
    Parameters:
      x_valid, y_valid : array_like
          Clean data used for fitting.
      x_min, x_max, x_range : floats
          Minimum, maximum, and range of x_valid.
          
    Returns:
      popt, pcov: Optimal parameters and covariance matrix.
    """
    K0 = np.max(y_valid)
    A2_0 = np.min(y_valid)
    A1_0 = (K0 + A2_0) / 2.0
    x01_0 = x_min + x_range / 3.0
    x02_0 = x_min + 2 * x_range / 3.0
    dx0 = x02_0 - x01_0
    B0 = 4.0 / x_range if x_range > 0 else 1.0
    B1_0 = B0
    B2_0 = B0
    
    min_dx = 0.1 * x_range if x_range > 0 else 0.0
    
    p0 = [K0, A1_0, A2_0, B1_0, B2_0, x01_0, dx0]
    lower_bounds = [K0 - 1, A1_0 - 1, A2_0 - 1,
                    0.1 * B0, 0.1 * B0, x_min, min_dx]
    upper_bounds = [K0 + 1, A1_0 + 1, A2_0 + 1,
                    10 * B0, 10 * B0, x_max, x_range]
    
    popt, pcov = curve_fit(double_decreasing_logistic_reparam, x_valid, y_valid,
                           p0=p0, bounds=(lower_bounds, upper_bounds))
    return popt, pcov


def compute_errors_single(pcov):
    """
    Computes the standard error for the inflection point in the single logistic model.
    The inflection point is parameter index 3 (x0).
    
    Parameters:
      pcov : 2D array
          Covariance matrix from the fit.
          
    Returns:
      se_x0 : float
          Standard error of the inflection point.
    """
    return np.sqrt(np.diag(pcov))[3]


def compute_errors_double(pcov):
    """
    Computes the standard errors for the inflection points in the reparameterized double logistic model.
    
    Parameters:
      pcov : 2D array
          Covariance matrix from the fit.
          
    Returns:
      se_x01 : float
          Standard error for the first inflection point (x01).
      se_x2 : float
          Standard error for the second inflection point, computed by error propagation on
          x2 = x01 + dx.
    """
    diag = np.sqrt(np.diag(pcov))
    se_x01 = diag[5]  # error on x01
    se_dx = diag[6]   # error on dx
    cov_x01_dx = pcov[5, 6]
    var_x2 = (se_x01 ** 2) + (se_dx ** 2) + 2 * cov_x01_dx
    se_x2 = np.sqrt(var_x2) if var_x2 > 0 else np.nan
    return se_x01, se_x2

# ---------------------------------------------------
# Main Plotting and Fitting Function
# ---------------------------------------------------

def plot_data_and_fit(ds, force_double_samples=None):
    """
    For each sample in the dataset, fits a logistic model (single or double as specified),
    computes the uncertainty on the inflection point(s) using the covariance matrix, and plots the results.
    
    For double logistic fits, two inflection points are determined:
      - inflection1: x01 (with standard error from parameter index 5)
      - inflection2: x02 = x01 + dx (with error propagated from indices 5 and 6)
    For single logistic fits, the inflection point is x0 (parameter index 3).
    
    The function also stores the inflection point coordinates and their errors in the dataset.
    
    Parameters:
      ds : xarray.Dataset
          Dataset with coordinates 'sample' and 'measurement', and variables 'moles_acid' and 'pH'.
      force_double_samples : list or None
          List of sample labels that should be fitted using the double logistic model.
    
    Returns:
      ds : xarray.Dataset
          Updated dataset with new variables for inflection point coordinates and errors.
    """
    if force_double_samples is None:
        force_double_samples = []
    
    # Lists for storing inflection point coordinates and errors.
    inflection1_x_list = []
    inflection1_y_list = []
    inflection2_x_list = []
    inflection2_y_list = []
    inflection1_error_list = []
    inflection2_error_list = []
    
    # Process each sample.
    for sample in ds.coords['sample'].values:
        # --- Extract and clean data ---
        sample_data = ds.sel(sample=sample)
        x_data = sample_data['moles_acid'].cumsum(dim='measurement').values
        y_data = sample_data['pH'].values
        result = get_valid_data(x_data, y_data)
        if result[0] is None:
            print(f"No valid data for sample {sample}. Skipping.")
            inflection1_x_list.append(np.nan)
            inflection1_y_list.append(np.nan)
            inflection2_x_list.append(np.nan)
            inflection2_y_list.append(np.nan)
            inflection1_error_list.append(np.nan)
            inflection2_error_list.append(np.nan)
            continue
        x_valid, y_valid, x_min, x_max, x_range = result
        
        # Determine which model to use.
        use_double = sample in force_double_samples
        fit_type = None
        err_inf1 = np.nan  # error for the first (or only) inflection point.
        err_inf2 = np.nan  # error for the second inflection point (if applicable).
        
        # --- Fit Double Logistic Model ---
        if use_double:
            try:
                popt, pcov = fit_double_logistic(x_valid, y_valid, x_min, x_max, x_range)
                # Unpack parameters:
                K_fit, A1_fit, A2_fit, B1_fit, B2_fit, x01_fit, dx_fit = popt
                x02_fit = x01_fit + dx_fit
                fit_type = "double"
                # Compute standard errors on inflection points.
                err_inf1, err_inf2 = compute_errors_double(pcov)
            except Exception as e:
                print(f"Double logistic fit failed for sample {sample}: {e}. Falling back to single logistic.")
                use_double = False
        
        # --- Fit Single Logistic Model (if double fit not used) ---
        if not use_double:
            try:
                popt, pcov = fit_single_logistic(x_valid, y_valid, x_min, x_max, x_range)
                A_fit, K_fit, B_fit, x0_fit = popt
                fit_type = "single"
                err_inf1 = compute_errors_single(pcov)
            except Exception as e:
                print(f"Single logistic fit failed for sample {sample}: {e}")
                fit_type = None
        
        # --- Compute Fitted Curve and Inflection Points ---
        if fit_type == "double":
            x_fit = np.linspace(x_min, x_max, 200)
            y_fit = double_decreasing_logistic_reparam(x_fit, K_fit, A1_fit, A2_fit, 
                                                       B1_fit, B2_fit, x01_fit, dx_fit)
            # First inflection point.
            inflection1_x = x01_fit
            inflection1_y = double_decreasing_logistic_reparam(x01_fit, K_fit, A1_fit, A2_fit, 
                                                               B1_fit, B2_fit, x01_fit, dx_fit)
            # Second inflection point.
            inflection2_x = x02_fit
            inflection2_y = double_decreasing_logistic_reparam(x02_fit, K_fit, A1_fit, A2_fit, 
                                                               B1_fit, B2_fit, x01_fit, dx_fit)
        elif fit_type == "single":
            x_fit = np.linspace(x_min, x_max, 200)
            y_fit = decreasing_logistic(x_fit, A_fit, K_fit, B_fit, x0_fit)
            inflection1_x = x0_fit
            inflection1_y = decreasing_logistic(x0_fit, A_fit, K_fit, B_fit, x0_fit)
            inflection2_x = np.nan
            inflection2_y = np.nan
        else:
            x_fit = None
            inflection1_x = np.nan
            inflection1_y = np.nan
            inflection2_x = np.nan
            inflection2_y = np.nan
        
        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x_valid, y_valid, 'o', label='Data', markersize=6)
        if x_fit is not None:
            ax.plot(x_fit, y_fit, '-', label='Fitted Curve', linewidth=2)
            if fit_type == "double":
                ax.plot(inflection1_x, inflection1_y, 'rx', markersize=10, label='Inflection 1')
                ax.plot(inflection2_x, inflection2_y, 'gx', markersize=10, label='Inflection 2')
                ax.errorbar(inflection1_x, inflection1_y, xerr=err_inf1, fmt='none', 
                            ecolor='r', elinewidth=2, capsize=4)
                ax.errorbar(inflection2_x, inflection2_y, xerr=err_inf2, fmt='none', 
                            ecolor='g', elinewidth=2, capsize=4)
            elif fit_type == "single":
                ax.plot(inflection1_x, inflection1_y, 'rx', markersize=10, label='Inflection')
                ax.errorbar(inflection1_x, inflection1_y, xerr=err_inf1, fmt='none', 
                            ecolor='r', elinewidth=2, capsize=4)
        ax.set_xlabel('Cumulative Acid')
        ax.set_ylabel('pH')
        ax.set_title(f"Sample: {sample} (Fit: {fit_type})")
        ax.grid(True)
        ax.legend()
        plt.show()
        
        # --- Store Inflection Point Coordinates and Errors ---
        inflection1_x_list.append(inflection1_x)
        inflection1_y_list.append(inflection1_y)
        inflection2_x_list.append(inflection2_x)
        inflection2_y_list.append(inflection2_y)
        inflection1_error_list.append(err_inf1)
        inflection2_error_list.append(err_inf2)
    
    # Add computed inflection point coordinates and their errors to the dataset.
    ds = ds.assign(
        inflection1_x=(('sample',), np.array(inflection1_x_list)),
        inflection1_y=(('sample',), np.array(inflection1_y_list)),
        inflection2_x=(('sample',), np.array(inflection2_x_list)),
        inflection2_y=(('sample',), np.array(inflection2_y_list)),
        inflection1_error=(('sample',), np.array(inflection1_error_list)),
        inflection2_error=(('sample',), np.array(inflection2_error_list))
    )
    
    return ds
