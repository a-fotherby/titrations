#!/usr/bin/env python
"""
This script reads titration data from a text file and creates an xarray.Dataset.
Any units embedded in metadata keys or in the titration data header are extracted
and stored as variable attributes under the key 'units'.

The expected input file format is:

    ### Numeric Samples (sorted by number)
    
    #### Sample 1
    Sample ID: 1
    Initial pH: 6.55
    Sample volume (mL): 30
    Acid (µL), pH
    200,6.58
    200,6.57
    ... (more data)
    -----------------------------------
    
    ... (more samples)
    
    ### Alphanumeric Samples (sorted alphabetically)
    
    #### Sample “8 / Decean5”
    Sample ID: 8 / Decean5
    Initial pH: 8.60
    Sample volume (mL): 30
    Note: Some headspace might have degassed
    Acid (µL), pH
    200,8.37
    1000,7.25
    ... (more data)
    -----------------------------------
    
For any metadata line with a key like "Key (unit): Value", the base key ("Key")
is used as the variable name and the unit is stored in that variable’s attrs.
Likewise, the header line for titration data (e.g. "Acid (µL), pH")
is parsed so that the acid values get an attribute of units (here, "µL").
"""

import re
import numpy as np
import xarray as xr

def parse_titration_data(text):
    """
    Parse the input text and return a list of dictionaries (one per sample).
    Each dictionary will include:
      - Metadata (with any units extracted into separate keys with suffix "_unit")
      - Two keys "acid" and "pH" holding lists of titration measurements.
    """
    # Split the text into sample blocks. The first block (before the first "#### Sample")
    # is ignored.
    blocks = re.split(r'(?m)^####\s*Sample', text)
    samples = []
    for block in blocks[1:]:
        # Prepend header token back.
        block = "#### Sample" + block
        lines = block.strip().splitlines()
        sample_meta = {}
        data_lines = []
        
        # Process the header line (e.g. "#### Sample 1" or "#### Sample “8 / Decean5”")
        header_line = lines[0].strip()
        m = re.match(r'####\s*Sample\s*(.*)', header_line)
        if m:
            sample_meta['sample_id'] = m.group(1).strip()
        else:
            sample_meta['sample_id'] = header_line

        in_data = False
        for line in lines[1:]:
            line = line.strip()
            if line.startswith('---'):
                continue

            # Look for the titration data header line.
            # This header is expected to be like: "Acid (µL), pH" (or possibly with units for pH too).
            match = re.match(r'^Acid\s*\(([^)]+)\),\s*pH(?:\s*\(([^)]+)\))?', line)
            if match:
                acid_unit = match.group(1).strip()
                sample_meta["acid_unit"] = acid_unit
                # Optionally, capture pH unit if provided.
                ph_unit = match.group(2).strip() if match.lastindex >= 2 and match.group(2) else None
                if ph_unit:
                    sample_meta["pH_unit"] = ph_unit
                in_data = True
                continue

            if not in_data:
                # Process a metadata line in the form "Key: Value"
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    # If the key contains units (e.g., "Sample volume (mL)"), extract them.
                    m_unit = re.match(r'^(.*?)\s*\(([^)]+)\)$', key)
                    if m_unit:
                        base_key = m_unit.group(1).strip()
                        unit = m_unit.group(2).strip()
                        sample_meta[base_key] = value
                        sample_meta[base_key + "_unit"] = unit
                    else:
                        sample_meta[key] = value
            else:
                # In the data section: each line should have two comma-separated numbers.
                if line:
                    data_lines.append(line)
        
        # Process the titration data lines.
        acid_vals = []
        pH_vals = []
        for dline in data_lines:
            parts = dline.split(',')
            if len(parts) >= 2:
                try:
                    acid_vals.append(float(parts[0].strip()))
                except Exception:
                    acid_vals.append(np.nan)
                try:
                    pH_vals.append(float(parts[1].strip()))
                except Exception:
                    pH_vals.append(np.nan)
        sample_meta['acid'] = acid_vals
        sample_meta['pH'] = pH_vals
        samples.append(sample_meta)
    return samples

def build_xarray_dataset(samples):
    """
    Build an xarray.Dataset from the list of sample dictionaries.
    
    The dataset will include:
      - Two 2D variables: "acid" and "pH" (dimensions: sample x measurement),
      - For each metadata field (other than titration data), a 1D variable along the "sample" dimension.
      
    If a metadata key originally contained units (e.g. "Sample volume (mL)"),
    the unit is removed from the key name (stored as the base key) and attached as an attribute.
    For titration data, the acid (and optionally pH) units are attached as attributes.
    """
    n_samples = len(samples)
    # Determine the maximum number of titration measurements across samples.
    max_points = max(len(s['acid']) for s in samples)
    
    # Create rectangular 2D arrays for titration data, padding with NaN where needed.
    acid_arr = np.full((n_samples, max_points), np.nan, dtype=float)
    pH_arr = np.full((n_samples, max_points), np.nan, dtype=float)
    
    for i, s in enumerate(samples):
        n_points = len(s['acid'])
        acid_arr[i, :n_points] = s['acid']
        pH_arr[i, :n_points] = s['pH']
    
    # Identify metadata keys (exclude titration data and any keys ending with "_unit").
    meta_keys = set()
    for s in samples:
        for key in s.keys():
            if key not in ('acid', 'pH', 'acid_unit', 'pH_unit') and not key.endswith('_unit'):
                meta_keys.add(key)
    meta_keys = list(meta_keys)
    
    # Collect per-sample metadata and associated units.
    meta_data = {}
    meta_units = {}
    for key in meta_keys:
        values = []
        unit_val = None
        for s in samples:
            val = s.get(key, None)
            values.append(val)
            # If a corresponding unit was parsed (stored under key+"_unit"), use it.
            if unit_val is None and (key + "_unit") in s:
                unit_val = s.get(key + "_unit")
        meta_data[key] = values
        if unit_val is not None:
            meta_units[key] = unit_val
    
    # Create the xarray.Dataset.
    ds = xr.Dataset(
        data_vars={
            'acid': (('sample', 'measurement'), acid_arr),
            'pH': (('sample', 'measurement'), pH_arr)
        },
        coords={
            'sample': meta_data.get('sample_id', [f'sample_{i+1}' for i in range(n_samples)]),
            'measurement': np.arange(max_points)
        }
    )
    
    # Add metadata variables (as 1D arrays along the sample dimension).
    for key, values in meta_data.items():
        ds[key] = (('sample',), values)
        # If a unit was extracted, attach it as an attribute.
        if key in meta_units:
            ds[key].attrs['units'] = meta_units[key]
    
    # Attach titration data units if available (assumes they are consistent across samples).
    if 'acid_unit' in samples[0]:
        ds['acid'].attrs['units'] = samples[0]['acid_unit']
    if 'pH_unit' in samples[0]:
        ds['pH'].attrs['units'] = samples[0]['pH_unit']
    
    return ds


def add_moles_variable(ds, acid_concentration=0.0102):
    """
    Add a new variable 'moles_acid' to the xarray.Dataset based on the 'acid' variable.
    
    The 'acid' variable is assumed to be in microliters (µL). This function converts the 
    acid volume to liters and then multiplies by the acid concentration (default 0.0102 M)
    to compute the moles of acid added.
    
    Parameters:
        ds (xarray.Dataset): The dataset containing the 'acid' variable.
        acid_concentration (float): Acid concentration in mol/L (default: 0.0102).
        
    Returns:
        xarray.Dataset: The input dataset with the new variable 'moles_acid' added.
    """
    # Convert µL to L and calculate moles of acid:
    ds['moles_acid'] = (ds['acid'] / 1e6) * acid_concentration
    ds['moles_acid'].attrs['units'] = 'mol'
    ds['moles_acid'].attrs['description'] = (
        'Calculated moles of acid added: (acid in µL / 1e6) * acid concentration (mol/L)'
    )
    return ds


def prepend_initial_ph(ds):
    """
    Prepend the 'Initial pH' value to the 'pH' variable for each sample,
    with a corresponding 'moles_acid' value of 0.
    
    This version builds a new dataset with updated 'pH' and 'moles_acid'
    variables (with a new 'measurement' coordinate) and also includes
    variables that are aligned along the 'sample' coordinate only, such as
    'Sample volume' and 'Note'.
    
    Parameters
    ----------
    ds : xarray.Dataset
        The original dataset containing variables:
          - 'pH' and 'moles_acid': 2D arrays with dimensions ("sample", "measurement")
          - 'Initial pH': 1D array with dimension ("sample",)
          - Optionally, 'Sample volume' and 'Note': 1D arrays along ("sample",)
    
    Returns
    -------
    new_ds : xarray.Dataset
        A new dataset with the updated 'pH' and 'moles_acid' variables (the first
        measurement is now the 'Initial pH' and 0, respectively) and the extra sample-only
        variables.
    """
    # Extract the original arrays.
    pH = ds['pH'].values            # shape: (n_samples, n_measurements)
    moles = ds['moles_acid'].values  # shape: (n_samples, n_measurements)
    init_pH = ds['Initial pH'].values  # shape: (n_samples,)
    
    n_samples, n_measurements = pH.shape

    # For each sample, count the number of valid (non-NaN) measurements.
    valid_counts = []
    for i in range(n_samples):
        # If the very first pH value is NaN, count is 0.
        if np.isnan(pH[i, 0]):
            count = 0
        else:
            # Find the first index where NaN occurs (assumes valid measurements are contiguous).
            nan_inds = np.where(np.isnan(pH[i]))[0]
            count = nan_inds[0] if nan_inds.size > 0 else n_measurements
        valid_counts.append(count)
    
    # Each sample’s new measurement length is (number of valid measurements + 1).
    new_lengths = np.array(valid_counts) + 1
    new_meas_len = new_lengths.max()

    # Create new arrays prefilled with NaN.
    new_pH = np.full((n_samples, new_meas_len), np.nan)
    new_moles = np.full((n_samples, new_meas_len), np.nan)

    # For each sample, place the 'Initial pH' and 0 (for moles_acid) at index 0,
    # and then copy the valid measurements.
    for i in range(n_samples):
        count = valid_counts[i]
        new_pH[i, 0] = init_pH[i]
        new_moles[i, 0] = 0
        if count > 0:
            new_pH[i, 1:count+1] = pH[i, :count]
            new_moles[i, 1:count+1] = moles[i, :count]

    # Define the new 'measurement' coordinate.
    new_measurement = np.arange(new_meas_len)

    # Create new DataArrays for the updated 2D variables.
    new_pH_da = xr.DataArray(
        new_pH,
        dims=("sample", "measurement"),
        coords={"sample": ds["sample"], "measurement": new_measurement},
        name="pH"
    )
    new_moles_da = xr.DataArray(
        new_moles,
        dims=("sample", "measurement"),
        coords={"sample": ds["sample"], "measurement": new_measurement},
        name="moles_acid"
    )

    # Build a new dataset including the updated variables.
    new_ds = xr.Dataset(
        {
            "pH": new_pH_da,
            "moles_acid": new_moles_da,
            "Initial pH": ds["Initial pH"]
        },
        coords={
            "sample": ds["sample"],
            "measurement": new_measurement  # new measurement coordinate
        }
    )

    # Bring along additional variables that are aligned along the 'sample' coordinate.
    for var in ["Sample volume", "Note"]:
        if var in ds.data_vars:
            new_ds[var] = ds[var]

    new_ds.drop_vars('Initial pH')

    return new_ds
