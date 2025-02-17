# Titration Data Analysis Tool

## Overview
This Python project provides a framework for processing, analyzing, and visualizing titration data. It extracts titration measurements from structured text files, converts them into an `xarray.Dataset`, performs logistic curve fitting to determine inflection points, and visualizes the results using `matplotlib`.

## Features
- Parses titration data from structured text files.
- Constructs `xarray.Dataset` objects for convenient data handling.
- Computes moles of acid added and stores metadata with appropriate units.
- Fits logistic models to titration curves for determining inflection points.
- Generates titration curves with inflection points highlighted.

## File Descriptions

- `dic.py`: Contains functions for computing dissolved inorganic carbon (DIC) from titration data.
- `plotting.py`: Provides functions for plotting titration curves, showing cumulative acid addition versus pH.
- `processing.ipynb`: A Jupyter Notebook for processing raw titration data and generating structured datasets.
- `dic_calc.ipynb`: A Jupyter Notebook that performs calculations related to dissolved inorganic carbon.
- `titration.py`: Implements functions for parsing titration data, constructing `xarray.Dataset` objects, and performing additional computations.

## Installation

Ensure you have Python installed, along with the required dependencies:

```sh
pip install numpy matplotlib xarray
```

## Usage

### Parsing Titration Data

```python
from titration import parse_titration_data, build_xarray_dataset

with open('data/titration_data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

samples = parse_titration_data(text)
ds = build_xarray_dataset(samples)
```

### Computing Moles of Acid

```python
from titration import add_moles_variable

ds = add_moles_variable(ds)
```

### Prepending Initial pH

```python
from titration import prepend_initial_ph

ds = prepend_initial_ph(ds)
```

### Plotting Titration Curves

```python
from plotting import plot_titration_curves

plot_titration_curves(ds)
```

## License
This project is open-source and available under the MIT License.


