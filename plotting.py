import matplotlib.pyplot as plt

def plot_titration_curves(ds):
    """
    Plots titration curves from an xarray Dataset using cumulative acid values.
    
    The dataset is expected to have:
      - Two coordinates: 'sample' and 'measurement'
      - Two data variables: 'acid' and 'pH'
      
    For each sample, a separate plot is generated with the cumulative sum of 'acid'
    on the x-axis and 'pH' on the y-axis.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset containing titration data.
    """
    # Loop over each sample in the dataset
    for sample in ds.coords['sample'].values:
        # Select data for the current sample along the 'measurement' dimension
        sample_data = ds.sel(sample=sample)
        
        # Compute the cumulative sum of acid along the 'measurement' dimension
        cumulative_acid = sample_data['moles_acid'].cumsum(dim='measurement').values
        
        # Extract the pH data (assumed to be aligned with the measurements)
        pH = sample_data['pH'].values
        
        # Create a new figure for the current sample
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(cumulative_acid, pH, marker='o', linestyle='-', color='b')
        
        # Label the axes and the plot
        ax.set_xlabel('Cumulative Acid / mol')
        ax.set_ylabel('pH')
        ax.set_title(f"Titration Curve for Sample: {sample}")
        ax.grid(True)
        
        # Display the plot
        return fig