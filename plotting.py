import matplotlib.pyplot as plt

def plot_dic_bar_chart(ds):
    """
    Plot a bar chart of the 'dic' variable from an xarray Dataset along the 'sample' coordinate,
    including error bars from the 'dic_error' variable.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset containing the variables 'dic', 'dic_error', and the coordinate 'sample'.
    """
    # Extract data for plotting
    samples = ds['sample'].values
    dic_values = ds['dic'].values
    dic_errors = ds['dic_error'].values

    # Create the bar chart with error bars
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(samples, dic_values, yerr=dic_errors, capsize=5, color='skyblue', edgecolor='black')
    
    # Label the axes and set a title
    ax.set_xlabel('Sample')
    ax.set_ylabel('DIC')
    ax.set_title('DIC Measurements with Error Bars')
    
    # Optionally, rotate the x-axis labels if needed
    plt.xticks(rotation=45)
    
    # Display the plot
    plt.tight_layout()
    plt.show()
