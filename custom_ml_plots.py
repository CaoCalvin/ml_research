from typing import Optional
import pandas as pd
import numpy as np
import ml_data_objects as mdo


# Plotting
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

from scipy.stats import pearsonr

from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def plot_prediction_vs_actual(axs: np.ndarray, row: int, col: int, predicted: pd.DataFrame, actual: pd.DataFrame, 
                                predicted_top: Optional[pd.DataFrame] = None, actual_top: Optional[pd.DataFrame] = None):
    """Creates a scatter plot at the specified 2D axis position [row, col].

    Created 09/19/2024
    
    Args:
        axs: 2d array of matplotlib axes
        row: Integer representing row index 
        col: Integer representing the column 
        predicted: DataFrame with predicted values
        actual: DataFrame with actual values
        predicted_top: Optional DataFrame containing binary boolean classifications for whether a value is a "top" value or not
        actual_top: Optional DataFrame containing binary boolean classifications for whether a value is a "top" value or not
    """
    # Determine colors for points based on provided top classifications
    if predicted_top is not None and actual_top is not None:
        colors = [
            'green' if pt and at else 'red' if pt and not at else 'blue' if not pt and at else 'grey'
            for pt, at in zip(predicted_top.iloc[:, 0], actual_top.iloc[:, 0])
        ]
    elif predicted_top is not None:
        colors = ['green' if pt else 'grey' for pt in predicted_top.iloc[:, 0]]
    elif actual_top is not None:
        colors = ['green' if at else 'grey' for at in actual_top.iloc[:, 0]]
    else:
        colors = 'blue'  # Default color if no binary classifications provided

    # Scatter plot
    sc = axs[row, col].scatter(actual, predicted, c=colors, alpha=0.5)
    axs[row, col].set_xlabel("Actual values")
    axs[row, col].set_ylabel("Predicted values")
    axs[row, col].set_title(actual.iloc[:, 0].name)
    
    # Plot y=x theoretical perfect line
    axs[row, col].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')

    # Show Pearson coefficient
    pearson_coef, _ = pearsonr(actual.iloc[:, 0], predicted.iloc[:, 0])
    axs[row, col].text(0.05, 0.95, f'Pearson Coefficient = {pearson_coef:.2f}', transform=axs[row, col].transAxes, fontsize=10, verticalalignment='top')
    
def plot_classification_results(predictions: pd.DataFrame, actuals: pd.DataFrame, 
                                predictions_top: Optional[pd.DataFrame] = None, actuals_top: Optional[pd.DataFrame] = None):
    """Plots all predictions of a multi-output ML model in a series of scatter plots in one row.

    Created 2024/09/20

    Args:
        predictions (pd.DataFrame): ML model predictions (can be one or multiple columns)
        actuals (pd.DataFrame): Actual values corresponding to predictions 
        predictions_top (pd.DataFrame, optional): Which predictions are considered "top" to highlight. Defaults to None.
        actuals_top (pd.DataFrame, optional): Which actual values are considered "top" to highlight. Defaults to None.
    """
    # make sure  predictions and actuals have the same shape
    assert predictions.shape == actuals.shape, "prediction and actual dataframes aren't the same size!"
    
    # num output variables (columns)
    num_vars = predictions.shape[1]
    
    # make subplots, one for each output variable
    fig, axs = plt.subplots(nrows=1, ncols=num_vars, figsize=(5 * num_vars, 5))
    axs = np.atleast_2d(axs)
    
    # if only one output variable, axs will not be an array, so wrap it
    if num_vars == 1:
        axs = np.array(axs)   
    for col in range(num_vars):
        # Extract actual & predicted values for the current variable
        predicted = predictions.iloc[:, col].to_frame()
        actual = actuals.iloc[:, col].to_frame()
        prediction_top = predictions_top.iloc[:, col].to_frame()
        actual_top = actuals_top.iloc[:, col].to_frame()
        plot_prediction_vs_actual(axs, 0, col, predicted, actual, 
                                prediction_top, actual_top)
    
    # legend
    handles = []
    if predictions_top is not None and actuals_top is not None:
        handles = [
            mpatches.Patch(color='green', label='Both predicted and actual top'),
            mpatches.Patch(color='red', label='Predicted top, actual not top'),
            mpatches.Patch(color='blue', label='Actual top, predicted not top'),
            mpatches.Patch(color='grey', label='Neither predicted nor actual top')
        ]
    elif predictions_top is not None or actuals_top is not None:
        handles = [
            mpatches.Patch(color='green', label='Top'),
            mpatches.Patch(color='grey', label='Not top')
        ]
    
    if handles:
        fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(1, 1.2))

    #adjust layout
    plt.tight_layout()
    plt.show()

def _render_pr_curves(horiz_data_grid: np.ndarray[pd.DataFrame],
                     vert_data_grid: np.ndarray[pd.DataFrame],
                     axs: np.ndarray,
                     num_rows: int,
                     num_cols: int):
    """Renders precision-recall curves for binary classification results.

    Date: 2024/11/02
    
    Args:
        horiz_data_grid: Grid of DataFrames containing predicted probabilities (0-1)
        vert_data_grid: Grid of DataFrames containing true binary labels (0 or 1)
        axs: Matplotlib axes grid to plot on
        num_rows: Number of rows in the grid
        num_cols: Number of columns in the grid
    """
    for i in range(num_rows):
        for j in range(num_cols):
            ax = axs[i, j]
            
            y_pred = horiz_data_grid[i, j].iloc[:, 0]
            y_true = vert_data_grid[i, j].iloc[:, 0]
            
            # Verify predictions are probabilities
            if not np.all((y_pred >= 0) & (y_pred <= 1)):
                raise ValueError("Predictions must be probabilities between 0 and 1")
            
            # Calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            
            # Calculate average precision score
            ap_score = average_precision_score(y_true, y_pred)
            
            # Plot PR curve
            ax.plot(recall, precision, color='blue', lw=2)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            
            # Add variable name and AP score to title
            var_name = horiz_data_grid[i, j].columns[0]
            ax.set_title(f'{var_name}\nAP = {ap_score:.2f}')
            
            # Add baseline at y=0.5
            ax.plot([0, 1], [0.5, 0.5], 'r--', alpha=0.3)
            
            # Set axis limits
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.grid(True, alpha=0.3)

def _render_roc_curves(horiz_data_grid: np.ndarray[pd.DataFrame], 
                      vert_data_grid: np.ndarray[pd.DataFrame], 
                      axs: np.ndarray, 
                      num_rows: int, 
                      num_cols: int):
    """Renders ROC curves for binary classification results.
    
    Args:
        horiz_data_grid: Grid of DataFrames containing predicted probabilities (0-1)
        vert_data_grid: Grid of DataFrames containing true binary labels (0 or 1)
        axs: Matplotlib axes grid to plot on
        num_rows: Number of rows in the grid
        num_cols: Number of columns in the grid
    """
    for i in range(num_rows):
        for j in range(num_cols):
            ax = axs[i, j]
            
            y_pred = horiz_data_grid[i, j].iloc[:, 0]
            y_true = vert_data_grid[i, j].iloc[:, 0]
            
            # Verify predictions are probabilities
            if not np.all((y_pred >= 0) & (y_pred <= 1)):
                raise ValueError("Predictions must be probabilities between 0 and 1")
                
            # Verify labels are binary
            if not np.all((y_true == 0) | (y_true == 1)):
                raise ValueError("True labels must be binary (0 or 1)")
            
            # Calculate ROC curve with many thresholds
            fpr, tpr, _ = roc_curve(y_true, y_pred, drop_intermediate=False)
            roc_auc = auc(fpr, tpr)
            
            # Plot curve with higher resolution
            ax.plot(fpr, tpr, color='navy', lw=2, 
                   label=f'ROC (AUC = {roc_auc:.2f})')
            ax.fill_between(fpr, tpr, alpha=0.2, color='navy')
            
            # Add diagonal random line
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
            
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.legend(loc='lower right')

def _render_scatter_plots(horiz_data_grid: np.ndarray[pd.DataFrame], vert_data_grid: np.ndarray[pd.DataFrame], axs: np.ndarray, num_rows: int, num_cols: int):

    """Renders scatter plots on a grid of subplots.
        Created: 2024/12/22
        horiz_data_grid (np.ndarray[pd.DataFrame]): A 2D numpy array where each element is a pandas DataFrame containing the horizontal data for the scatter plots.
        vert_data_grid (np.ndarray[pd.DataFrame]): A 2D numpy array where each element is a pandas DataFrame containing the vertical data for the scatter plots.
        axs (np.ndarray): A 2D numpy array of matplotlib Axes objects where the scatter plots will be rendered.
        num_rows (int): The number of rows in the grid of subplots.
        num_cols (int): The number of columns in the grid of subplots.
    """
    # Determine overall min and max values
    overall_min = min(np.min([df.min().min() for df in horiz_data_grid.flatten()]), 
                      np.min([df.min().min() for df in vert_data_grid.flatten()]))
    overall_max = max(np.max([df.max().max() for df in horiz_data_grid.flatten()]), 
                      np.max([df.max().max() for df in vert_data_grid.flatten()]))

    # Add a small padding to ensure all points are visible
    padding = (overall_max - overall_min) * 0.05
    plot_min = overall_min - padding
    plot_max = overall_max + padding

    for i in range(num_rows):
        for j in range(num_cols):
            ax = axs[i, j]
            x_data = horiz_data_grid[i, j]
            y_data = vert_data_grid[i, j]

           # Scatter plot
            ax.scatter(x_data, y_data, color='navy', s = 10, alpha=0.25)

            # Set labels
            ax.set_xlabel(f"Predicted")
            ax.set_ylabel(f"Actual")

            # Set the same limits for both x and y axes
            ax.set_xlim(plot_min, plot_max)
            ax.set_ylim(plot_min, plot_max)

def _create_multiplot_grid(horiz_data_grid: np.ndarray[pd.DataFrame], vert_data_grid: np.ndarray[pd.DataFrame], horiz_axis: mdo.AxisParams, vert_axis: mdo.AxisParams, plot_func: callable, suptitle: str = "") -> tuple[plt.Figure, np.ndarray]:
    """Plots a grid of scatter plots for the given data. 
    
    Created: 2024/11/02

    Args:
        x_grid (np.ndarray): 2D numpy array of DataFrames, each with one column of data.
        y_grid (np.ndarray): 2D numpy array of DataFrames, each with one column of data.
        x_axis_labels (list): List of labels for the x-axis.
        x_axis_name (str): Name of the x-axis.
        y_axis_labels (list): List of labels for the y-axis.
        y_axis_name (str): Name of the y-axis.

    Returns:
        tuple[plt.Figure, np.ndarray]: Figure and 2D numpy array of axes.
    """

    assert horiz_data_grid.shape == vert_data_grid.shape, f"x and y grids must have the same shape, but they have shapes {horiz_data_grid.shape} and {vert_data_grid.shape}"
    for i in range(horiz_data_grid.shape[0]):
        for j in range(horiz_data_grid.shape[1]):
            assert isinstance(horiz_data_grid[i, j], pd.DataFrame), \
                f"All dataframes in x_grid must be DataFrames, but {type(horiz_data_grid[i, j])} was found at ({i}, {j})"       
            assert isinstance(vert_data_grid[i, j], pd.DataFrame), \
                f"All dataframes in y_grid must be DataFrames, but {type(vert_data_grid[i, j])} was found at ({i}, {j})"
            assert horiz_data_grid[i, j].shape[1] == 1, \
                f"All dataframes in x_grid must have only 1 column, but {horiz_data_grid[i, j].shape[1]} was found at ({i}, {j})"
            assert vert_data_grid[i, j].shape[1] == 1, \
                f"All dataframes in y_grid must have only 1 column, but {vert_data_grid[i, j].shape[1]} was found at ({i}, {j})"
            assert horiz_data_grid[i, j].shape[0] == vert_data_grid[i, j].shape[0], \
                f"All corresponding dataframes in x_grid and y_grid must have the same number of datapoints, but {horiz_data_grid[i, j].shape[0]} and {vert_data_grid[i, j].shape[0]} were found at ({i}, {j})"
    
    # Get the number of rows and columns for the grid of plots
    num_rows, num_cols = horiz_data_grid.shape

    # Create a figure with a grid of subplots
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(5 * num_cols, 5 * num_rows))

    # Set entire subplot grid title
    plt.subplots_adjust(top=0.93)
    fig.suptitle(suptitle, fontweight='bold', fontsize=16, y=0.95)

    plot_func(horiz_data_grid, vert_data_grid, axs, num_rows, num_cols)

    for i in range(num_rows):
        for j in range(num_cols):
            # Set title denoting which parameters were used
            ax = axs[i, j]
            ax.set_title(f"{horiz_axis.name} = {horiz_axis.values[i]}, {vert_axis.name} = {vert_axis.values[j]}")

    return fig, axs

def create_scatter_grid(horiz_data_grid: np.ndarray[pd.DataFrame], vert_data_grid: np.ndarray[pd.DataFrame], horiz_axis: mdo.AxisParams, vert_axis: mdo.AxisParams, suptitle: str = "") -> tuple[plt.Figure, np.ndarray]:
    """
    Creates a grid of scatter plots.   
    Created: 2024/12/22
    Parameters:
    horiz_data_grid (np.ndarray[pd.DataFrame]): A 2D array of pandas DataFrames containing the horizontal data for the scatter plots.
    vert_data_grid (np.ndarray[pd.DataFrame]): A 2D array of pandas DataFrames containing the vertical data for the scatter plots.
    horiz_axis (mdo.AxisParams): Axis parameters for the horizontal axis.
    vert_axis (mdo.AxisParams): Axis parameters for the vertical axis.
    suptitle (str, optional): The super title for the entire grid of plots. Defaults to an empty string.
    Returns:
    tuple[plt.Figure, np.ndarray]: A tuple containing the matplotlib Figure object and a 2D array of Axes objects.
    """
    
    return _create_multiplot_grid(horiz_data_grid, vert_data_grid, horiz_axis, vert_axis, _render_scatter_plots, suptitle)

def create_pr_grid(predicted_values: pd.DataFrame, actual_values: pd.DataFrame,
                  num_rows: int, num_cols: int) -> tuple[plt.Figure, np.ndarray]:
    """Creates a grid of precision-recall curves for binary classification results.
    
    Args:
        predicted_values: DataFrame with predicted probabilities (0-1)
        actual_values: DataFrame with true binary labels (0 or 1)
        num_rows: Number of rows in visualization grid
        num_cols: Number of columns in visualization grid
        
    Returns:
        tuple: (matplotlib figure, array of axes)
    """
    # Create figure and axes grid
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))
    axs = np.atleast_2d(axs)
    
    # Reshape data into grids matching the visualization layout
    horiz_data = np.array(np.split(predicted_values, num_rows*num_cols, axis=1))
    horiz_data = horiz_data.reshape(num_rows, num_cols)
    
    vert_data = np.array(np.split(actual_values, num_rows*num_cols, axis=1))
    vert_data = vert_data.reshape(num_rows, num_cols)
    
    # Render the PR curves
    _render_pr_curves(horiz_data, vert_data, axs, num_rows, num_cols)
    
    plt.tight_layout()
    return fig, axs

def create_roc_grid(horiz_data_grid: np.ndarray[pd.DataFrame], vert_data_grid: np.ndarray[pd.DataFrame], horiz_axis: mdo.AxisParams, vert_axis: mdo.AxisParams, suptitle: str = "") -> tuple[plt.Figure, np.ndarray]:
    """
    Creates a grid of ROC (Receiver Operating Characteristic) curves.
    Created: 2024/12/20
    Parameters:
    horiz_data_grid (np.ndarray[pd.DataFrame]): A 2D numpy array where each element is a pandas DataFrame containing the horizontal data for the ROC curves.
    vert_data_grid (np.ndarray[pd.DataFrame]): A 2D numpy array where each element is a pandas DataFrame containing the vertical data for the ROC curves.
    horiz_axis (mdo.AxisParams): Axis parameters for the horizontal axis.
    vert_axis (mdo.AxisParams): Axis parameters for the vertical axis.
    suptitle (str, optional): The super title for the entire grid of plots. Defaults to an empty string.
    Returns:
    tuple[plt.Figure, np.ndarray]: A tuple containing the matplotlib Figure object and a numpy array of Axes objects.
    """
    
    return _create_multiplot_grid(horiz_data_grid, vert_data_grid, horiz_axis, vert_axis, _render_roc_curves, suptitle)

def color_spectrum(fig: plt.Figure, axs: np.ndarray, values: np.ndarray, label: str = "Value") -> tuple[plt.Figure, np.ndarray]:
    """
    Highlight the plots according to their given values.

    Created: 2024/11/03

    Args:
        fig (plt.Figure): The figure containing the subplots
        axs (np.ndarray): Array of axes for each subplot
        values (np.ndarray): 2D numpy array of values to be highlighted
        label (str): Label to use for the text in each subplot. Default is "Value"

    Returns:
        tuple[plt.Figure, np.ndarray]: 
            - The modified figure
            - The modified axes array
    """
    num_rows, num_cols = values.shape
    best_indices = np.unravel_index(np.argmax(values), values.shape)

    # Define colormap (red to yellow to green)
    cmap = LinearSegmentedColormap.from_list("", ["#FFB3BA", "#FFFFB3", "#BAFFC9"])

    # Normalize values to [0, 1] for coloring
    norm_values = (values - np.min(values)) / (np.max(values) - np.min(values))

    for i in range(num_rows):
        for j in range(num_cols):
            ax = axs[i, j]
            value = values[i, j]
            
            # Set background color based on normalized value
            color = cmap(norm_values[i, j])
            ax.set_facecolor(color)
            
            # Add text with value
            ax.text(0.05, 0.95, f'{label} = {value:.2f}', 
                    transform=ax.transAxes, fontsize=10, verticalalignment='top')

    # Highlight the best value in bright cyan
    axs[best_indices[0], best_indices[1]].set_facecolor('cyan')

    return fig, axs

def add_best_fit(axs):
    """
    Adds a best fit line (y=x) to the given matplotlib axes.
    Created: 2024/12/01
    Parameters:
    axs (matplotlib.axes.Axes or numpy.ndarray of matplotlib.axes.Axes): 
        A single matplotlib Axes object or an array of Axes objects.
    Notes:
    - The function ensures that the best fit line stays within the plot limits.
    - The line is plotted in red with a dotted style, a linewidth of 1.5, and an alpha of 0.7.
    - The original x and y limits of the axes are restored after plotting the line.
    """

    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    
    for ax in axs.flatten():
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Use the smaller range to ensure the line stays within the plot
        min_val = max(min(xlim[0], ylim[0]), -1000)  # Limit to prevent extreme values
        max_val = min(max(xlim[1], ylim[1]), 1000)   # Limit to prevent extreme values
        
        x = np.linspace(min_val, max_val, 100)
        y = x  # For y=x line
        
        ax.plot(x, y, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
        
        # Reset the original limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

def plot_model_metrics(df : pd.DataFrame, model_name : str) -> plt.Figure:
    """
        Plots the performance metrics of a given model.
        Created: 2025/01/02
         Parameters:
        df (pd.DataFrame): DataFrame containing the performance metrics. Each column represents a different metric.
        model_name (str): Name of the model to be displayed in the plot title.
        Returns:
        plt.Figure: The matplotlib figure object containing the plots.
    """
    
    # Style parameters
    plt.rcParams.update({'font.size': 12})
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f1c40f', '#1abc9c']
    
    # Get dimensions
    n_metrics = len(df.columns)
    
    # Create figure and subplots
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
    
    # Add bold super-title
    fig.suptitle(f'Model {model_name} Performance Metrics', fontsize=18, fontweight='bold', y=1.05)
    
    # Handle case where there's only one metric
    if n_metrics == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(df.columns):
        valid_data = df[metric].dropna()
        bars = axes[i].bar(range(len(valid_data)), valid_data, 
                          color=colors[:len(valid_data)])
        
        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom',
                        fontsize=14)
        
        axes[i].set_title(metric, fontsize=16, pad=15)
        axes[i].set_xticks(range(len(valid_data)))
        axes[i].set_xticklabels(valid_data.index, rotation=45, fontsize=12)
        axes[i].tick_params(axis='y', labelsize=12)
        axes[i].set_ylim(0, 1.0)
    
    plt.tight_layout()
    return fig