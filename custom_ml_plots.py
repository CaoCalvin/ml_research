import itertools
from typing import Optional
import pandas as pd
import numpy as np
import ml_data_objects as mdo


# Plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.stats import pearsonr

from matplotlib.colors import LinearSegmentedColormap

from sklearn.preprocessing import StandardScaler



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

def plot_grid(x_grid: np.ndarray[pd.DataFrame], y_grid: np.ndarray[pd.DataFrame], x_axis_name: str, x_axis_labels: list, y_axis_name: str, y_axis_labels: list) -> tuple[int, int]:
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
        tuple[int, int]: Indices of the grid with the best Pearson coefficient.
    """

    assert x_grid.shape == y_grid.shape, f"x and y grids must have the same shape, but they have shapes {x_grid.shape} and {y_grid.shape}"
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            assert isinstance(x_grid[i, j], pd.DataFrame), \
                f"All dataframes in x_grid must be DataFrames, but {type(x_grid[i, j])} was found at ({i}, {j})"       
            assert isinstance(y_grid[i, j], pd.DataFrame), \
                f"All dataframes in y_grid must be DataFrames, but {type(y_grid[i, j])} was found at ({i}, {j})"
            assert x_grid[i, j].shape[1] == 1, \
                f"All dataframes in x_grid must have only 1 column, but {x_grid[i, j].shape[1]} was found at ({i}, {j})"
            assert y_grid[i, j].shape[1] == 1, \
                f"All dataframes in y_grid must have only 1 column, but {y_grid[i, j].shape[1]} was found at ({i}, {j})"
            assert x_grid[i, j].shape[0] == y_grid[i, j].shape[0], \
                f"All corresponding dataframes in x_grid and y_grid must have the same number of datapoints, but {x_grid[i, j].shape[0]} and {y_grid[i, j].shape[0]} were found at ({i}, {j})"
    
    # Get the number of rows and columns for the grid of plots
    num_rows, num_cols = x_grid.shape

    # Create a figure with a grid of subplots
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(5 * num_cols, 5 * num_rows))

    # Set entire subplot grid title
    fig.suptitle(f"{x_axis_name} vs. {y_axis_name}")

    # Define the colormap to color graphs based on Pearson coefficients from red to green
    cmap = LinearSegmentedColormap.from_list('custom', ['#ffcccc', '#ffffcc', '#ccffcc'])
    
    # Calculate the Pearson coefficients for each plot
    pearson_coefs = np.zeros(x_grid.shape)
    max_pearson_coef = 0
    max_pearson_coef_idx = None
    for i in range(num_rows):
        for j in range(num_cols):
            # Get the data for the current plot
            x = x_grid[i, j]
            y = y_grid[i, j]

            # Calculate the Pearson coefficient for the current plot
            pearson_coef, _ = pearsonr(x.iloc[:, 0], y.iloc[:, 0])
            pearson_coefs[i, j] = pearson_coef

            # Plot the data using plot_prediction_vs_actual
            plot_prediction_vs_actual(axs, i, j, x, y)

            # Set the title of the plot
            axs[i, j].set_title(f"{x_axis_name}={x_axis_labels[i]}, {y_axis_name}={y_axis_labels[j]}")

            # Highlight plot with highest Pearson coefficient in bright green
            if pearson_coef > max_pearson_coef:
                max_pearson_coef = pearson_coef
                max_pearson_coef_idx = (i, j)

    # Normalize the Pearson coefficients to range from 0 to 1
    pearson_coefs = (pearson_coefs - pearson_coefs.min()) / (pearson_coefs.max() - pearson_coefs.min())

    # Color the plots based on the Pearson coefficients
    for i in range(num_rows):
        for j in range(num_cols):
            axs[i, j].set_facecolor(cmap(pearson_coefs[i, j]))
            
    if max_pearson_coef_idx is not None:
        axs[max_pearson_coef_idx[0], max_pearson_coef_idx[1]].set_facecolor('#00bfff')

    # Adjust layout
    plt.tight_layout()
    plt.show()

    return max_pearson_coef_idx

def eval_grid_search(X_test: pd.DataFrame, X_scaler: StandardScaler, y_test: pd.DataFrame, y_scaler: StandardScaler, model_grid: np.ndarray, x_params: mdo.AxisParams, y_params: mdo.AxisParams) -> tuple[int, int]:
    """Scales the data and makes predictions using the model_grid.

    Created: 2024/11/02

    Args:
        X_test (pd.DataFrame): Test data.
        X_scaler (StandardScaler): Scaler for the test data.
        y_test (pd.DataFrame): Test labels.
        y_scaler (StandardScaler): Scaler for the test labels.
        model_grid (np.ndarray): 2D numpy array of DataFrames. Each cell contains a trained model.
        x_params (AxisParams): Parameters for the x-axis.
        y_params (AxisParams): Parameters for the y-axis.        
        target_var (str): Name of the target column.

    Returns:
    tuple[int, int]: Indices of the grid with the best Pearson coefficient.
    """
    # Scale X_test
    X_test_scaled = X_scaler.transform(X_test)

    # Convert X_test_scaled to a DataFrame
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

    # Initialize numpy array to store predictions
    y_pred = np.zeros(model_grid.shape, dtype=pd.DataFrame)

    # Make predictions using the trained models on scaled X_test
    for i in range(model_grid.shape[0]):
        for j in range(model_grid.shape[1]):
            model = model_grid[i, j]
            single_pred_scaled = pd.DataFrame(model.predict(X_test_scaled), index=y_test.index, columns=y_test.columns)
            single_pred = pd.DataFrame(y_scaler.inverse_transform(single_pred_scaled), index=y_test.index, columns=y_test.columns)
            y_pred[i, j] = single_pred    

    y_test_grid = np.empty_like(y_pred)  # Initialize an empty array with the same shape as y_pred
    for i in range(y_pred.shape[0]):
        for j in range(y_pred.shape[1]):
            y_test_grid[i, j] = y_test.copy()  # Copy y_test for each element
    
    top_pearson_coef_idx = plot_grid(y_pred, y_test_grid, x_params.name, x_params.values, y_params.name, y_params.values)

    return top_pearson_coef_idx
