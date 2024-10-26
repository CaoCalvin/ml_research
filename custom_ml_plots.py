import pandas as pd
import numpy as np

# function typechecking
from typeguard import typechecked

# Plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.stats import pearsonr


@typechecked
def plot_prediction_vs_actual(axs: np.ndarray, row: int, col: int, predicted: pd.DataFrame, actual: pd.DataFrame, 
                                predicted_top: pd.DataFrame = None, actual_top: pd.DataFrame = None):
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
    

@typechecked 
def plot_predictions_vs_actuals(predictions: pd.DataFrame, actuals: pd.DataFrame, 
                                predictions_top: pd.DataFrame = None, actuals_top: pd.DataFrame = None):
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
        axs = [axs]    
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

    