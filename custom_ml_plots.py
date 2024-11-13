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

# import k-fold cross-validation
from sklearn.model_selection import KFold



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

def plot_grid(x_grid: np.ndarray[pd.DataFrame], y_grid: np.ndarray[pd.DataFrame], x_axis: mdo.AxisParams, y_axis: mdo.AxisParams) -> tuple[plt.Figure, np.ndarray]:
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
    fig.suptitle(f"{x_axis.name} vs. {y_axis.name}")

    # Determine overall min and max values
    overall_min = min(np.min([df.min().min() for df in x_grid.flatten()]), 
                      np.min([df.min().min() for df in y_grid.flatten()]))
    overall_max = max(np.max([df.max().max() for df in x_grid.flatten()]), 
                      np.max([df.max().max() for df in y_grid.flatten()]))

    # Add a small padding to ensure all points are visible
    padding = (overall_max - overall_min) * 0.05
    plot_min = overall_min - padding
    plot_max = overall_max + padding

    for i in range(num_rows):
        for j in range(num_cols):
            ax = axs[i, j]
            x_data = x_grid[i, j]
            y_data = y_grid[i, j]

            # Scatter plot
            ax.scatter(x_data, y_data, color='navy', s = 10, alpha=0.25)

            # Set labels
            ax.set_xlabel(f"Predicted")
            ax.set_ylabel(f"Actual")

            # Set the same limits for both x and y axes
            ax.set_xlim(plot_min, plot_max)
            ax.set_ylim(plot_min, plot_max)

            # Set title denoting which parameters were used
            ax.set_title(f"{x_axis.name} = {x_axis.values[i]}, {y_axis.name} = {y_axis.values[j]}")


    plt.tight_layout()

    return fig, axs

def calculate_pearson_coefficients(x_grid: np.ndarray[pd.DataFrame], y_grid: np.ndarray[pd.DataFrame]) -> np.ndarray:
    """
    Calculate Pearson coefficients for the given data.

    Args:
        x_grid (np.ndarray): 2D numpy array of DataFrames for x values
        y_grid (np.ndarray): 2D numpy array of DataFrames for y values

    Returns:
        np.ndarray: 2D numpy array of Pearson coefficients
    """
    num_rows, num_cols = x_grid.shape
    pearson_coeffs = np.zeros((num_rows, num_cols))

    for i in range(num_rows):
        for j in range(num_cols):
            x_data = x_grid[i, j]
            y_data = y_grid[i, j]
            pearson_coef, _ = pearsonr(x_data.iloc[:, 0], y_data.iloc[:, 0])
            pearson_coeffs[i, j] = pearson_coef

    return pearson_coeffs

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

def eval_k_fold_grid_search(X_sc: pd.DataFrame, y_sc: pd.DataFrame, 
                            kfold: KFold, model_grid: np.ndarray, 
                            x_params: mdo.AxisParams, y_params: mdo.AxisParams) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates the k-fold grid search results using the entire scaled dataset.

    Created: 2024/11/03

    Args:
        X_sc (pd.DataFrame): Scaled feature set.
        y_sc (pd.DataFrame): Scaled label set.
        kfold (KFold): KFold object for cross-validation.
        model_grid (np.ndarray): 2D numpy array of lists. Each cell contains k trained models.
        x_params (AxisParams): Parameters for the x-axis.
        y_params (AxisParams): Parameters for the y-axis.

    Returns:
        tuple[np.ndarray, np.ndarray]: 2D numpy array of predictions where each cell contains list of k predictions and 2D numpy array of actual values
    """
    # Initialize numpy array to store predictions for each parameter combination
    y_pred_grid = np.zeros(model_grid.shape, dtype=object)
    y_test_grid = np.zeros(model_grid.shape, dtype=object)

    # Iterate through each fold
    for fold, (_, test_index) in enumerate(kfold.split(X_sc)):
        X_CV = X_sc.iloc[test_index]
        y_CV = y_sc.iloc[test_index]

        # Make predictions using the k-th model in each cell
        for i in range(model_grid.shape[0]):
            for j in range(model_grid.shape[1]):
                model = model_grid[i, j][fold]
                fold_pred = pd.DataFrame(model.predict(X_CV), index=y_CV.index, columns=y_CV.columns)
                
                if y_pred_grid[i, j] is 0:  # First iteration
                    y_pred_grid[i, j] = fold_pred
                    y_test_grid[i, j] = y_CV
                else:
                    y_pred_grid[i, j] = pd.concat([y_pred_grid[i, j], fold_pred])
                    y_test_grid[i, j] = pd.concat([y_test_grid[i, j], y_CV])
    return y_pred_grid, y_test_grid

def eval_k_fold_grid_search_train(X_sc: pd.DataFrame, y_sc: pd.DataFrame, 
                                  kfold: KFold, model_grid: np.ndarray, 
                                  x_params: mdo.AxisParams, y_params: mdo.AxisParams) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates the k-fold grid search results using the training folds to assess overfitting.

    Created: 2024/11/12

    Args:
        X_sc (pd.DataFrame): Scaled feature set.
        y_sc (pd.DataFrame): Scaled label set.
        kfold (KFold): KFold object for cross-validation.
        model_grid (np.ndarray): 2D numpy array of lists. Each cell contains k trained models.
        x_params (AxisParams): Parameters for the x-axis.
        y_params (AxisParams): Parameters for the y-axis.

    Returns:
        tuple[np.ndarray, np.ndarray]: 2D numpy array of predictions where each cell contains predictions for training data,
                                       and 2D numpy array of actual values for training data
    """
    # Initialize numpy array to store predictions for each parameter combination
    y_pred_grid = np.zeros(model_grid.shape, dtype=object)
    y_train_grid = np.zeros(model_grid.shape, dtype=object)

    # Iterate through each fold
    for fold, (train_index, _) in enumerate(kfold.split(X_sc)):
        X_train = X_sc.iloc[train_index]
        y_train = y_sc.iloc[train_index]

        # Make predictions using the k-th model in each cell on its training data
        for i in range(model_grid.shape[0]):
            for j in range(model_grid.shape[1]):
                model = model_grid[i, j][fold]
                fold_pred = pd.DataFrame(model.predict(X_train), index=y_train.index, columns=y_train.columns)
                
                if y_pred_grid[i, j] is 0:  # First iteration
                    y_pred_grid[i, j] = fold_pred
                    y_train_grid[i, j] = y_train
                else:
                    y_pred_grid[i, j] = pd.concat([y_pred_grid[i, j], fold_pred])
                    y_train_grid[i, j] = pd.concat([y_train_grid[i, j], y_train])

    return y_pred_grid, y_train_grid

def add_best_fit(axs):
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