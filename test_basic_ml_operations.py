from custom_dataset_tools import classification_metrics
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from basic_ml_operations import (
    find_optimal_threshold_absolute, train_XGB_regressor, train_SVM_regressor, train_model_grid, power_list,
    grid_predict, calculate_pearson_coefficients
)
from ml_data_objects import AxisParams


@pytest.fixture
def sample_data():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    return pd.DataFrame(X), pd.DataFrame(y)

def test_train_XGB(sample_data):
    X, y = sample_data
    model = train_XGB_regressor(X, y.values.ravel())
    assert hasattr(model, 'predict')
    predictions = model.predict(X)
    assert predictions.shape == (100,)

def test_train_SVM_reg(sample_data):
    X, y = sample_data
    model = train_SVM_regressor(X, y.values.ravel())
    assert hasattr(model, 'predict')
    predictions = model.predict(X)
    assert predictions.shape == (100,)
    
def test_train_model_grid(sample_data):
    X, y = sample_data
    axis1_params = AxisParams("param1", [1, 2, 3])
    axis2_params = AxisParams("param2", [0.1, 0.2, 0.3])
    
    def dummy_train_model(X, y, param1, param2):
        return f"Model with param1={param1}, param2={param2}"

    model_grid = train_model_grid(X, y, axis1_params, axis2_params, dummy_train_model)
    assert model_grid.shape == (3, 3)
    assert model_grid[0, 0] == "Model with param1=1, param2=0.1"
    assert model_grid[0, 1] == "Model with param1=1, param2=0.2"
    assert model_grid[0, 2] == "Model with param1=1, param2=0.3"
    assert model_grid[1, 0] == "Model with param1=2, param2=0.1"
    assert model_grid[1, 1] == "Model with param1=2, param2=0.2"
    assert model_grid[1, 2] == "Model with param1=2, param2=0.3"
    assert model_grid[2, 0] == "Model with param1=3, param2=0.1"
    assert model_grid[2, 1] == "Model with param1=3, param2=0.2"
    assert model_grid[2, 2] == "Model with param1=3, param2=0.3"

def test_power_list():
    result = power_list(2, 0, 3)
    assert result == [1, 2, 4, 8]

    result = power_list(10, -1, 1)
    assert result == [0.1, 1, 10]

    with pytest.raises(AssertionError):
        power_list(2, 3, 1)  # start > end

def test_grid_predict(sample_data):
    X, _ = sample_data
    
    class DummyModel:
        def predict(self, X):
            return np.ones(X.shape[0])

    model_grid = np.array([[DummyModel(), DummyModel()], [DummyModel(), DummyModel()]])
    predictions = grid_predict(X, model_grid)
    
    assert predictions.shape == (2, 2)
    assert isinstance(predictions[0, 0], pd.DataFrame)
    assert predictions[0, 0].shape == (100, 1)
    assert np.all(predictions[0, 0] == 1)

def test_calculate_pearson_coefficients():
    X_grid = np.zeros((2, 2), dtype=object)
    for row in range(2):
        for col in range(2):
            X_grid[row, col] = pd.DataFrame({'A': [1, 2, 3]})

    y_grid = np.zeros((2, 2), dtype=object)  
    for row in range(2):
        for col in range(2):
            y_grid[row, col] = pd.DataFrame({'B': [1, 2, 3]})

    coeffs = calculate_pearson_coefficients(X_grid, y_grid)
    assert coeffs.shape == (2, 2)
    assert np.allclose(coeffs, 1.0)  # All correlations should be 1.0

    # Test with different values
    y_grid[0, 0] = pd.DataFrame({'B': [3, 2, 1]})
    coeffs = calculate_pearson_coefficients(X_grid, y_grid)
    assert np.allclose(coeffs[0, 0], -1.0)  # Perfect negative correlation

def test_classification_metrics():
    # Test case 1: Perfect predictions
    pred_perfect = pd.DataFrame({'col1': [True, False, True]})
    actual_perfect = pd.DataFrame({'col1': [True, False, True]})
    metrics_perfect = classification_metrics(pred_perfect, actual_perfect)
    assert np.allclose(metrics_perfect['F1 Score'], 1.0)
    assert np.allclose(metrics_perfect['Sensitivity'], 1.0)
    assert np.allclose(metrics_perfect['Specificity'], 1.0)
    assert np.allclose(metrics_perfect['Kappa'], 1.0)

    # Test case 2: Mixed predictions
    pred_mixed = pd.DataFrame({'col1': [True, False, True, False]})
    actual_mixed = pd.DataFrame({'col1': [True, True, False, False]})
    metrics_mixed = classification_metrics(pred_mixed, actual_mixed)
    assert np.allclose(metrics_mixed['F1 Score'], 0.5)
    assert np.allclose(metrics_mixed['Sensitivity'], 0.5)
    assert np.allclose(metrics_mixed['Specificity'], 0.5)
    assert np.allclose(metrics_mixed['Kappa'], 0.0)

    # Test case 3: Complete misclassification
    pred_wrong = pd.DataFrame({'col1': [False, True, False]})
    actual_wrong = pd.DataFrame({'col1': [True, False, True]})
    metrics_wrong = classification_metrics(pred_wrong, actual_wrong)
    assert np.allclose(metrics_wrong['F1 Score'], 0.0)
    assert np.allclose(metrics_wrong['Sensitivity'], 0.0)
    assert np.allclose(metrics_wrong['Specificity'], 0.0)
    assert np.allclose(metrics_wrong['Kappa'], -0.8)

    # Test case with unequal precision and recall
    pred_unequal = pd.DataFrame({'col1': [True, True, True, False]})
    actual_unequal = pd.DataFrame({'col1': [True, False, True, False]})
    metrics_unequal = classification_metrics(pred_unequal, actual_unequal)
    assert np.allclose(metrics_unequal['Sensitivity'], 1.0)
    assert np.allclose(metrics_unequal['Specificity'], 0.5)

    # Test error conditions
    with pytest.raises(ValueError):
        pred_wrong_shape = pd.DataFrame({'col1': [True, False]})
        actual_wrong_shape = pd.DataFrame({'col1': [True, False, True]})
        classification_metrics(pred_wrong_shape, actual_wrong_shape)

    with pytest.raises(ValueError):
        pred_non_bool = pd.DataFrame({'col1': [1, 0, 1]})
        actual_non_bool = pd.DataFrame({'col1': [1, 0, 1]})
        classification_metrics(pred_non_bool, actual_non_bool)

    with pytest.raises(ValueError):
        pred_nan = pd.DataFrame({'col1': [True, False, np.nan]})
        actual_nan = pd.DataFrame({'col1': [True, False, True]})
        classification_metrics(pred_nan, actual_nan)

def test_find_optimal_threshold():
    # Test basic functionality with balanced data
    y_true = pd.DataFrame({'col1': [True, False, True, False]})
    y_pred = pd.DataFrame({'col1': [0.8, 0.2, 0.7, 0.3]})
    threshold = find_optimal_threshold_absolute(y_true, y_pred)
    assert 0 <= threshold <= 1
    
    # Test with perfect predictions
    y_true = pd.DataFrame({'col1': [True, False, True, False]})
    y_pred = pd.DataFrame({'col1': [1.0, 0.0, 1.0, 0.0]})
    threshold = find_optimal_threshold_absolute(y_true, y_pred)
    assert 0 <= threshold <= 1
    
    # Test with larger dataset
    y_true = pd.DataFrame({'col1': [True] * 50 + [False] * 50})
    y_pred = pd.DataFrame({'col1': [0.7] * 50 + [0.3] * 50})
    threshold = find_optimal_threshold_absolute(y_true, y_pred)
    assert 0 <= threshold <= 1
    
    # Test with imbalanced data
    y_true = pd.DataFrame({'col1': [True] * 80 + [False] * 20})
    y_pred = pd.DataFrame({'col1': [0.8] * 80 + [0.2] * 20})
    threshold = find_optimal_threshold_absolute(y_true, y_pred)
    assert 0 <= threshold <= 1
    
    # Test error conditions
    # Invalid binary values
    with pytest.raises(ValueError):
        y_true_invalid = pd.DataFrame({'col1': [2, 1, 0]})
        y_pred_valid = pd.DataFrame({'col1': [0.8, 0.2, 0.3]})
        find_optimal_threshold_absolute(y_true_invalid, y_pred_valid)
    
    # Probabilities out of range
    with pytest.raises(ValueError):
        y_true_valid = pd.DataFrame({'col1': [True, False, True]})
        y_pred_invalid = pd.DataFrame({'col1': [1.2, -0.1, 0.5]})
        find_optimal_threshold_absolute(y_true_valid, y_pred_invalid)

    # Very small differences in predictions
    y_true = pd.DataFrame({'col1': [True, False, True]})
    y_pred = pd.DataFrame({'col1': [0.501, 0.499, 0.501]})
    threshold = find_optimal_threshold_absolute(y_true, y_pred)
    assert 0 <= threshold <= 1

    # Edge case - all predictions same value
    y_true = pd.DataFrame({'col1': [True, False, True]})
    y_pred = pd.DataFrame({'col1': [0.5, 0.5, 0.5]})
    threshold = find_optimal_threshold_absolute(y_true, y_pred)
    assert 0 <= threshold <= 1
    


if __name__ == "__main__":
    pytest.main()

