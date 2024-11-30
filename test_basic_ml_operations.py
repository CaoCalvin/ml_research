import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from basic_ml_operations import (
    train_XGB, train_SVR, train_model_grid, power_list,
    grid_predict, calculate_pearson_coefficients
)
from ml_data_objects import AxisParams

@pytest.fixture
def sample_data():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    return pd.DataFrame(X), pd.DataFrame(y)

def test_train_XGB(sample_data):
    X, y = sample_data
    model = train_XGB(X, y.values.ravel())
    assert hasattr(model, 'predict')
    predictions = model.predict(X)
    assert predictions.shape == (100,)

def test_train_SVR(sample_data):
    X, y = sample_data
    model = train_SVR(X, y.values.ravel())
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

if __name__ == "__main__":
    pytest.main()