import pytest
import pandas as pd
import numpy as np
from custom_dataset_tools import classification_metrics

def test_classification_metrics():
    # Test case 1: Perfect predictions
    predictions = pd.DataFrame({'A': [True, False, True], 'B': [False, True, False]})
    actual = pd.DataFrame({'A': [True, False, True], 'B': [False, True, False]})
    result = classification_metrics(predictions, actual)
    assert result.shape == (2, 4)
    assert np.allclose(result.values, 1.0)  # All metrics should be 1.0 for perfect predictions

    # Test case 2: Completely wrong predictions
    predictions = pd.DataFrame({'A': [False, True, False], 'B': [True, False, True]})
    actual = pd.DataFrame({'A': [True, False, True], 'B': [False, True, False]})
    result = classification_metrics(predictions, actual)
    assert result.shape == (2, 4)
    assert np.allclose(result.values[:, 0:3], 0.0)  # All metrics except kappa should be 0.0 for completely wrong predictions
    assert all(kappa < 0 for kappa in result.values[:, 3])  # kappa should be less than 0

    # Test case 3: Mixed predictions
    predictions = pd.DataFrame({'A': [True, False, True, False], 'B': [False, True, False, True]})
    actual = pd.DataFrame({'A': [True, False, False, True], 'B': [False, True, True, False]})
    result = classification_metrics(predictions, actual)
    assert result.shape == (2, 4)
    # Check if the results are within a reasonable range (0 to 1)
    assert np.all((result >= 0) & (result <= 1))

def test_classification_metrics_input_validation():
    # Test case 6: Mismatched shapes
    predictions = pd.DataFrame({'A': [True, False], 'B': [False, True]})
    actual = pd.DataFrame({'A': [True, False, True], 'B': [False, True, False]})
    with pytest.raises(ValueError):
        classification_metrics(predictions, actual)

    # Test case 7: Non-boolean data
    predictions = pd.DataFrame({'A': [1, 0, 1], 'B': [0, 1, 0]})
    actual = pd.DataFrame({'A': [True, False, True], 'B': [False, True, False]})
    with pytest.raises(ValueError):
        classification_metrics(predictions, actual)

    # Test case 8: Missing values
    predictions = pd.DataFrame({'A': [True, False, np.nan], 'B': [False, True, False]})
    actual = pd.DataFrame({'A': [True, False, True], 'B': [False, True, False]})
    with pytest.raises(ValueError):
        classification_metrics(predictions, actual)

if __name__ == "__main__":
    pytest.main()