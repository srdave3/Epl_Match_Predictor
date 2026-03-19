import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import poisson

def score_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluation metrics for goal predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Exact score accuracy
    exact_acc = np.mean(np.isclose(y_true, y_pred, atol=0.01))
    
    # Result accuracy (convert to W/D/L)
    def to_result(ht, at):
        if ht > at:
            return 0  # Home win
        elif at > ht:
            return 2  # Away win
        else:
            return 1  # Draw
    
    result_true = np.array([to_result(ht, at) for ht, at in zip(y_true[:len(y_true)//2], y_true[len(y_true)//2:])])
    # Simplified for single target
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'Exact Score %': exact_acc * 100,
    }

def brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Brier score for probability calibration."""
    return np.mean((probs - y_true)**2)

