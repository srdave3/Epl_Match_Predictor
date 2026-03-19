import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Tuple
from utils.features import prepare_training_data


class RFModel:
    def __init__(self):
        self.home_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.away_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.feature_names = None

    def fit(self):
        """Train RF models on features."""
        (X_train, y_home_train, y_away_train), _ = prepare_training_data()

        print("X_train shape:", X_train.shape)
        print("y_home_train shape:", y_home_train.shape)
        print("y_away_train shape:", y_away_train.shape)

        if X_train.empty:
            raise ValueError("prepare_training_data() returned 0 rows. Check feature engineering logic.")

        self.feature_names = X_train.columns.tolist()

        print("Training RF home goals...")
        self.home_model.fit(X_train, y_home_train)

        print("Training RF away goals...")
        self.away_model.fit(X_train, y_away_train)

        print("RF models trained!")

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict home and away goals."""
        # Recover feature names from loaded sklearn models if needed
        if self.feature_names is None:
            if hasattr(self.home_model, "feature_names_in_"):
                self.feature_names = list(self.home_model.feature_names_in_)
            else:
                raise ValueError("Model not trained and feature names are unavailable.")

        X = X.reindex(columns=self.feature_names, fill_value=0)

        home_pred = self.home_model.predict(X)
        away_pred = self.away_model.predict(X)

        return home_pred, away_pred

    def predict_match(self, features_dict: Dict) -> Dict:
        """Predict for single match features."""
        X = pd.DataFrame([features_dict])

        home_pred, away_pred = self.predict(X)

        return {
            "home_goals": float(home_pred[0]),
            "away_goals": float(away_pred[0])
        }

    def save(self, path_home: str, path_away: str):
        """Save trained RF models."""
        joblib.dump(self.home_model, path_home)
        joblib.dump(self.away_model, path_away)

    @classmethod
    def load(cls, path_home: str, path_away: str):
        """Load trained RF models."""
        model = cls()
        model.home_model = joblib.load(path_home)
        model.away_model = joblib.load(path_away)

        # Restore feature names from sklearn model metadata if available
        if hasattr(model.home_model, "feature_names_in_"):
            model.feature_names = list(model.home_model.feature_names_in_)

        return model


if __name__ == "__main__":
    model = RFModel()
    model.fit()

