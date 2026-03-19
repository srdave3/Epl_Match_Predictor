import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from utils.features import prepare_training_data
from typing import Tuple

class XGBModel:
    def __init__(self):
        self.home_model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        self.away_model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        self.feature_names = None
    
    def fit(self):
        """Train XGBoost models."""
        (X_train, y_home_train, y_away_train), _ = prepare_training_data()
        
        self.feature_names = X_train.columns.tolist()
        
        print('Training XGBoost home goals...')
        self.home_model.fit(X_train, y_home_train)
        
        print('Training XGBoost away goals...')
        self.away_model.fit(X_train, y_away_train)
        
        print('XGBoost models trained!')
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = X.reindex(columns=self.feature_names, fill_value=0)
        home_pred = self.home_model.predict(X)
        away_pred = self.away_model.predict(X)
        return home_pred, away_pred
    
    def save(self, path_home: str, path_away: str):
        joblib.dump(self.home_model, path_home)
        joblib.dump(self.away_model, path_away)
    
    @classmethod
    def load(cls, path_home: str, path_away: str):
        model = cls()
        model.home_model = joblib.load(path_home)
        model.away_model = joblib.load(path_away)
        return model

if __name__ == '__main__':
    model = XGBModel()
    model.fit()

