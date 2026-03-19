import numpy as np

class EnsembleModel:

    def __init__(self):
        self.models = {}

    def load_all(self):
        """Placeholder loader"""
        pass

    def predict(self, preds):
        """Average predictions from models"""
        return np.mean(preds, axis=0)