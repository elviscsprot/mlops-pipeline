import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def load_data(filepath: str) -> np.ndarray:
    df = pd.read_csv(filepath)
    return df[["lat", "lng"]].to_numpy()

def prepare_features(points: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    features = scaler.fit_transform(points)
    return features, scaler