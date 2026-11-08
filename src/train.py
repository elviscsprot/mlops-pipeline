import numpy as np
from sklearn.cluster import KMeans

def train_model(features: np.ndarray, n_clusters: int = 5, random_state: int = 42) -> KMeans:
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    model.fit(features)
    return model