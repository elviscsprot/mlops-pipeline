import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import geopy.distance

def predict(model: KMeans, scaler: StandardScaler, lat: float, lng: float) -> dict:
    point = np.array([[lat, lng]])
    point_scaled = scaler.transform(point)

    cluster = int(model.predict(point_scaled)[0])

    centroid_scaled = model.cluster_centers_[cluster].reshape(1, -1)
    centroid_lat, centroid_lng = scaler.inverse_transform(centroid_scaled)[0]

    distance_km = geopy.distance.geodesic((lat, lng), (centroid_lat, centroid_lng)).km

    return {"cluster": cluster, "distance_km": round(float(distance_km), 4)}