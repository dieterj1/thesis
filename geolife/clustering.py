from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
from pyproj import Transformer

def cluster_trace(eps, min_samples, trace):
    xy_clusters = []
    lonlat_clusters = [[]]

    # Beijing in UTM 50N 
    wgs84_to_utm = Transformer.from_crs(4326, 32650, always_xy=True)

    for i in trace:
        try:
            x, y = wgs84_to_utm.transform(i[1], i[0]) 
            xy_clusters.append([x, y]) 
        except Exception as e:
            print(f"Error transforming point {i}: {e}")

    # Convert to a NumPy array for better handling
    xy_clusters = np.array(xy_clusters)

    # Check for NaN or infinite values
    if not np.all(np.isfinite(xy_clusters)):
        print("Warning: Found NaN or infinite values in xy_clusters.")
        xy_clusters = xy_clusters[np.isfinite(xy_clusters).all(axis=1)]  # Filter out invalid points

    # Run DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xy_clusters)

    cluster_nr = 0
    for i in range(len(clustering.labels_)):  
        new_cluster_nr = clustering.labels_[i]
        if new_cluster_nr != -1:
            if cluster_nr != new_cluster_nr:
                cluster_nr = new_cluster_nr
            if new_cluster_nr >= len(lonlat_clusters):
                lonlat_clusters.append([]) 
            lonlat_clusters[cluster_nr].append(trace[i]) 

    return lonlat_clusters
