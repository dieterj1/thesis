import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
import folium
import requests
from joblib import Parallel, delayed
import re
from tqdm import tqdm

# Constants
NOISE_THRESHOLD = 50  # meters
DBSCAN_EPS = 200  # meters
MIN_SAMPLES = 3
EARTH_RADIUS = 6371000  # meters

# Parse LINESTRING into coordinates
def parse_linestring(linestring):
    coords_str = re.match(r'LINESTRING\((.*)\)', linestring).group(1)
    return [tuple(map(float, pt.split())) for pt in coords_str.split(', ')]

# Detect stops in a single trace
def detect_stops(trace):
    points = parse_linestring(trace)
    stops = []
    for i in range(1, len(points)):
        if great_circle(points[i-1], points[i]).meters <= NOISE_THRESHOLD:
            stops.append(points[i])

    print(f"Detected {len(stops)} stops in trace.")  # Debug print
    return stops

# Process a chunk of the CSV
def process_chunk(chunk):
    stops = []
    for _, row in chunk.iterrows():
        detected = detect_stops(row['geom'])
        print(f"Processed {row['geom']} -> {detected}")
        stops.extend(detected)
    return stops

# Query OSM for amenities
def get_amenities(lat, lon, radius=100):
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    node["amenity"](around:{radius},{lat},{lon});
    out;
    """
    response = requests.post(overpass_url, data=query)
    elements = response.json().get('elements', [])
    return [{
        'type': elem['tags'].get('amenity', ''),
        'name': elem['tags'].get('name', '')
    } for elem in elements if 'amenity' in elem.get('tags', {})]

# Main function
def main():
    df_sample = pd.read_csv('taxi.csv', nrows=5)
    print(df_sample.head())
    # Read CSV in chunks
    chunk_size = 10000  # Adjust based on memory constraints
    chunks = pd.read_csv('taxi.csv', chunksize=chunk_size)

    # Count total rows for progress tracking
    total_rows = sum(1 for _ in pd.read_csv('taxi.csv', chunksize=chunk_size))

    # Parallelize stop detection with progress tracking
    all_stops = Parallel(n_jobs=-1)(
        delayed(process_chunk)(chunk)
        for chunk in tqdm(chunks, total=total_rows // chunk_size, desc="Processing chunks")
    )
    print("Number of detected stops:", len(all_stops))
    print("Sample stops:", all_stops[:5])
    print("Shape of coords:", coords.shape)
    if not all_stops:
        print("No stops detected, exiting.")
        return

    all_stops = [stop for sublist in all_stops for stop in sublist]  # Flatten list

    # Cluster stops using DBSCAN
    coords = np.array([(lat, lon) for (lon, lat) in all_stops])
    coords_rad = np.radians(coords)
    eps_rad = DBSCAN_EPS / EARTH_RADIUS

    db = DBSCAN(eps=eps_rad, min_samples=MIN_SAMPLES, metric='haversine')
    labels = db.fit_predict(coords_rad)

    # Get cluster centers
    clusters = {}
    for label, (lon, lat) in tqdm(zip(labels, all_stops), total=len(all_stops), desc="Clustering stops"):
        if label != -1:  # Ignore noise
            clusters.setdefault(label, []).append((lat, lon))

    cluster_centers = [np.mean(points, axis=0) for points in clusters.values()]

    # Visualize clusters with Folium
    map_rome = folium.Map(location=[41.9028, 12.4964], zoom_start=12)

    for center in tqdm(cluster_centers, desc="Adding markers to map"):
        lat, lon = center
        amenities = get_amenities(lat, lon)
        amenity_counts = {}
        for a in amenities:
            amenity_counts[a['type']] = amenity_counts.get(a['type'], 0) + 1
        top_amenities = sorted(amenity_counts.items(), key=lambda x: -x[1])[:3]

        popup = folium.Popup(
            '<br>'.join([f"{k}: {v}" for k, v in top_amenities]),
            max_width=250
        )
        folium.Marker([lat, lon], popup=popup).add_to(map_rome)

    map_rome.save('taxi_stops.html')

if __name__ == '__main__':
    main()