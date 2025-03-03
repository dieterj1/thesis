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
NOISE_THRESHOLD = 10  # meters
MIN_STOP_DURATION = 60  # seconds
DBSCAN_EPS = 100  # meters
MIN_SAMPLES = 5
EARTH_RADIUS = 6371000  # meters

def parse_point(point_str):
    """Parse POINT(lat lon) into (latitude, longitude)."""
    match = re.match(r'POINT\(([\d.]+) ([\d.]+)\)', point_str)
    return (float(match.group(1)), float(match.group(2))) if match else (None, None)

def process_taxi(taxi_group):
    """Detect stops in a single taxi's trajectory."""
    taxi_group = taxi_group.sort_values('timestamp')
    stops, current_stop = [], []
    
    for _, row in taxi_group.iterrows():
        if current_stop:
            # Calculate distance from the last point
            last = current_stop[-1]
            distance = great_circle((last['lat'], last['lon']), (row['lat'], row['lon'])).meters
            if distance <= NOISE_THRESHOLD:
                current_stop.append(row)
                continue
        
        # Check if current_stop qualifies as a stop
        if len(current_stop) >= 2:
            duration = (current_stop[-1]['timestamp'] - current_stop[0]['timestamp']).total_seconds()
            if duration >= MIN_STOP_DURATION:
                avg_lat = np.mean([p['lat'] for p in current_stop])
                avg_lon = np.mean([p['lon'] for p in current_stop])
                stops.append((avg_lat, avg_lon))
        
        current_stop = [row]  # Reset for new potential stop
    
    # Check the last sequence
    if len(current_stop) >= 2:
        duration = (current_stop[-1]['timestamp'] - current_stop[0]['timestamp']).total_seconds()
        if duration >= MIN_STOP_DURATION:
            avg_lat = np.mean([p['lat'] for p in current_stop])
            avg_lon = np.mean([p['lon'] for p in current_stop])
            stops.append((avg_lat, avg_lon))
    
    return stops

def get_amenities(lat, lon, radius=10):
    """Query OSM for nearby amenities."""
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    node["amenity"](around:{radius},{lat},{lon});
    out;
    """
    try:
        response = requests.post(overpass_url, data=query, timeout=10)
        elements = response.json().get('elements', [])
        return [{
            'type': elem['tags'].get('amenity', ''),
            'name': elem['tags'].get('name', '')
        } for elem in elements if 'amenity' in elem.get('tags', {})]
    except Exception as e:
        print(f"OSM query failed: {e}")
        return []

def main():
    # Read and preprocess data
    print("Reading data...")
    df = pd.read_csv('4.txt', sep=';', header=None, names=['id', 'timestamp', 'geom'])
    df['lat'], df['lon'] = zip(*df['geom'].apply(parse_point))
    df = df.dropna(subset=['lat', 'lon'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['id', 'timestamp'])
    
    # Group by taxi ID and process in parallel
    taxi_groups = [group for _, group in df.groupby('id')]
    print(f"Processing {len(taxi_groups)} taxis...")
    all_stops = Parallel(n_jobs=-1)(
        delayed(process_taxi)(group) for group in tqdm(taxi_groups, desc="Taxis processed"))
    all_stops = [stop for taxi_stops in all_stops for stop in taxi_stops]
    
    if not all_stops:
        raise ValueError("No stops detected. Adjust noise/duration thresholds.")
    
    # Cluster stops
    coords = np.radians(np.array(all_stops))
    eps_rad = DBSCAN_EPS / EARTH_RADIUS
    labels = DBSCAN(eps=eps_rad, min_samples=MIN_SAMPLES, metric='haversine').fit_predict(coords)
    
    # Calculate cluster centers
    clusters = {}
    for label, (lat, lon) in zip(labels, all_stops):
        if label != -1:
            clusters.setdefault(label, []).append((lat, lon))
    cluster_centers = [np.mean(points, axis=0) for points in clusters.values()]
    
    # Create map
    map_rome = folium.Map(location=[41.9028, 12.4964], zoom_start=12)
    for center in tqdm(cluster_centers, desc="Adding markers"):
        lat, lon = center
        amenities = get_amenities(lat, lon)
        top_amenities = sorted(
            {a['type']: sum(1 for x in amenities if x['type'] == a['type']) for a in amenities}.items(),
            key=lambda x: -x[1]
        )[:3]
        popup = folium.Popup('<br>'.join([f"{k}: {v}" for k, v in top_amenities]), max_width=250)
        folium.Marker([lat, lon], popup=popup).add_to(map_rome)
    
    map_rome.save('taxi_stops10.html')
    print("Map saved to taxi_stops10.html")

if __name__ == '__main__':
    main()