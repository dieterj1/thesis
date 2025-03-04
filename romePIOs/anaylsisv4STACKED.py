import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
import folium
import requests
from joblib import Parallel, delayed
import re
from tqdm import tqdm
from datetime import timedelta

# Constants
NOISE_THRESHOLD = 10  # meters
MIN_STOP_DURATION = 40  # seconds
DBSCAN_EPS = 50  # meters
MIN_SAMPLES = 3
EARTH_RADIUS = 6371000  # meters
MAX_CLUSTER_DURATION = timedelta(hours=3)  # Maximum duration for a sub-cluster
LOCATION_GROUPING_THRESHOLD = 50  # meters

EXCLUDED_AMENITIES = {
    'bench', 'waste_basket', 'drinking_water', 'clock',
    'motorcycle_parking', 'toilets', 'telephone', 'post_box'
}

def parse_point(point_str):
    """Parse POINT(lat lon) into (latitude, longitude)."""
    match = re.match(r'POINT\(([\d.]+) ([\d.]+)\)', point_str)
    return (float(match.group(1)), float(match.group(2))) if match else (None, None)

def process_taxi(taxi_group):
    """Detect stops with timestamps and point counts."""
    taxi_group = taxi_group.sort_values('timestamp')
    stops, current_stop = [], []
    
    for _, row in taxi_group.iterrows():
        if current_stop:
            last = current_stop[-1]
            distance = great_circle((last['lat'], last['lon']), (row['lat'], row['lon'])).meters
            if distance <= NOISE_THRESHOLD:
                current_stop.append(row)
                continue
        
        if len(current_stop) >= 2:
            duration = (current_stop[-1]['timestamp'] - current_stop[0]['timestamp']).total_seconds()
            if duration >= MIN_STOP_DURATION:
                stops.append({
                    'lat': np.mean([p['lat'] for p in current_stop]),
                    'lon': np.mean([p['lon'] for p in current_stop]),
                    'start': current_stop[0]['timestamp'],
                    'end': current_stop[-1]['timestamp'],
                    'points': len(current_stop)
                })
        
        current_stop = [row]
    
    if len(current_stop) >= 2:
        duration = (current_stop[-1]['timestamp'] - current_stop[0]['timestamp']).total_seconds()
        if duration >= MIN_STOP_DURATION:
            stops.append({
                'lat': np.mean([p['lat'] for p in current_stop]),
                'lon': np.mean([p['lon'] for p in current_stop]),
                'start': current_stop[0]['timestamp'],
                'end': current_stop[-1]['timestamp'],
                'points': len(current_stop)
            })
    
    return stops

def split_cluster_by_time(stops):
    """Split a cluster into sub-clusters based on time gaps."""
    stops = sorted(stops, key=lambda x: x['start'])
    sub_clusters = []
    current_sub_cluster = []
    
    for stop in stops:
        if not current_sub_cluster:
            current_sub_cluster.append(stop)
        else:
            last_stop = current_sub_cluster[-1]
            time_gap = stop['start'] - last_stop['end']
            if time_gap <= MAX_CLUSTER_DURATION:
                current_sub_cluster.append(stop)
            else:
                sub_clusters.append(current_sub_cluster)
                current_sub_cluster = [stop]
    
    if current_sub_cluster:
        sub_clusters.append(current_sub_cluster)
    
    return sub_clusters

def group_clusters_by_location(clusters):
    """Group clusters by location using DBSCAN."""
    coords = np.radians(np.array([[c['lat'], c['lon']] for c in clusters]))
    eps_rad = LOCATION_GROUPING_THRESHOLD / EARTH_RADIUS
    labels = DBSCAN(eps=eps_rad, min_samples=1, metric='haversine').fit_predict(coords)
    
    grouped_clusters = {}
    for label, cluster in zip(labels, clusters):
        grouped_clusters.setdefault(label, []).append(cluster)
    
    return grouped_clusters

def get_amenity_details(lat, lon, radius=100):
    """Get amenities with names and filter unwanted types."""
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    node["amenity"](around:{radius},{lat},{lon});
    out;
    """
    try:
        response = requests.post(overpass_url, data=query, timeout=10)
        elements = response.json().get('elements', [])
        
        amenities = []
        for elem in elements:
            amenity_type = elem['tags'].get('amenity', '')
            name = elem['tags'].get('name', '').strip()
            
            if amenity_type in EXCLUDED_AMENITIES or not name:
                continue
                
            amenities.append({
                'type': amenity_type,
                'name': name,
                'original_type': amenity_type
            })
        
        return amenities
    
    except Exception as e:
        print(f"OSM query failed: {e}")
        return []

def create_popup_content(cluster_group):
    """Create detailed popup with timing and point information for all sub-clusters."""
    time_format = "%Y-%m-%d %H:%M:%S"
    content = []
    
    for i, cluster in enumerate(cluster_group):
        first_seen = cluster['first_seen'].strftime(time_format)
        last_seen = cluster['last_seen'].strftime(time_format)
        total_duration = str(cluster['total_duration'])
        
        content.extend([
            f"<b>Sub-Cluster {i + 1}:</b>",
            f"First Seen: {first_seen}",
            f"Last Seen: {last_seen}",
            f"Total Duration: {total_duration}",
            f"GPS Points: {cluster['total_points']}",
            "<hr>"
        ])
    
    amenities = get_amenity_details(cluster_group[0]['lat'], cluster_group[0]['lon'])
    if amenities:
        amenity_groups = {}
        for a in amenities:
            key = (a['type'], a['original_type'])
            amenity_groups.setdefault(key, []).append(a['name'])
        
        # Add named amenities
        named_added = False
        for (display_type, original_type), names in amenity_groups.items():
            if original_type in ['restaurant', 'cafe', 'bar', 'fast_food', 'shop']:
                unique_names = sorted(list(set(names)))
                content.append(f"<b>{display_type.title()}:</b>")
                content += [f"- {name}" for name in unique_names[:3]]
                named_added = True
        
        # Add other amenities as counts
        other_amenities = {}
        for (display_type, original_type), names in amenity_groups.items():
            if original_type not in ['restaurant', 'cafe', 'bar', 'fast_food', 'shop']:
                other_amenities[display_type] = len(names)
        
        if other_amenities:
            content.append("<b>Other Amenities:</b>")
            content += [f"{k}: {v}" for k, v in sorted(other_amenities.items(), key=lambda x: -x[1])]
    
    return '<br>'.join(content) if content else "No significant amenities found"

def main():
    # Read and preprocess data
    print("Reading data...")
    df = pd.read_csv('4.txt', sep=';', header=None, names=['id', 'timestamp', 'geom'])
    df['lat'], df['lon'] = zip(*df['geom'].apply(parse_point))
    df = df.dropna(subset=['lat', 'lon'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['id', 'timestamp'])
    
    # Process data
    taxi_groups = [group for _, group in df.groupby('id')]
    print(f"Processing {len(taxi_groups)} taxis...")
    all_stops = Parallel(n_jobs=-1)(
        delayed(process_taxi)(group) for group in tqdm(taxi_groups, desc="Taxis processed")
    )
    all_stops = [stop for taxi_stops in all_stops for stop in taxi_stops]
    
    if not all_stops:
        raise ValueError("No stops detected. Adjust noise/duration thresholds.")
    
    # Cluster stops
    coords = np.radians(np.array([[s['lat'], s['lon']] for s in all_stops]))
    eps_rad = DBSCAN_EPS / EARTH_RADIUS
    labels = DBSCAN(eps=eps_rad, min_samples=MIN_SAMPLES, metric='haversine').fit_predict(coords)
    
    # Split clusters by time
    clusters = {}
    for label, stop in zip(labels, all_stops):
        if label != -1:
            clusters.setdefault(label, []).append(stop)
    
    sub_clusters = []
    for label, stops in clusters.items():
        sub_clusters.extend(split_cluster_by_time(stops))
    
    # Calculate metadata for each sub-cluster
    cluster_data = []
    for sub_cluster in sub_clusters:
        cluster_data.append({
            'lat': np.mean([s['lat'] for s in sub_cluster]),
            'lon': np.mean([s['lon'] for s in sub_cluster]),
            'first_seen': min(s['start'] for s in sub_cluster),
            'last_seen': max(s['end'] for s in sub_cluster),
            'total_duration': max(s['end'] for s in sub_cluster) - min(s['start'] for s in sub_cluster),
            'total_points': sum(s['points'] for s in sub_cluster)
        })
    
    # Group sub-clusters by location
    grouped_clusters = group_clusters_by_location(cluster_data)
    
    # Create map
    map_rome = folium.Map(location=[41.9028, 12.4964], zoom_start=12)
    for label, cluster_group in tqdm(grouped_clusters.items(), desc="Adding markers"):
        avg_lat = np.mean([c['lat'] for c in cluster_group])
        avg_lon = np.mean([c['lon'] for c in cluster_group])
        
        popup_content = create_popup_content(cluster_group)
        popup = folium.Popup(popup_content, max_width=450)
        
        folium.Marker(
            location=[avg_lat, avg_lon],
            popup=popup,
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(map_rome)
    
    map_rome.save('taxi_stopsSTACKED.html')
    print("Map saved to taxi_stopsSTACKED.html")

if __name__ == '__main__':
    main()