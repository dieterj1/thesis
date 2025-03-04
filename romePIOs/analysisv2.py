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

# Amenities to exclude
EXCLUDED_AMENITIES = {
    'bench', 'waste_basket', 'drinking_water', 'clock',
    'motorcycle_parking', 'toilets', 'telephone', 'post_box'
}

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
            last = current_stop[-1]
            distance = great_circle((last['lat'], last['lon']), (row['lat'], row['lon'])).meters
            if distance <= NOISE_THRESHOLD:
                current_stop.append(row)
                continue
        
        if len(current_stop) >= 2:
            duration = (current_stop[-1]['timestamp'] - current_stop[0]['timestamp']).total_seconds()
            if duration >= MIN_STOP_DURATION:
                avg_lat = np.mean([p['lat'] for p in current_stop])
                avg_lon = np.mean([p['lon'] for p in current_stop])
                stops.append((avg_lat, avg_lon))
        
        current_stop = [row]
    
    if len(current_stop) >= 2:
        duration = (current_stop[-1]['timestamp'] - current_stop[0]['timestamp']).total_seconds()
        if duration >= MIN_STOP_DURATION:
            avg_lat = np.mean([p['lat'] for p in current_stop])
            avg_lon = np.mean([p['lon'] for p in current_stop])
            stops.append((avg_lat, avg_lon))
    
    return stops

def get_amenity_details(lat, lon, radius=100):
    """Get amenities with names, filtering out unwanted types."""
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
                'name': name or 'Unnamed',
                'original_type': amenity_type
            })
        
        return amenities
    
    except Exception as e:
        print(f"OSM query failed: {e}")
        return []

def create_popup_content(amenities):
    """Create organized popup content with names for relevant amenities."""
    content = []
    
    # Group by type
    amenity_groups = {}
    for a in amenities:
        key = (a['type'], a['original_type'])
        amenity_groups.setdefault(key, []).append(a['name'])
    
    # Add restaurants and shops with names first
    for (display_type, original_type), names in amenity_groups.items():
        if original_type in ['restaurant', 'cafe', 'bar', 'fast_food', 'shop']:
            unique_names = sorted(list(set(names)))
            content.append(f"<b>{display_type.title()}:</b>")
            content += [f"- {name}" for name in unique_names[:3]]  # Show top 3 unique names
    
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
        amenities = get_amenity_details(lat, lon)
        
        if not amenities:
            continue
            
        popup_content = create_popup_content(amenities)
        popup = folium.Popup(popup_content, max_width=450)
        
        folium.Marker(
            location=[lat, lon],
            popup=popup,
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(map_rome)
    
    map_rome.save('taxi_stopsv2.html')
    print("Map saved to taxi_stopsv2.html")

if __name__ == '__main__':
    main()