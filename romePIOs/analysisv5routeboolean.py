import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
import folium
import re
from joblib import Parallel, delayed
from tqdm import tqdm
from datetime import timedelta
import osmium
import pyproj
import rtree
import os
from shapely.geometry import Point
import geopandas as gpd

# Constants
NOISE_THRESHOLD = 10  # meters
MIN_STOP_DURATION = 40  # seconds
DBSCAN_EPS = 50  # meters
MIN_SAMPLES = 3
EARTH_RADIUS = 6371000  # meters
MAX_CLUSTER_DURATION = timedelta(hours=3)  # Maximum duration for a sub-cluster
LOCATION_GROUPING_THRESHOLD = 50  # meters
ROUTE_GAP_THRESHOLD = timedelta(minutes=5)  # Time gap to split route

EXCLUDED_AMENITIES = {
    'bench', 'waste_basket', 'drinking_water', 'clock',
    'motorcycle_parking', 'toilets', 'telephone', 'post_box'
}

# Boolean to control full route mapping
map_full_route = True  # Set to False to disable full route mapping

# Rome bounding box coordinates
ROME_BOUNDS = {
    'min_lat': 41.8,
    'max_lat': 42.0,
    'min_lon': 12.4,
    'max_lon': 12.6
}

class AmenityHandler(osmium.SimpleHandler):
    def __init__(self):
        super(AmenityHandler, self).__init__()
        self.amenities = []
        
    def node(self, n):
        # Check if this node is within Rome's boundaries
        if (ROME_BOUNDS['min_lat'] <= n.location.lat <= ROME_BOUNDS['max_lat'] and
            ROME_BOUNDS['min_lon'] <= n.location.lon <= ROME_BOUNDS['max_lon']):
            
            # Check if this node has an amenity tag
            if 'amenity' in n.tags:
                amenity_type = n.tags.get('amenity')
                name = n.tags.get('name', '')
                
                # Skip excluded amenities and those without names
                if amenity_type in EXCLUDED_AMENITIES or not name:
                    return
                
                self.amenities.append({
                    'id': n.id,
                    'lat': n.location.lat,
                    'lon': n.location.lon,
                    'type': amenity_type,
                    'name': name,
                    'original_type': amenity_type
                })

def download_rome_osm_data():
    """Download OSM data for Rome once and save locally."""
    import requests
    
    print("Downloading OSM data for Rome...")
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:xml];
    (
      node["amenity"]({ROME_BOUNDS['min_lat']},{ROME_BOUNDS['min_lon']},{ROME_BOUNDS['max_lat']},{ROME_BOUNDS['max_lon']});
    );
    out body;
    """
    
    response = requests.post(overpass_url, data=query, timeout=120)
    
    if response.status_code == 200:
        with open('rome_amenities.osm', 'wb') as f:
            f.write(response.content)
        print("Rome OSM data downloaded successfully.")
    else:
        print(f"Failed to download OSM data: {response.status_code}")
        raise Exception("Failed to download OSM data")

def process_osm_to_geopackage():
    """Process the OSM file and save amenities to a GeoPackage."""
    if not os.path.exists('rome_amenities.osm'):
        download_rome_osm_data()
    
    print("Processing OSM data...")
    handler = AmenityHandler()
    handler.apply_file('rome_amenities.osm')
    
    if not handler.amenities:
        print("No amenities found in the OSM data.")
        return False
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        handler.amenities, 
        geometry=[Point(a['lon'], a['lat']) for a in handler.amenities],
        crs="EPSG:4326"
    )
    
    # Save to GeoPackage
    gdf.to_file('rome_amenities.gpkg', driver='GPKG')
    print(f"Processed {len(handler.amenities)} amenities and saved to GeoPackage.")
    return True

def build_spatial_index(gdf):
    """Build a spatial index for quick proximity searches."""
    idx = rtree.index.Index()
    for i, geom in enumerate(gdf.geometry):
        idx.insert(i, geom.bounds)
    return idx

def get_amenity_details_local(lat, lon, radius=100, gdf=None, spatial_idx=None):
    """Get amenities with names from local data."""
    if gdf is None:
        # Load amenities from GeoPackage if not provided
        try:
            gdf = gpd.read_file('rome_amenities.gpkg')
            spatial_idx = build_spatial_index(gdf)
        except Exception as e:
            print(f"Error loading amenities: {e}")
            return []
    
    # Convert radius from meters to degrees (approximate)
    radius_deg = radius / 111000  # ~111km per degree at the equator
    
    # Define search bounds
    bounds = (
        lon - radius_deg,  # min_x
        lat - radius_deg,  # min_y
        lon + radius_deg,  # max_x
        lat + radius_deg   # max_y
    )
    
    # Find candidates using spatial index
    candidates_idx = list(spatial_idx.intersection(bounds))
    
    if not candidates_idx:
        return []
    
    # Get candidate rows
    candidates = gdf.iloc[candidates_idx]
    
    # Create a point for the search location
    point = Point(lon, lat)
    
    # Calculate distances and filter by actual radius
    amenities = []
    for _, row in candidates.iterrows():
        # Calculate actual distance in meters
        distance = point.distance(row.geometry) * 111000  # approximate conversion
        
        if distance <= radius:
            amenities.append({
                'type': row['type'],
                'name': row['name'],
                'original_type': row['original_type']
            })
    
    return amenities

# No longer need parse_point function as coordinates are now directly provided in the CSV

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

def create_popup_content(cluster_group, amenities_gdf, spatial_idx):
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
    
    amenities = get_amenity_details_local(
        cluster_group[0]['lat'], 
        cluster_group[0]['lon'],
        gdf=amenities_gdf,
        spatial_idx=spatial_idx
    )
    
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

def split_route_by_time_gaps(points):
    """Split route into segments based on time gaps."""
    segments = []
    current_segment = []
    
    for point in points:
        if not current_segment:
            current_segment.append(point)
        else:
            last_point = current_segment[-1]
            time_gap = point['timestamp'] - last_point['timestamp']
            if time_gap <= ROUTE_GAP_THRESHOLD:
                current_segment.append(point)
            else:
                segments.append(current_segment)
                current_segment = [point]
    
    if current_segment:
        segments.append(current_segment)
    
    return segments

def main():
    # Check if we need to process OSM data
    if not os.path.exists('rome_amenities.gpkg'):
        if not process_osm_to_geopackage():
            print("Failed to process OSM data. Exiting.")
            return
    
    # Load amenities data and build spatial index
    print("Loading amenities data...")
    amenities_gdf = gpd.read_file('rome_amenities.gpkg')
    spatial_idx = build_spatial_index(amenities_gdf)
    print(f"Loaded {len(amenities_gdf)} amenities.")
    
    # Read and preprocess taxi data in the new format
    print("Reading taxi data...")
    df = pd.read_csv('taxi-2-rit-196.csv')  # Using default comma separator with headers
    # Rename columns to match the rest of the code
    df = df.rename(columns={
        'ID': 'id',
        'Timestamp': 'timestamp',
        'Latitude': 'lat',
        'Longitude': 'lon'
    })
    df = df.dropna(subset=['lat', 'lon'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
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
    
    # Add markers for clustered stops
    for label, cluster_group in tqdm(grouped_clusters.items(), desc="Adding markers"):
        avg_lat = np.mean([c['lat'] for c in cluster_group])
        avg_lon = np.mean([c['lon'] for c in cluster_group])
        
        popup_content = create_popup_content(cluster_group, amenities_gdf, spatial_idx)
        popup = folium.Popup(popup_content, max_width=450)
        
        folium.Marker(
            location=[avg_lat, avg_lon],
            popup=popup,
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(map_rome)
    
    # Add full route if enabled
    if map_full_route:
        print("Adding traces...")
        df = df.head(10000)  # Limit to 10000 points to prevent crashes
        
        for _, group in df.groupby('id'):
            points = group[['lat', 'lon', 'timestamp']].to_dict('records')
            segments = split_route_by_time_gaps(points)
            
            for segment in segments:
                # Add PolyLine for the segment
                folium.PolyLine(
                    locations=[[p['lat'], p['lon']] for p in segment],
                    color='gray',
                    weight=2,
                    opacity=0.7
                ).add_to(map_rome)
                
                # Add markers for each point in the segment
                for point in segment:
                    folium.CircleMarker(
                        location=[point['lat'], point['lon']],
                        radius=3,
                        color='red',
                        fill=True,
                        fill_color='red',
                        popup=f"Timestamp: {point['timestamp']}"
                    ).add_to(map_rome)
    
    map_rome.save('taxi_stops_local_data.html')
    print("Map saved to taxi_stops_local_data.html")

if __name__ == '__main__':
    main()