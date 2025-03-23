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
NOISE_THRESHOLD = 20  # meters
MIN_STOP_DURATION = 10  # seconds
DBSCAN_EPS = 80  # meters
MIN_SAMPLES = 2
EARTH_RADIUS = 6371000  # meters
MAX_CLUSTER_DURATION = timedelta(hours=3)  # Maximum duration for a sub-cluster
LOCATION_GROUPING_THRESHOLD = 50  # meters
ROUTE_GAP_THRESHOLD = timedelta(minutes=5)  # Time gap to split route

EXCLUDED_AMENITIES = {
    'bench', 'waste_basket', 'drinking_water', 'clock',
    'motorcycle_parking', 'toilets', 'telephone', 'post_box'
}

# Boolean to control full route mapping
map_full_route = False  # Set to False to disable full route mapping

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
        output_path = os.path.join('.', 'rome_amenities.osm')
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print("Rome OSM data downloaded successfully.")
    else:
        print(f"Failed to download OSM data: {response.status_code}")
        raise Exception("Failed to download OSM data")

def process_osm_to_geopackage():
    """Process the OSM file and save amenities to a GeoPackage."""
    osm_file_path = os.path.join('.', 'rome_amenities.osm')
    gpkg_file_path = os.path.join('.', 'rome_amenities.gpkg')
    
    if not os.path.exists(osm_file_path):
        download_rome_osm_data()
    
    print("Processing OSM data...")
    handler = AmenityHandler()
    handler.apply_file(osm_file_path)
    
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
    gdf.to_file(gpkg_file_path, driver='GPKG')
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
    # Check if we have any clusters to process
    if not clusters:
        print("Warning: No clusters to group.")
        return {}
        
    coords = np.radians(np.array([[c['lat'], c['lon']] for c in clusters]))
    
    # Ensure we have a proper 2D array
    if coords.size == 0:
        print("Warning: Empty coordinates array.")
        return {}
    
    # Make sure coords is properly shaped for DBSCAN
    if len(coords.shape) == 1:
        print("Warning: Reshaping 1D array to 2D.")
        coords = coords.reshape(-1, 2)
    
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

def process_single_file(filepath, amenities_gdf, spatial_idx):
    """Process a single taxi trip file and return the stops."""
    print(f"Processing file: {filepath}")
    
    try:
        # Read and preprocess taxi data in the new format
        df = pd.read_csv(filepath)
        
        # Rename columns to match the rest of the code
        df = df.rename(columns={
            'ID': 'id',
            'Timestamp': 'timestamp',
            'Latitude': 'lat',
            'Longitude': 'lon'
        })
        df = df.dropna(subset=['lat', 'lon'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Print a sample of the data to help debug
        print(f"Data sample (first 3 rows from {filepath}):")
        print(df.head(3))
        
        # Process data
        taxi_groups = [group for _, group in df.groupby('id')]
        
        if not taxi_groups:
            print(f"Warning: No valid taxi groups found in {filepath}.")
            return []
            
        print(f"Found {len(taxi_groups)} taxi IDs in {filepath}")
        
        # Process each taxi group to identify stops
        file_stops = []
        for group in taxi_groups:
            stops = process_taxi(group)
            file_stops.extend(stops)
            
        print(f"Detected {len(file_stops)} potential stops in {filepath}")
        return file_stops
        
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return []

def find_trip_files(folder_path):
    """Find all trip files in the specified folder."""
    pattern = r'matched_taxi-2-rit-.*\.csv'
    trip_files = []
    
    # Use relative path from current directory
    folder_path = os.path.join('.', folder_path)
    
    print(f"Looking for files in: {folder_path}")
    print(f"Looking for pattern: {pattern}")
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist!")
        return trip_files
    
    for filename in os.listdir(folder_path):
        print(f"Checking file: {filename}")
        if re.match(pattern, filename):
            trip_files.append(os.path.join(folder_path, filename))
    
    return trip_files

def main():
    # Get folder path from command line or use default
    import sys
    folder_path = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    # Make sure we store and read files from the current directory
    current_dir = os.path.abspath('.')
    
    # Check if we need to process OSM data
    osm_gpkg_path = os.path.join(current_dir, 'rome_amenities.gpkg')
    if not os.path.exists(osm_gpkg_path):
        if not process_osm_to_geopackage():
            print("Failed to process OSM data. Exiting.")
            return
    
    # Load amenities data and build spatial index
    print("Loading amenities data...")
    amenities_gdf = gpd.read_file(osm_gpkg_path)
    spatial_idx = build_spatial_index(amenities_gdf)
    print(f"Loaded {len(amenities_gdf)} amenities.")
    
    # Find all trip files
    trip_files = find_trip_files(folder_path)
    
    if not trip_files:
        print(f"No trip files found in {folder_path}. Looking for files matching 'matched_taxi-2-rit-*.csv'")
        return
        
    print(f"Found {len(trip_files)} trip files to process")
    
    # Process all files and collect stops
    all_stops = []
    for filepath in trip_files:
        file_stops = process_single_file(filepath, amenities_gdf, spatial_idx)
        all_stops.extend(file_stops)
    
    print(f"Total stops detected across all files: {len(all_stops)}")
    
    if not all_stops:
        print("Warning: No stops detected across all files. Adjust noise/duration thresholds or check data quality.")
        print(f"NOISE_THRESHOLD: {NOISE_THRESHOLD}m, MIN_STOP_DURATION: {MIN_STOP_DURATION}s")
        return
    
    # Cluster stops
    if not all_stops:
        print("Error: No stops detected. Adjust thresholds or check input data.")
        return
        
    coords = np.radians(np.array([[s['lat'], s['lon']] for s in all_stops]))
    
    # Ensure coords is not empty
    if coords.size == 0:
        print("Error: Empty coordinates array. No valid stops found.")
        return
        
    eps_rad = DBSCAN_EPS / EARTH_RADIUS
    labels = DBSCAN(eps=eps_rad, min_samples=MIN_SAMPLES, metric='haversine').fit_predict(coords)
    
    # Split clusters by time
    clusters = {}
    for label, stop in zip(labels, all_stops):
        if label != -1:
            clusters.setdefault(label, []).append(stop)
            
    if not clusters:
        print("Warning: No clusters found after DBSCAN. Consider adjusting DBSCAN parameters.")
        print(f"Total stops: {len(all_stops)}, but none formed clusters with min_samples={MIN_SAMPLES}")
        return
    
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
    if grouped_clusters:
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
    else:
        print("Warning: No grouped clusters to add to map.")
    
    # Add full route if enabled
    if map_full_route:
        print("Adding traces...")
        
        # Process each file for traces, but limit the total number of points
        total_points = 0
        max_total_points = 10000  # Maximum total points across all files
        
        for filepath in trip_files:
            if total_points >= max_total_points:
                print(f"Reached maximum of {max_total_points} trace points. Stopping trace processing.")
                break
                
            try:
                # Read file again for trace processing
                trace_df = pd.read_csv(filepath)
                
                # Rename columns to match the rest of the code
                trace_df = trace_df.rename(columns={
                    'ID': 'id',
                    'Timestamp': 'timestamp',
                    'Latitude': 'lat',
                    'Longitude': 'lon'
                })
                
                # Calculate remaining points we can process
                points_to_take = min(len(trace_df), max_total_points - total_points)
                if points_to_take <= 0:
                    break
                    
                # Take only the points we need
                trace_df = trace_df.head(points_to_take)
                total_points += points_to_take
                
                print(f"Adding {points_to_take} trace points from {os.path.basename(filepath)} (total: {total_points})")
                
                # Process traces for this file
                for _, group in trace_df.groupby('id'):
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
            
            except Exception as e:
                print(f"Error processing traces for {filepath}: {e}")
                continue
    
    # Generate a timestamp for the output filename
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save files in the current directory
    output_file = os.path.join('.', f'taxi_stops_{timestamp}.html')
    summary_file = os.path.join('.', f'analysis_summary_{timestamp}.txt')
    
    map_rome.save(output_file)
    print(f"Map saved to {output_file}")
    
    # Also save a summary of the analysis results
    with open(summary_file, 'w') as f:
        f.write(f"Taxi Trip Analysis Summary\n")
        f.write(f"------------------------\n")
        f.write(f"Time of analysis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of trip files processed: {len(trip_files)}\n")
        f.write(f"Total stops detected: {len(all_stops)}\n")
        f.write(f"Number of clusters: {len(grouped_clusters) if grouped_clusters else 0}\n\n")
        
        f.write(f"Parameters used:\n")
        f.write(f"NOISE_THRESHOLD: {NOISE_THRESHOLD}m\n")
        f.write(f"MIN_STOP_DURATION: {MIN_STOP_DURATION}s\n")
        f.write(f"DBSCAN_EPS: {DBSCAN_EPS}m\n")
        f.write(f"MIN_SAMPLES: {MIN_SAMPLES}\n\n")
        
        f.write(f"Files processed:\n")
        for file in trip_files:
            f.write(f"- {os.path.basename(file)}\n")
    
    print(f"Analysis summary saved to {summary_file}")

if __name__ == '__main__':
    import sys
    
    # Display help if requested
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("""
Taxi Trip Analysis Tool
Usage: python script.py [folder_path]

Arguments:
  folder_path    Path to the folder containing trip files (default: current directory)
                 Files should follow the pattern 'taxi-2-rit-*.csv'

Example:
  python script.py /path/to/trip/data
  
The script will:
1. Load or download Rome amenity data
2. Process all trip files in the folder
3. Identify stops and points of interest
4. Generate an interactive map with the results
""")
        sys.exit(0)
    
    main()