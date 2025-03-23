import pandas as pd
import folium
import os
import re
import sys
from datetime import datetime

def find_trip_files(folder_path='.'):
    """Find all trip files in the specified folder."""
    pattern = r'matched_taxi-2-rit-.*\.csv'
    trip_files = []
    
    print(f"Looking for files in: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist!")
        return trip_files
    
    for filename in os.listdir(folder_path):
        if re.match(pattern, filename):
            trip_files.append(os.path.join(folder_path, filename))
            print(f"Found file: {filename}")
    
    return trip_files

def visualize_first_1000_points(folder_path='.'):
    """Visualize the first 1000 points from taxi data files."""
    # Find trip files
    trip_files = find_trip_files(folder_path)
    
    if not trip_files:
        print(f"No trip files found in {folder_path}. Looking for files matching 'matched_taxi-2-rit-*.csv'")
        return
    
    print(f"Found {len(trip_files)} trip files")
    
    # Create a map centered on Rome
    map_rome = folium.Map(location=[41.9028, 12.4964], zoom_start=12)
    
    # Process files until we have 1000 points
    points_added = 0
    max_points = 1000
    
    for filepath in trip_files:
        if points_added >= max_points:
            break
            
        try:
            print(f"Processing file: {filepath}")
            
            # Read the CSV file
            df = pd.read_csv(filepath)
            
            # Standardize column names
            if 'Latitude' in df.columns and 'Longitude' in df.columns:
                df = df.rename(columns={
                    'ID': 'id',
                    'Timestamp': 'timestamp',
                    'Latitude': 'lat',
                    'Longitude': 'lon'
                })
            
            # Ensure required columns exist
            required_cols = ['lat', 'lon']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: Required columns {required_cols} not found in {filepath}")
                continue
                
            # Drop rows with missing coordinates
            df = df.dropna(subset=['lat', 'lon'])
            
            # Convert timestamp if it exists
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Calculate how many points we can take from this file
            points_to_take = min(len(df), max_points - points_added)
            df_subset = df.head(points_to_take)
            points_added += points_to_take
            
            print(f"Adding {points_to_take} points from {os.path.basename(filepath)} (total: {points_added})")
            
            # Add points to the map
            for _, row in df_subset.iterrows():
                popup_text = f"ID: {row.get('id', 'N/A')}<br>"
                if 'timestamp' in df.columns:
                    popup_text += f"Time: {row['timestamp']}<br>"
                popup_text += f"Coords: [{row['lat']:.6f}, {row['lon']:.6f}]"
                
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=3,
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.7,
                    popup=popup_text
                ).add_to(map_rome)
            
            # If this file has taxi IDs, add route lines for each taxi
            if 'id' in df.columns:
                for taxi_id, group in df_subset.groupby('id'):
                    # Only add line if there are multiple points
                    if len(group) > 1:
                        folium.PolyLine(
                            locations=group[['lat', 'lon']].values.tolist(),
                            color='blue',
                            weight=2,
                            opacity=0.5,
                            popup=f"Taxi ID: {taxi_id}"
                        ).add_to(map_rome)
            
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            continue
    
    # Generate timestamp for the output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join('.', f'taxi_first1000_points_{timestamp}.html')
    
    # Save the map
    map_rome.save(output_file)
    print(f"\nVisualization complete! Map saved to: {output_file}")
    print(f"Total points added: {points_added}")

if __name__ == '__main__':
    # Get folder path from command line or use default
    folder_path = sys.argv[1] if len(sys.argv) > 1 else '.'
    visualize_first_1000_points(folder_path)