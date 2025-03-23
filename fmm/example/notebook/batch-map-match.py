#!/usr/bin/env python3
import sys
import os
import glob
import pandas as pd
import numpy as np
from shapely.geometry import LineString
from fmm import Network, NetworkGraph, UBODT, FastMapMatch, FastMapMatchConfig

def find_trip_files(folder_path):
    """Find all trip files in the given folder that match the pattern."""
    pattern = os.path.join(folder_path, "taxi-*-rit-*.csv")
    return glob.glob(pattern)

def process_single_file(filepath, model, fmm_config):
    """Process a single trip file and perform map matching."""
    try:
        # Extract taxi_id and rit_id from filename
        filename = os.path.basename(filepath)
        parts = filename.split('-')
        if len(parts) < 4:
            print(f"Skipping {filename}: filename does not match expected pattern")
            return None, None, None
        
        taxi_id = parts[1]
        rit_id = parts[3].split('.')[0]
        
        # Read the trip data
        trip_df = pd.read_csv(filepath)
        
        # Check if the file has the expected columns
        required_columns = ['ID', 'Timestamp', 'Latitude', 'Longitude']
        if not all(col in trip_df.columns for col in required_columns):
            print(f"Skipping {filename}: missing required columns")
            return None, None, None
        
        # Extract coordinates (longitude, latitude)
        coordinates = [(row['Longitude'], row['Latitude']) for _, row in trip_df.iterrows()]
        
        # Create a LineString
        if len(coordinates) < 2:
            print(f"Skipping {filename}: not enough coordinates for a LineString")
            return None, None, None
        
        wkt_trace = LineString(coordinates)
        wkt_trace_str = wkt_trace.wkt
        
        # Perform map matching
        print(f"Starting map matching for {filename}")
        result = model.match_wkt(wkt_trace_str, fmm_config)
        print(f"Map matching completed for {filename}")
        
        # Return the original dataframe along with the result and IDs
        return taxi_id, rit_id, result, trip_df
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None, None, None, None

def save_mapmatched_result(taxi_id, rit_id, result, original_df, output_folder):
    """Save the map-matched result to a file in the output folder."""
    if result is None or original_df is None:
        return
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_file = os.path.join(output_folder, f"taxi-{taxi_id}-rit-{rit_id}-mapmatched.csv")
    
    try:
        # Extract the matched geometry LineString
        if hasattr(result, 'mgeom'):
            matched_geom = result.mgeom
            
            # Check if we have a valid geometry
            if matched_geom is None:
                print(f"No matched geometry for {taxi_id}-{rit_id}, using original coordinates")
                output_df = original_df.copy()
            else:
                # Create a new dataframe with the same structure as the original
                output_df = original_df.copy()
                
                # Try to get coordinates from the matched geometry
                try:
                    # Get WKT string from the matched geometry
                    wkt_str = str(matched_geom)
                    
                    # Parse it to get coordinates - format is typically "LINESTRING (lon1 lat1, lon2 lat2, ...)"
                    # Remove "LINESTRING (" and ")" parts
                    coords_str = wkt_str.replace("LINESTRING (", "").replace(")", "")
                    
                    # Split by commas and then by spaces to get individual coordinates
                    coords_pairs = [pair.strip().split() for pair in coords_str.split(",")]
                    coords = [(float(pair[1]), float(pair[0])) for pair in coords_pairs]  # (lat, lon)
                    
                    # If number of points doesn't match, interpolate or sample
                    if len(coords) != len(original_df):
                        print(f"Warning: Number of matched points ({len(coords)}) doesn't match original ({len(original_df)}) for {taxi_id}-{rit_id}")
                        # Simple approach: use original points if count mismatch
                        if len(coords) < len(original_df):
                            print("Using original coordinates due to mismatch")
                        else:
                            # Sample the matched points to match original count
                            indices = np.linspace(0, len(coords)-1, len(original_df), dtype=int)
                            coords = [coords[i] for i in indices]
                    
                    # Update the Latitude and Longitude columns with matched coordinates
                    for i, (lat, lon) in enumerate(coords[:len(output_df)]):
                        output_df.loc[i, 'Latitude'] = lat
                        output_df.loc[i, 'Longitude'] = lon
                    
                except Exception as e:
                    print(f"Error extracting coordinates from matched geometry: {e}")
                    print("Using original coordinates")
        else:
            print(f"No 'mgeom' attribute in result for {taxi_id}-{rit_id}, using original coordinates")
            output_df = original_df.copy()
        
        # Save to CSV with the original format
        output_df.to_csv(output_file, index=False)
        
    except Exception as e:
        print(f"Error saving map-matched result: {e}")
        # Fallback to saving the original data
        original_df.to_csv(output_file, index=False)
        print(f"Saved original data instead for {taxi_id}-{rit_id}")

def main():
    # Get folder path from command line or use default
    folder_path = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    # Create output folder
    output_folder = "taxi-gt-mapmatched"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load network data and graph
    network = Network("../osmnx_example/rome/edges.shp", "fid", "u", "v")
    print(f"Nodes {network.get_node_count()} edges {network.get_edge_count()}")
    graph = NetworkGraph(network)
    
    ubodt = UBODT.read_ubodt_csv("../osmnx_example/rome/ubodt.txt")
    
    model = FastMapMatch(network, graph, ubodt)
    
    # Configuration parameters
    k = 10
    radius = 400/100_000
    gps_error = 10/100_000
    
    fmm_config = FastMapMatchConfig(k, radius, gps_error, obfuscation=False, reverse_tolerance=10)
    
    # Find all trip files
    trip_files = find_trip_files(folder_path)
    
    if not trip_files:
        print(f"No trip files found in {folder_path}. Looking for files matching 'taxi-*-rit-*.csv'")
        return
    
    print(f"Found {len(trip_files)} trip files to process")
    
    # Process all files
    for filepath in trip_files:
        taxi_id, rit_id, result, original_df = process_single_file(filepath, model, fmm_config)
        if taxi_id and rit_id and result and original_df is not None:
            save_mapmatched_result(taxi_id, rit_id, result, original_df, output_folder)
    
    print(f"All files processed. Results saved to {output_folder}")

if __name__ == "__main__":
    main()