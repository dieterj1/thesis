import os
import csv
import pandas as pd
import tracers as tr
from shapely.geometry import LineString
from fmm import Network, NetworkGraph, FastMapMatch, FastMapMatchConfig, UBODT
from shapely.wkt import loads

# Load network data and graph
network = Network("../osmnx_example/rome/edges.shp", "fid", "u", "v")
graph = NetworkGraph(network)
ubodt = UBODT.read_ubodt_csv("../osmnx_example/rome/ubodt.txt")
model = FastMapMatch(network, graph, ubodt)

# Configuration
k = 10
radius = 400/100_000
gps_error = 10/100_000
fmm_config = FastMapMatchConfig(k, radius, gps_error, obfuscation=False, reverse_tolerance=10)
space_noise, time_min_period = 200.0, 60

# Folders
input_folder = "taxi_2"  # Change this to your input folder
output_folder = "perturbed_matched_results"  # Change this to your output folder

# Create output folder
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each file
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        # Read file
        df = pd.read_csv(os.path.join(input_folder, filename))
        
        # Convert timestamp to epoch seconds
        df['Timestamp'] = pd.to_datetime(df['Timestamp']).astype(int) / 10**9
        
        # Sort by time
        df = df.sort_values(by='Timestamp')
        
        # Extract data
        trace = df[['Longitude', 'Latitude', 'Timestamp']].to_numpy()
        trace = [tuple(x) for x in trace]
        trace = [(x[0], x[1], int(x[2])) for x in trace]
        #trace = trace[:50]

        single_id = df['ID'].iloc[0]
        
        # Perturb trace
        perturbed_traces = tr.perturb_traces((space_noise, time_min_period), [trace])

        
        # Extract coordinates
        coordinates_p = [(lon, lat) for lon, lat, _ in perturbed_traces[0]]
        wkt_trace_p = LineString(coordinates_p)
        wkt_trace_str_p = wkt_trace_p.wkt
        
        # Map matching
        print("starting matching")
        result_raw_p = model.match_wkt(wkt_trace_str_p, fmm_config)
        pert_mm_wtk = loads(result_raw_p.mgeom.export_wkt())
        
        # Extract matched coordinates
        pert_mm_coords = list(pert_mm_wtk.coords)
        print("writing output")
        # Write output
        output_file = os.path.join(output_folder, f"matched_{filename}")
        with open(output_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["ID", "Latitude", "Longitude"])
            for lon, lat in pert_mm_coords:
                writer.writerow([single_id, lat, lon])

print("done")