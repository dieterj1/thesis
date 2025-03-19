from fmm import Network, NetworkGraph, FastMapMatch, FastMapMatchConfig, UBODT
import pandas as pd
from shapely.wkt import loads
from shapely.geometry import LineString
import numpy as np
import os
import logging
from metrics import calculate_trace_areas
from itertools import product

"""
Find the optimal FMM parameters for map matching non-perturbated traces
"""

# Load network data and graph 
network = Network("fmm/example/osmnx_example/rome/edges.shp", "fid", "u", "v")
print("Nodes {} edges {}".format(network.get_node_count(), network.get_edge_count()))
graph = NetworkGraph(network)

# Load UBODT
ubodt = UBODT.read_ubodt_csv("fmm/example/osmnx_example/rome/ubodt.txt")
model = FastMapMatch(network, graph, ubodt)

root_path = '/mnt/d/maart18maxDist500noDROPv2'

METER_PER_DEGREE = 109662.80313373724

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  
        logging.FileHandler("fmm/example/mm_evaluation/mm_parameter_tuner.txt")  
    ]
)

def process_user(user_id, k, radius, gps_error, reverse_tolerance):
    user_path = f'taxi_{user_id}'
    folder_path = os.path.join(root_path, user_path)

    if not os.path.exists(folder_path):
        print("Folder path not found")
        return None  

    total_area = 0
    trace_count = 0  

    for file in os.scandir(folder_path):  
        df = pd.read_csv(file, sep=",")
        df = df.rename(columns={"ID": "id", "Latitude": "x", "Longitude": "y", "Timestamp": "timestamp"})
        
        # Convert timestamp to epoch 
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", errors="coerce").astype(int) / 10**9
        df["timestamp"] = df["timestamp"].astype(int)
        df = df.sort_values(by="timestamp")
        trace = df[['y', 'x', 'timestamp']].to_numpy()
        trace = [tuple(x) for x in trace]
        trace = [(x[0], x[1], int(x[2])) for x in trace]
        
        # Create WKT from lat lon data
        original_linestring = LineString(zip(df["y"], df["x"]))
        original_wkt = original_linestring.wkt
        
        # Map matching on the original trajectory 
        fmm_config = FastMapMatchConfig(k, radius / METER_PER_DEGREE, gps_error / METER_PER_DEGREE, reverse_tolerance, False)
        result = model.match_wkt(original_wkt, fmm_config)
        original_linestring = loads(original_wkt)
        if result.cpath:
            result_wkt = result.mgeom.export_wkt()
            try:
                result_geom = loads(result_wkt)        
                if result_geom.geom_type != "LineString" or len(result_geom.coords) < 2:
                    print(f"Skipping user {user_id}, invalid LineString: {result_wkt}")
                    continue

            except Exception as e:
                print(f"Skipping trace, error loading WKT: {e}")
                continue 

            if (not original_linestring.is_empty and not result_geom.is_empty and original_linestring.is_valid and result_geom.is_valid and len(original_linestring.coords) > 1 and len(result_geom.coords) > 1):
                if original_linestring.geom_type != "LineString" or result_geom.geom_type != "LineString":
                    print(f"Skipping due to invalid geometry type: {original_linestring.geom_type}, {result_geom.geom_type}")
                    continue  
                
                area = calculate_trace_areas(original_linestring, result_geom)[2]
                if area > 0:
                    total_area += area
                    trace_count += 1
                    print(f"Trace of taxi {user_id}, area: {area}")
                else:
                    print(f"Trace of taxi {user_id} gives zero area!")

            else:
                print(f"Skipping due to invalid or empty geometries for user {user_id}")
    
    if trace_count > 0:
        return total_area / trace_count  
    else:
        return None  

    
# parameter grid
#k_range = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
gps_error_range = [5, 10, 15, 20, 25]
#radius_range = [200, 300, 400, 500]
#reverse_tolerance_range = [0, 0.2, 0.4, 0.6, 0.8, 1]
reverse_tolerance_range = [0,0.1,0.2,0.3,0.4,0.5]
radius_range = [300]
k_range = [5,8,11,14]

lowest_avg = float('inf') 
best_k, best_radius, best_gps_error, best_reverse_tolerance = None, None, None, None

nr_combinations = len(list(product(k_range, radius_range, gps_error_range, reverse_tolerance_range)))
i = 0

for k, radius, gps_error, reverse_tolerance in product(k_range, radius_range, gps_error_range, reverse_tolerance_range):
    total_area = 0
    count = 0
    i+=1
    print(f"Combination {i}/{nr_combinations}")
    
    for user_id in range(13, 14): 
        print(f"Calculating traces for taxi {user_id}")
        area = process_user(user_id, k, radius, gps_error, reverse_tolerance)

        if area is None:
            print(f"No valid area returned by metric for taxi {user_id}")
            continue
        else:
            total_area += area
            count += 1
        
    if count > 0:  
        avg = total_area / count
        if avg < lowest_avg:
            lowest_avg = avg
            best_k = k
            best_radius = radius
            best_gps_error = gps_error
            best_reverse_tolerance = reverse_tolerance

            logging.info(f"New lowest area: {avg} for (k={k}, r={radius}, stdev={gps_error}, reverse tol={reverse_tolerance})")
    else:
        logging.info(f"No valid traces found for (k={k}, r={radius}, stdev={gps_error}, reverse tol={reverse_tolerance})")

if best_k is not None:
    logging.info(f"Search done, lowest area: {lowest_avg} for (k={best_k}, r={best_radius}, stdev={best_gps_error}, reverse tol={best_reverse_tolerance})")
