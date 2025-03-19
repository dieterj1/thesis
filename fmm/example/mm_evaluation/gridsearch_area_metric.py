from fmm import Network, NetworkGraph, FastMapMatch, FastMapMatchConfig, UBODT
import pandas as pd
from shapely.wkt import loads
from shapely.geometry import LineString
import tracers as tr
import numpy as np
import os
from metrics import calculate_trace_areas
import logging

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
        logging.FileHandler("fmm/example/mm_evaluation/gridsearch_area_metric.txt")  
    ]
)

def process_user(user_id):
    user_path = f'taxi_{user_id}'
    folder_path = os.path.join(root_path, user_path)

    if not os.path.exists(folder_path):
        print("folder path not found")
        return ([],[])
    
    traces = []
    ground_truths = []  
        
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
        fmm_config = FastMapMatchConfig(15, 400 / METER_PER_DEGREE, 10 / METER_PER_DEGREE, perturbation=False, reverse_tolerance=0.5)
        result = model.match_wkt(original_wkt, fmm_config)
        
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

            print(f"Map matching succeeded for user {user_id}")
            ground_truths.append(loads(result_wkt))
            traces.append(trace)
    
    return traces, ground_truths
        
def mapmatch_perturbated(perturbed_wkt, gt, k, radius, gps_error, reverse_tolerance, perturbation, space_noise, time_min_period):

    if perturbation:
        fmm_config_pert = FastMapMatchConfig(k, radius, space_noise, reverse_tolerance, perturbation)
    else:
        fmm_config_pert = FastMapMatchConfig(k, radius, gps_error, reverse_tolerance, perturbation)

    
    result_pert = model.match_wkt(perturbed_wkt, fmm_config_pert)
    
    if result_pert.cpath: 
        perturbed_wkt = result_pert.mgeom.export_wkt()
        
        try:
            perturbed_geom = loads(perturbed_wkt)        
    
            if perturbed_geom.geom_type != "LineString" or len(perturbed_geom.coords) < 2:
                print(f"Skipping user {user_id}, invalid LineString: {perturbed_wkt}")
                return None

        except Exception as e:
            print(f"Skipping trace, error loading WKT: {e}")
            return None 

        if (not perturbed_geom.is_empty and not gt.is_empty and perturbed_geom.is_valid and gt.is_valid and len(perturbed_geom.coords) > 1 and len(gt.coords) > 1):
            if perturbed_geom.geom_type != "LineString" or gt.geom_type != "LineString":
                print(f"Skipping due to invalid geometry type: {perturbed_geom.geom_type}, {gt.geom_type}")
                return None
            
            area = calculate_trace_areas(gt, perturbed_geom)[2]
            if area > 0:
                print(f" trace area: {area}")
                return area
            else:
                print(f"Trace  gives zero area!")

        else:
            print(f"Skipping due to invalid or empty geometries for user {user_id}")
            return None

    return None

# parameter grid
k_range = [5, 7, 9, 11, 13, 15, 17]
space_noise_range = [20, 30, 40, 50, 60, 70, 80, 90, 100]
time_min_period_range = [30]
gps_error_range = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210]

best_changed_model = {}  
best_original_model = {} 

#taxi range
for user_id in range(13,14):  
    traces, ground_truths = process_user(user_id)
    print(f"user {user_id} non-perturbated traces processed")
    if not traces or not ground_truths:
        continue

    time_min_period = 30

    # Perturb traces first (once per user)
    for space_noise in space_noise_range:
        print(f"Perturbing traces for (space_noise, time_min_period): ({space_noise}, {time_min_period})")
        perturbed_traces = tr.perturb_traces((space_noise, time_min_period), traces, picker_str='closest')

        for perturbed_trace, gt in zip(perturbed_traces, ground_truths):
            df_pert = pd.DataFrame(perturbed_trace, columns=["y", "x", "timestamp"])
            df_pert["id"] = user_id  
            perturbed_wkt = LineString(zip(df_pert["y"], df_pert["x"])).wkt

            # Perturbation=True: use specific distribution for space_noise
            for k in k_range:
                area = mapmatch_perturbated(
                    perturbed_wkt, gt, k, 400 / METER_PER_DEGREE, 0,  
                    0.5, True, space_noise, time_min_period)

                if area is not None:
                    key = (space_noise, time_min_period, k)  
                    if key not in best_changed_model:
                        best_changed_model[key] = (area, 1)  # (sum_accuracy, count)
                    else:
                        sum_area, count = best_changed_model[key]
                        best_changed_model[key] = (sum_area + area, count + 1)
                else:
                    print(f"MM failed for (space_noise,k)=({space_noise},{k})")
            # Perturbation=False uses gps_error (gaussian distribution) instead of space_noise
            for gps_error in gps_error_range:
                for k in k_range:
                    area = mapmatch_perturbated(
                        perturbed_wkt, gt, k, 400 / METER_PER_DEGREE, gps_error / METER_PER_DEGREE,
                        0.5, False, 0, time_min_period)

                    if area is not None:
                        key = (space_noise, time_min_period, gps_error, k)  
                        if key not in best_original_model:
                            best_original_model[key] = (area, 1)  
                        else:
                            sum_area, count = best_original_model[key]
                            best_original_model[key] = (sum_area + area, count + 1)
                    else:
                        print(f"MM failed for (space_noise,k,gps_error)=({space_noise},{k},{gps_error})")

# Final results for both models
original_model_results = {}
changed_model_results = {}

# Calculate average area and for changed model
for key in sorted(best_changed_model.keys()):
    sum_area, count = best_changed_model[key]
    area = sum_area / count
    if (key[0], key[1]) not in changed_model_results:
        changed_model_results[(key[0], key[1])] = area, key[2]
    elif area < changed_model_results[(key[0], key[1])][0]:
        changed_model_results[(key[0], key[1])] = area, key[2]

# Calculate average area and for original model
for key in sorted(best_original_model.keys()):
    sum_area, count = best_original_model[key]
    area = sum_area / count
    if (key[0], key[1]) not in original_model_results:
        original_model_results[(key[0], key[1])] = area, key[3], key[2]
    elif area > original_model_results[(key[0], key[1])][0]:
        original_model_results[(key[0], key[1])] = area, key[3], key[2]

for key in original_model_results:
    area, k, stdev = original_model_results[key]
    logging.info(f"Original model ({key[0]},{key[1]}): smallest area: {area} k: {k}, stdev: {stdev}")
    area, k = changed_model_results[key]
    logging.info(f"Changed model  ({key[0]},{key[1]}): smallest area: {area} k: {k}\n")

