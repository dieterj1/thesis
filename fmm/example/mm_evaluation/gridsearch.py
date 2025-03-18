from fmm import Network, NetworkGraph, FastMapMatch, FastMapMatchConfig, UBODT
import pandas as pd
from shapely.wkt import loads
from shapely.geometry import LineString
import tracers as tr
import numpy as np
import os
from itertools import product

# Load network data and graph 
network = Network("fmm/example/osmnx_example/rome/edges.shp", "fid", "u", "v")
print("Nodes {} edges {}".format(network.get_node_count(), network.get_edge_count()))
graph = NetworkGraph(network)

# Load UBODT
ubodt = UBODT.read_ubodt_csv("fmm/example/osmnx_example/rome/ubodt.txt")
model = FastMapMatch(network, graph, ubodt)

root_path = '/mnt/d/romataxi_preprocessed_T30'

METER_PER_DEGREE = 109662.80313373724

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
            print(f"Map matching succeeded for user {user_id}")
            ground_truths.append(loads(result.mgeom.export_wkt()))
            traces.append(trace)
    
    return traces, ground_truths
        
def mapmatch_perturbated(perturbed_wkt, gt, k, radius, gps_error, reverse_tolerance, perturbation, space_noise, time_min_period):
    if perturbation:
        #Custom noise distribution
        fmm_config_pert = FastMapMatchConfig(k, radius, space_noise, reverse_tolerance, perturbation)
    else:
        #Gaussian noise distribution
        fmm_config_pert = FastMapMatchConfig(k, radius, gps_error, reverse_tolerance, perturbation)

    result_pert = model.match_wkt(perturbed_wkt, fmm_config_pert)
    
    if result_pert.cpath: 
        perturbed_geom = loads(result_pert.mgeom.export_wkt())
        
        # Calculate intersection length
        intersection = gt.intersection(perturbed_geom).length
        gt_length = gt.length
        mm_length = perturbed_geom.length
        if mm_length > 0 and gt_length > 0:
            return intersection / max(mm_length, gt_length)
        
    return None

# parameter grid
k_range = [5, 7, 9, 11, 13, 15, 17]
space_noise_range = [20, 30, 40, 50, 60, 70, 80, 90, 100]
time_min_period_range = [30]
gps_error_range = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210]

best_changed_model = {}  
best_original_model = {} 

#user limit
for user_id in range(193,262):  
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
                accuracy = mapmatch_perturbated(
                    perturbed_wkt, gt, k, 400 / METER_PER_DEGREE, 0,  
                    0.5, True, space_noise, time_min_period)

                if accuracy is not None:
                    key = (space_noise, time_min_period, k)  
                    if key not in best_changed_model:
                        best_changed_model[key] = (accuracy, 1)  # (sum_accuracy, count)
                    else:
                        sum_accuracy, count = best_changed_model[key]
                        best_changed_model[key] = (sum_accuracy + accuracy, count + 1)

            # Perturbation=False uses gps_error (gaussian distribution) instead of space_noise
            for gps_error in gps_error_range:
                for k in k_range:
                    accuracy = mapmatch_perturbated(
                        perturbed_wkt, gt, k, 400 / METER_PER_DEGREE, gps_error / METER_PER_DEGREE,
                        0.5, False, 0, time_min_period)

                    if accuracy is not None:
                        key = (space_noise, time_min_period, gps_error, k)  
                        if key not in best_original_model:
                            best_original_model[key] = (accuracy, 1)  # (sum_accuracy, count)
                        else:
                            sum_accuracy, count = best_original_model[key]
                            best_original_model[key] = (sum_accuracy + accuracy, count + 1)

# Final results for both models
original_model_results = {}
changed_model_results = {}

# Calculate average accuracy for changed model
for key in sorted(best_changed_model.keys()):
    sum_acc, count = best_changed_model[key]
    accuracy = sum_acc / count
    if (key[0], key[1]) not in changed_model_results:
        changed_model_results[(key[0], key[1])] = accuracy, key[2]
    elif accuracy > changed_model_results[(key[0], key[1])][0]:
        changed_model_results[(key[0], key[1])] = accuracy, key[2]

# Calculate average accuracy for original model
for key in sorted(best_original_model.keys()):
    sum_acc, count = best_original_model[key]
    accuracy = sum_acc / count
    if (key[0], key[1]) not in original_model_results:
        original_model_results[(key[0], key[1])] = accuracy, key[3], key[2]
    elif accuracy > original_model_results[(key[0], key[1])][0]:
        original_model_results[(key[0], key[1])] = accuracy, key[3], key[2]

# Print final results
for key in original_model_results:
    accuracy, k, stdev = original_model_results[key]
    print(f"Original model ({key[0]},{key[1]}): best accuracy: {accuracy} k: {k}, stdev: {stdev}")
    accuracy, k = changed_model_results[key]
    print(f"Changed model  ({key[0]},{key[1]}): best accuracy: {accuracy} k: {k}\n")

print("Traces count (approximate): ",count)