import numpy as np
import pandas as pd
import glob
import os.path
import datetime
import os


def read_plt(file_path):
    df = pd.read_csv(file_path, skiprows=6, header=None, names=[
        "Latitude", "Longitude", "Unused", "Altitude", "Dayspassed", "Date", "Time"])

    # Drop unused columns
    df = df.drop(columns=["Unused", "Altitude", "Dayspassed"])

    # Make datetime from date and time
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

    # Drop original Date and Time columns
    df = df.drop(columns=["Date", "Time"])

    return df

def read_all_of_user(user_id):
    folder_name = f"{user_id:03d}"
    folder_path = f"/mnt/c/Users/diete/Documents/Geolife/Data/{folder_name}/Trajectory/"
    
    plt_files = glob.glob(os.path.join(folder_path, "*.plt"))
    
    dataframes = []
    
    for plt_file in plt_files:
        try:
            df = read_plt(plt_file)
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {plt_file}: {e}")
        
    return dataframes

# Convert geolife data to csv files to input to FMM
def all_data_to_csv():
    base_folder = "/mnt/c/Users/diete/Documents/Geolife/Data"
    
    for user_id in range(1, 182):  
        folder_name = f"{user_id:03d}"  
        folder_path = os.path.join(base_folder, folder_name, "Trajectory")
        
        if not os.path.exists(folder_path):
            print(f"User {user_id} folder not found, skipping.")
            continue

        data = read_all_of_user(user_id)

        if not data:
            print(f"No data found for user {user_id}, skipping.")
            continue 

        df = pd.concat(data, ignore_index=True)
        df["id"] = user_id  
        df = df.rename(columns={"Longitude": "x", "Latitude": "y", "Datetime": "timestamp"})
        df['timestamp'] = (df['timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        df = df.sort_values(by=["id", "timestamp"])
        df = df[["id", "x", "y", "timestamp"]]

        output_dir = f"/mnt/c/Users/diete/Documents/GeolifeCSV/{folder_name}/"
        os.makedirs(output_dir, exist_ok=True)  
        output_file = os.path.join(output_dir, "trajectory.csv")        
        df.to_csv(output_file, sep=";", index=False)
        print(f"CSV saved for user {user_id}")
