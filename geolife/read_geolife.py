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

def read_all_of_user(folder_number):
    folder_path = f"/mnt/c/Users/diete/Documents/Geolife/Data/{folder_number}/Trajectory/"
    
    plt_files = glob.glob(os.path.join(folder_path, "*.plt"))
    
    dataframes = []
    
    for plt_file in plt_files:
        try:
            df = read_plt(plt_file)
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {plt_file}: {e}")
        
    return dataframes


