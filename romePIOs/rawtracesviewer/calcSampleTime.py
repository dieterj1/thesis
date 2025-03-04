from datetime import datetime

# Function to calculate average time between samples
def average_time_between_samples(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract timestamps
    timestamps = []
    for line in lines:
        parts = line.split(';')
        if len(parts) > 1:
            timestamp_str = parts[1]
            # Remove the timezone information
            timestamp_str = timestamp_str.split('+')[0]  # Keep only the part before the '+'
            # Parse the timestamp using strptime
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
            timestamps.append(timestamp)

    # Calculate time differences
    time_differences = []
    for i in range(1, len(timestamps)):
        time_diff = (timestamps[i] - timestamps[i - 1]).total_seconds()
        time_differences.append(time_diff)

    # Calculate average time difference
    if time_differences:
        average_time_diff = sum(time_differences) / len(time_differences)
        return average_time_diff
    else:
        return 0

# Path to the taxi.txt file
file_path = '8.txt'
average_time = average_time_between_samples(file_path)
print(f'Average time between samples: {average_time} seconds')
