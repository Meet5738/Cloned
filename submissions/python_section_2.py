#Answer : 9
import pandas as pd
import numpy as np

df = pd.read_csv('dataset-2.csv')

def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    
    locations = pd.concat([df['id_start'], df['id_end']]).unique()
    
    distance_matrix = pd.DataFrame(np.inf, index=locations, columns=locations)
    np.fill_diagonal(distance_matrix.values, 0)
    
    for _, row in df.iterrows():
        start, end, distance = row['id_start'], row['id_end'], row['distance']
        distance_matrix.at[start, end] = distance
        distance_matrix.at[end, start] = distance 
    
    # Apply Floyd-Warshall Algorithm to calculate cumulative distances
    for k in locations:
        for i in locations:
            for j in locations:
                distance_matrix.at[i, j] = min(distance_matrix.at[i, j], distance_matrix.at[i, k] + distance_matrix.at[k, j])
    
    return distance_matrix

distance_matrix = calculate_distance_matrix(df)
print(distance_matrix)

#Answer : 10
import pandas as pd

def unroll_distance_matrix(distance_matrix: pd.DataFrame) -> pd.DataFrame:
    data = []

    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:  
                distance = distance_matrix.at[id_start, id_end]
                data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})
    
    result_df = pd.DataFrame(data)
    
    return result_df

unrolled_matrix = unroll_distance_matrix(distance_matrix)
print(unrolled_matrix)

#Answer : 11
import pandas as pd
unrolled_df = unroll_distance_matrix(distance_matrix)

def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: str) -> list:
    
    # Calculate the average distance for the reference ID
    reference_distances = df[df['id_start'] == reference_id]['distance']
    reference_avg = reference_distances.mean()
    
    # Calculate the 10% threshold
    lower_bound = reference_avg * 0.90  
    upper_bound = reference_avg * 1.10  
    
    # Find IDs whose average distance falls within the threshold
    valid_ids = []
    
    for id_start in df['id_start'].unique():
        id_distances = df[df['id_start'] == id_start]['distance']
        id_avg = id_distances.mean()
        
        if lower_bound <= id_avg <= upper_bound:
            valid_ids.append(id_start)
    
    # Return the sorted list of valid IDs
    return sorted(valid_ids)

result = find_ids_within_ten_percentage_threshold(unrolled_df, 'A')
print(result)

#Answer : 12
import pandas as pd

def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    
    # Rate coefficients for different vehicle types
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Calculate toll rates 
    df['moto'] = df['distance'] * rates['moto']
    df['car'] = df['distance'] * rates['car']
    df['rv'] = df['distance'] * rates['rv']
    df['bus'] = df['distance'] * rates['bus']
    df['truck'] = df['distance'] * rates['truck']
    
    return df

result_df = calculate_toll_rate(unrolled_df)
print(result_df)

#Answer : 13
import pandas as pd
import datetime

def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    
    # Define discount factors
    def get_discount_factor(day: str, time: datetime.time) -> float:
        if day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:  
            if time < datetime.time(10, 0):
                return 0.8
            elif time < datetime.time(18, 0):
                return 1.2
            else:
                return 0.8
        else:  
            return 0.7

    # Sample day and time assignments
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    start_times = [datetime.time(0, 0), datetime.time(12, 0)]  
    end_times = [datetime.time(10, 0), datetime.time(18, 0), datetime.time(23, 59, 59)]

    # Create new columns and initialize with default values
    df['start_day'] = None
    df['start_time'] = None
    df['end_day'] = None
    df['end_time'] = None

    # Iterate over the DataFrame and assign days/times using index
    for index in df.index:
        day_index = index % len(days)
        start_time_index = index % len(start_times)
        end_time_index = index % len(end_times)

        df.at[index, 'start_day'] = days[day_index]
        df.at[index, 'start_time'] = start_times[start_time_index]
        df.at[index, 'end_day'] = days[day_index]  # Assuming end day is the same as start day
        df.at[index, 'end_time'] = end_times[end_time_index]
        
        # Calculate and apply toll discounts for the respective vehicle types
        discount_factor = get_discount_factor(days[day_index], start_times[start_time_index])
        for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
            df.at[index, vehicle] *= discount_factor

    return df


result_df = calculate_time_based_toll_rates(result_df)
print(result_df)
