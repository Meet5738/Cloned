#Answer : 1
from typing import List

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements. 
    """
    result = []
    for i in range(0, len(lst), n):
        group = lst[i:i+n]
        reversed_group = []
        for j in range(len(group) - 1, -1, -1):
            reversed_group.append(group[j])
        result.extend(reversed_group)
    return result

# test cases
print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))  
print(reverse_by_n_elements([1, 2, 3, 4, 5], 2))           
print(reverse_by_n_elements([10, 20, 30, 40, 50, 60, 70], 4))  

#Answer : 2
from typing import List, Dict

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    
    length_dict = {}
    
    for word in lst:
        length = len(word)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(word)
    
    # Sorting the dictionary by keys (lengths) 
    return dict(sorted(length_dict.items()))

# Test cases
print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))

print(group_by_length(["one", "two", "three", "four"]))

# Answer : 3
from typing import Any, Dict

def flatten_dict(nested_dict: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    """
    flattened = {}
    
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        if isinstance(value, dict):  
            flattened.update(flatten_dict(value, new_key, sep=sep))
        elif isinstance(value, list):  
            for i, item in enumerate(value):
                flattened.update(flatten_dict({f"{key}[{i}]": item}, parent_key, sep=sep))
        else:  
            flattened[new_key] = value
    
    return flattened

# Example usage:
nested_dictionary = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

# Flatten the nested dictionary
flattened_dict = flatten_dict(nested_dictionary)
print(flattened_dict)

#Answer : 4
from typing import List

def unique_permutations(nums: List[int]) -> List[List[int]]:
    def backtrack(start: int):
        # If we have a complete permutation, add a copy of it to the result
        if start == len(nums):
            result.append(nums[:])
            return
        
        visited = set()  
        
        for i in range(start, len(nums)):
            if nums[i] in visited:
                continue
            visited.add(nums[i])
            nums[start], nums[i] = nums[i], nums[start]
            
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]

    nums.sort()  
    result = []
    backtrack(0)
    return result

# Example usage:
nums = [1, 1, 2]
unique_perms = unique_permutations(nums)
print(unique_perms)

#Answer : 5
import re
from typing import List

def find_all_dates(text: str) -> List[str]:
    # Define the regex patterns for each date format
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  # dd-mm-yyyy format
        r'\b\d{2}/\d{2}/\d{4}\b',  # mm/dd/yyyy format
        r'\b\d{4}\.\d{2}\.\d{2}\b'  # yyyy.mm.dd format
    ]
    
    # Compile the patterns and find all matches in the text
    all_dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        all_dates.extend(matches)
    
    return all_dates

# Example usage:
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
dates = find_all_dates(text)
print(dates)

#Answer : 6
import polyline
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two latitude-longitude points.
    """
    R = 6371000  # Radius of Earth in meters
    phi1, phi2 = radians(lat1), radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)

    a = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    """
    # Decode the polyline string into a list of (latitude, longitude) tuples
    coordinates = polyline.decode(polyline_str)

    # Create a DataFrame from the list of coordinates
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    # Initialize the distance column with zeros
    df['distance'] = 0.0

    # Calculate the distance for each row starting from the second point
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i-1, ['latitude', 'longitude']]
        lat2, lon2 = df.loc[i, ['latitude', 'longitude']]
        df.loc[i, 'distance'] = haversine(lat1, lon1, lat2, lon2)

    return df

# Example usage:
polyline_str = 'u{~vFvyys@f}@~q@dAbP'
df = polyline_to_dataframe(polyline_str)
print(df)

#Answer : 7
from typing import List

def rotate_and_transform_matrix(matrix: List[List[int]]) -> List[List[int]]:
    n = len(matrix)
    
    # Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    
    # Replace each element with the sum of all elements in the same row and column, excluding itself
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            final_matrix[i][j] = row_sum + col_sum
    
    return final_matrix

# Example usage:
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
final_matrix = rotate_and_transform_matrix(matrix)

for row in final_matrix:
    print(row)

#Answer : 8
import pandas as pd
import numpy as np

def time_check(df: pd.DataFrame) -> pd.Series:
    full_week = set(range(7))
    result = pd.Series(index=pd.MultiIndex.from_frame(df[['id', 'id_2']].drop_duplicates()), dtype=bool)
    
    for (id_, id_2), group in df.groupby(['id', 'id_2']):
        covered_days = set(group['startDay'].unique())
        
        full_week_covered = (covered_days == full_week)
        
        full_coverage = True
        for day in covered_days:
            day_group = group[group['startDay'] == day]
            
            day_group['startTimeSec'] = pd.to_timedelta(day_group['startTime']).dt.total_seconds()
            day_group['endTimeSec'] = pd.to_timedelta(day_group['endTime']).dt.total_seconds()

            day_group = day_group.sort_values(by='startTimeSec')

            last_end_time = 0
            for _, row in day_group.iterrows():
                if row['startTimeSec'] > last_end_time:
                    full_coverage = False
                    break
                last_end_time = max(last_end_time, row['endTimeSec'])

            if last_end_time < 86400:
                full_coverage = False
        
        result[(id_, id_2)] = not (full_week_covered and full_coverage)
    
    return result

