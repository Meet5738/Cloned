from typing import List

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements without using slicing or built-in reverse functions.
    """
    result = []
    for i in range(0, len(lst), n):
        # Manually reverse the sublist of n elements
        group = lst[i:i+n]
        reversed_group = []
        for j in range(len(group) - 1, -1, -1):
            reversed_group.append(group[j])
        result.extend(reversed_group)
    return result

# Example test cases
print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))  # Output: [3, 2, 1, 6, 5, 4, 8, 7]
print(reverse_by_n_elements([1, 2, 3, 4, 5], 2))           # Output: [2, 1, 4, 3, 5]
print(reverse_by_n_elements([10, 20, 30, 40, 50, 60, 70], 4))  # Output: [40, 30, 20, 10, 70, 60, 50]

from typing import List, Dict

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary where keys are string lengths,
    and values are lists of strings with the same length.
    """
    length_dict = {}
    
    for word in lst:
        length = len(word)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(word)
    
    # Sorting the dictionary by keys (lengths) to ensure the result is in ascending order of length
    return dict(sorted(length_dict.items()))

# Test cases
print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))
# Output: {3: ['bat', 'car', 'dog'], 4: ['bear'], 5: ['apple'], 8: ['elephant']}

print(group_by_length(["one", "two", "three", "four"]))
# Output: {3: ['one', 'two'], 4: ['four'], 5: ['three']}
from typing import Any, Dict

def flatten_dict(nested_dict: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.

    :param nested_dict: The dictionary object to flatten
    :param parent_key: The base string for parent keys (used for recursion)
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    flattened = {}
    
    for key, value in nested_dict.items():
        # Construct new key with parent key and separator
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        if isinstance(value, dict):  # If value is a dictionary, recurse
            flattened.update(flatten_dict(value, new_key, sep=sep))
        elif isinstance(value, list):  # If value is a list, handle each element by index
            for i, item in enumerate(value):
                flattened.update(flatten_dict({f"{key}[{i}]": item}, parent_key, sep=sep))
        else:  # Base case: value is neither dict nor list
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

from typing import List

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.

    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(start: int):
        # If we have a complete permutation, add a copy of it to the result
        if start == len(nums):
            result.append(nums[:])
            return
        
        visited = set()  # To avoid duplicate permutations in the current recursion level
        
        for i in range(start, len(nums)):
            # Skip if the element has already been considered at this position
            if nums[i] in visited:
                continue
            visited.add(nums[i])

            # Swap the current element to the current start position
            nums[start], nums[i] = nums[i], nums[start]
            
            # Recurse for the next position
            backtrack(start + 1)
            
            # Swap back to restore the original list order
            nums[start], nums[i] = nums[i], nums[start]

    nums.sort()  # Sort the list to handle duplicates effectively
    result = []
    backtrack(0)
    return result

# Example usage:
nums = [1, 1, 2]
unique_perms = unique_permutations(nums)
print(unique_perms)

import re
from typing import List

def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    :param text: A string containing dates in various formats
    :return: A list of valid dates in the formats specified
    """
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

import polyline
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two latitude-longitude points.
    
    :param lat1: Latitude of the first point
    :param lon1: Longitude of the first point
    :param lat2: Latitude of the second point
    :param lon2: Longitude of the second point
    :return: Distance in meters between the two points
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
    
    :param polyline_str: The encoded polyline string
    :return: A Pandas DataFrame with latitude, longitude, and distance columns
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

from typing import List

def rotate_and_transform_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the matrix by 90 degrees clockwise and replace each element with 
    the sum of all elements in the same row and column, excluding itself.

    :param matrix: 2D list representing the square matrix (n x n)
    :return: Transformed matrix
    """
    n = len(matrix)
    
    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    
    # Step 2: Replace each element with the sum of all elements in the same row and column, excluding itself
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            # Row sum excluding the current element
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            # Column sum excluding the current element
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            # Set the final value for this element
            final_matrix[i][j] = row_sum + col_sum
    
    return final_matrix

# Example usage:
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
final_matrix = rotate_and_transform_matrix(matrix)

# Print the result
for row in final_matrix:
    print(row)



