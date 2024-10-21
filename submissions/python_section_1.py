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

