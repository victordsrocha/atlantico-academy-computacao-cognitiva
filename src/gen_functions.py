import numpy as np


def dist_euclid(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculates the Euclidean distance of two points in any dimension
    
    Args:
        point1: a numpy array containing the coordinates of the first point
        point2: a numpy array containing the coordinates of the second point

    Returns:
        The Euclidean distance of the points
    """
    dist = np.linalg.norm(point1 - point2)
    return dist


def closest_to_point(points_list: np.ndarray, focus_point: np.ndarray) -> np.ndarray:
    """
    Calculates which point is closest to the focus point from a list of points.

    Args:
        points_list: a numpy array containing the coordinates of the points.
        focus_point: a numpy array containing the coordinate of the focus point.

    Returns:
        a numpy array containing the coordinate of the closest point
    """

    closest_point_distance = float('inf')
    closest_point = None
    for point in points_list:
        current_point_distance = dist_euclid(focus_point, point)
        if current_point_distance < closest_point_distance:
            closest_point_distance = current_point_distance
            closest_point = point
    return closest_point


def sort_string_list_by_length(string_list: list[str], descending: bool = False) -> list[str]:
    """
    Sorts a list of strings by the number of elements in each item

    Args:
        string_list: list of strings
        descending: true for descending order and false otherwise

    Returns:
        Returns the sorted list
    """
    sorted_list = string_list.copy()
    if descending:
        sorted_list.sort(key=lambda x: len(x), reverse=True)
    else:
        sorted_list.sort(key=lambda x: len(x))
    return sorted_list


def occurrences(string_list: list[str], string_key: str) -> int:
    """
    Finds the number of elements equal to the string_key in the string_list

    Args:
        string_list: list of strings
        string_key: string

    Returns:
        Number of elements equal to the string_key in the string_list
    """
    return string_list.count(string_key)


def map_occurrences(string_list: list[str]) -> dict[str, int]:
    """
    Calculates the count of each string in a list of strings.

    Args:
        string_list: list of strings.

    Returns:
        A dictionary containing each string with its count.
    """
    string_count = {}
    for item in string_list:
        if item not in string_count:
            count = string_list.count(item)
            string_count[item] = count
    return string_count


def string_proximity(string_list: list[str], analyzed_string: str, threshold: int) -> list[str]:
    """
    Performs a string proximity analysis on a list of strings.

    Args:
        string_list:
            A list of strings.
        analyzed_string:
            Analyzed string.
        threshold: 
            An integer indicating the analysis distance threshold.
    Returns:
        A list containing the previous and subsequent items of each instance of the analyzed string, without
        repetitions, according to the chosen threshold.

        Example:
            string_list : ['avocado', 'pear', 'grape', 'banana', 'apple' , 'cabbage', 'grape', 'bean', 'rice']
            analysis string: banana; threshold: 2
            returns: [ 'pear', 'grape', 'apple' , 'cabbage' ]
    """

    # finds the position of the instances of the analysis string
    analyzed_string_indexes = []
    for i in range(len(string_list)):
        if string_list[i] == analyzed_string:
            analyzed_string_indexes.append(i)

    # finds the position of the strings within the distance threshold
    threshold_indexes = set()
    for i in range(1, threshold + 1):
        for p in analyzed_string_indexes:
            new_threshold_index = p - i
            if new_threshold_index >= 0:
                threshold_indexes.add(new_threshold_index)
            new_threshold_index = p + i
            if new_threshold_index <= (len(string_list) - 1):
                threshold_indexes.add(new_threshold_index)

    # sorts the positions
    threshold_indexes = list(threshold_indexes)
    threshold_indexes.sort()

    # creates a list with the positions found in the previous steps
    result_list = []
    for p in threshold_indexes:
        if string_list[p] not in result_list:
            result_list.append(string_list[p])

    return result_list


def removes_repetitions_from_list(input_list: list) -> list:
    """
    Creates a list from the input list with no repeated items.

    Args:
        input_list:
            A list with elements of any type.

    Returns:
        The input list without repetitions.
    """
    no_repeat_list = input_list.copy()
    no_repeat_list.reverse()
    i = 0
    while i < (len(input_list) - 1):
        current_item = input_list[i]
        while no_repeat_list.count(current_item) > 1:
            no_repeat_list.remove(current_item)
        i += 1
    no_repeat_list.reverse()
    return no_repeat_list


def region_of_interest(matrix: np.ndarray, roi_center: tuple[int, int], roi_shape: tuple[int, int]) -> np.ndarray:
    """
    Creates a region of interest from a 2D numpy.ndarray (ROI)

    Args:
        matrix:
            A numpy.ndarray with two dimensions.
        roi_center:
            A tuple of integers containing the coordinates of the center of the region of interest.
            For example, if the center is in row 3 and column 4, roi_center should get (3,4)
        roi_shape:
            A tuple of integers containing the dimensions of the region of interest. The dimensions must be odd.
            For example, if the region of interest has height 3 and width 5, roi_shape should get (3,5)

    Returns:
        An ROI matrix (a cutout of the original matrix).
    """

    # TODO: tratar o caso no qual as dimensões da entrada não são ímpares.
    # TODO: tratar o caso no qual a ROI extrapola a matriz de entrada.

    roi_center_x, roi_center_y = roi_center
    roi_height, roi_width = roi_shape
    roi = matrix[
          roi_center_x - (roi_width - 1) // 2: 1 + roi_center_x + (roi_width - 1) // 2,
          roi_center_y - (roi_height - 1) // 2: 1 + roi_center_y + (roi_height - 1) // 2]
    return roi


def region_of_interest_list(matrix: np.ndarray, roi_center_list: list[tuple[int, int]],
                            roi_shape_list: list[tuple[int, int]]) -> list[np.ndarray]:
    """
    Creates multiple ROIs from a matrix.

    Args:
        matrix:
            A numpy.ndarray with two dimensions.
        roi_center_list:
            A list containing tuples. Each tuple is composed of integers containing the coordinates of the
            center of each region of interest.
        roi_shape_list:
            A list containing tuples. Each tuple is composed of integers containing the dimensions of each
            region of interest.

    Returns:
        A list of ROIs.
    """
    roi_list = []
    for i in range(len(roi_center_list)):
        new_roi = region_of_interest(matrix, roi_center_list[i], roi_shape_list[i])
        roi_list.append(new_roi)
    return roi_list


def matrix_thresholding(matrix: np.ndarray, threshold: float) -> np.ndarray:
    """
    Assigns the value zero for all elements of a matrix smaller than a given
    threshold and the value one otherwise.

    Args:
        matrix: a numpy.ndarray with two dimensions
        threshold: float

    Returns:
        A binary matrix with the same dimensions as the input matrix,
        assigning the value zero for all elements smaller than the threshold and 1 otherwise.
    """
    return np.vectorize(lambda x: 1 if x >= threshold else 0)(matrix)


def concatenate_lists_without_repetitions(*args):
    concatenated_list = []
    for arg in args:
        concatenated_list += arg
    return removes_repetitions_from_list(concatenated_list)


def convert_list_to_string(_corpus, seperator=' '):
    """
    Converte uma lista de strings em uma string unica, separando os itens com um espaço em branco.
    """
    return seperator.join(_corpus)
