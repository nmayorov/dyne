import numpy as np


def verify_array(array, size, name):
    array = np.asarray(array)
    if array.ndim == 2:
        array = [array] * size
    elif array.ndim == 3:
        if len(array) != size:
            raise ValueError(f"Incorrect size of {name}")
    else:
        raise ValueError(f"Incorrect shape of array {name}")
    return array


def verify_function(function, size, name):
    try:
        if len(function) != size:
            raise ValueError(f"Incorrect size of {name}")
    except TypeError:
        function = [function] * size
    return function
