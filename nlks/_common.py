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


class Bunch(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join(['{}: {}'.format(k.rjust(m), type(v))
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())
