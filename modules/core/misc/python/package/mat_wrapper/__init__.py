__all__ = []

import numpy as np
import cv2 as cv
from typing import TYPE_CHECKING, Any

# Same as cv2.typing.NumPyArrayNumeric, but avoids circular dependencies
if TYPE_CHECKING:
    _NumPyArrayNumeric = np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]]
else:
    _NumPyArrayNumeric = np.ndarray

# NumPy documentation: https://numpy.org/doc/stable/user/basics.subclassing.html


class Mat(_NumPyArrayNumeric):
    '''
    cv.Mat wrapper for numpy array.

    Stores extra metadata information how to interpret and process of numpy array for underlying C++ code.
    '''

    def __new__(cls, arr, **kwargs):
        obj = arr.view(Mat)
        return obj

    def __init__(self, arr, **kwargs):
        self.wrap_channels = kwargs.pop('wrap_channels', getattr(arr, 'wrap_channels', False))
        if len(kwargs) > 0:
            raise TypeError('Unknown parameters: {}'.format(repr(kwargs)))

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.wrap_channels = getattr(obj, 'wrap_channels', None)


Mat.__module__ = cv.__name__
cv.Mat = Mat
cv._registerMatType(Mat)
