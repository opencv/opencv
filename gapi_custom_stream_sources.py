#!/usr/bin/env python3
"""
OpenCV G-API Custom Stream Sources for Python

This module provides the base interface and utilities for creating custom
stream sources in Python for use with OpenCV G-API streaming computations.
"""

import cv2 as cv
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Any, Union, Optional

class PyStreamSource(ABC):
    """
    Abstract base class for custom Python stream sources.
    
    Subclass this to create custom stream sources that can be used with
    G-API streaming computations. The source provides data to the G-API
    pipeline on demand.
    
    Example:
        class MyCustomSource(PyStreamSource):
            def pull(self):
                # Generate or fetch data
                img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                return True, img
            
            def descr_of(self):
                return cv.gapi.descr_of(np.zeros((480, 640, 3), dtype=np.uint8))
        
        # Use with G-API
        source = cv.gapi.wip.make_python_src(MyCustomSource())
        compiled.setSource(cv.gin(source))
    """
    
    @abstractmethod
    def pull(self) -> Tuple[bool, Any]:
        """
        Pull the next data item from the stream.
        
        This method is called by the G-API framework when it needs new data
        from the stream. It should return a tuple containing a success flag
        and the data.
        
        Returns:
            tuple: (success, data) where:
                - success (bool): True if data was successfully retrieved,
                                  False if the stream has ended
                - data (Any): The data to pass to the G-API pipeline. Can be:
                    * cv.Mat for single image streams
                    * tuple of values for multi-input streams  
                    * np.ndarray (will be converted to cv.Mat)
                    * None if stream ended (success should be False)
        
        Raises:
            Exception: Any exception raised will be propagated to the G-API
                      framework and may cause the pipeline to fail.
        """
        pass
    
    @abstractmethod  
    def descr_of(self):
        """
        Return metadata description of the stream output.
        
        This method should return metadata that describes the type and shape
        of data that will be produced by pull(). This is used by G-API for
        pipeline compilation and optimization.
        
        Returns:
            cv.GMetaArg: Metadata describing the stream output. Use
                        cv.gapi.descr_of() to create appropriate metadata
                        from example data.
        
        Example:
            def descr_of(self):
                # For single Mat output
                return cv.gapi.descr_of(np.zeros((480, 640, 3), dtype=np.uint8))
                
            def descr_of(self):
                # For multi-input output
                return cv.GIn(
                    cv.gapi.descr_of(np.zeros((480, 640, 3), dtype=np.uint8)),
                    cv.gapi.descr_of(np.zeros((10,), dtype=np.float32))
                )
        """
        pass
    
    def halt(self):
        """
        Request the stream source to halt/stop (optional override).
        
        This method is called when the G-API pipeline is being stopped.
        Override this method if your source needs to perform cleanup,
        close connections, or stop background processes.
        
        The default implementation does nothing.
        """
        pass


class RandomImageSource(PyStreamSource):
    """
    Example implementation: generates random images.
    
    This is a simple example source that generates random color images
    of a specified size for a given number of frames.
    """
    
    def __init__(self, width: int, height: int, count: int, channels: int = 3):
        """
        Initialize the random image source.
        
        Args:
            width (int): Image width in pixels
            height (int): Image height in pixels  
            count (int): Number of images to generate before ending stream
            channels (int): Number of color channels (1 or 3)
        """
        self.width = width
        self.height = height
        self.count = count
        self.channels = channels
        self.generated = 0
        
        if channels not in [1, 3]:
            raise ValueError("channels must be 1 (grayscale) or 3 (color)")
    
    def pull(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.generated >= self.count:
            return False, None
        
        if self.channels == 1:
            shape = (self.height, self.width)
        else:
            shape = (self.height, self.width, self.channels)
            
        img = np.random.randint(0, 255, shape, dtype=np.uint8)
        self.generated += 1
        return True, img
    
    def descr_of(self):
        if self.channels == 1:
            sample = np.zeros((self.height, self.width), dtype=np.uint8)
        else:
            sample = np.zeros((self.height, self.width, self.channels), dtype=np.uint8)
        return cv.gapi.descr_of(sample)


class CounterSource(PyStreamSource):
    """
    Example implementation: generates incrementing counter values.
    
    This source generates integer counter values, useful for testing
    or as a simple data generator.
    """
    
    def __init__(self, start: int = 0, end: int = 100, step: int = 1):
        """
        Initialize the counter source.
        
        Args:
            start (int): Starting counter value
            end (int): Ending counter value (exclusive)
            step (int): Increment step
        """
        self.current = start
        self.end = end
        self.step = step
    
    def pull(self) -> Tuple[bool, Optional[int]]:
        if self.current >= self.end:
            return False, None
        
        value = self.current
        self.current += self.step
        return True, value
    
    def descr_of(self):
        return cv.gapi.descr_of(0)  # int metadata


class ListSource(PyStreamSource):
    """
    Example implementation: streams data from a Python list.
    
    This source iterates through a pre-defined list of data items,
    useful for testing with known data sets.
    """
    
    def __init__(self, data_list):
        """
        Initialize the list source.
        
        Args:
            data_list: List of data items to stream
        """
        self.data_list = data_list
        self.index = 0
    
    def pull(self) -> Tuple[bool, Any]:
        if self.index >= len(self.data_list):
            return False, None
        
        data = self.data_list[self.index]
        self.index += 1
        return True, data
    
    def descr_of(self):
        if not self.data_list:
            raise ValueError("Cannot determine metadata from empty list")
        
        # Use first item to determine metadata
        sample = self.data_list[0]
        return cv.gapi.descr_of(sample)


# Factory function (to be exposed via Python bindings)
def make_python_src(source_instance: PyStreamSource):
    """
    Create a G-API stream source from a Python object.
    
    This function creates a stream source that can be used with G-API
    streaming computations from a Python object implementing the
    PyStreamSource interface.
    
    Args:
        source_instance: Instance of PyStreamSource subclass
        
    Returns:
        Stream source compatible with G-API streaming compilation
        
    Example:
        source = make_python_src(RandomImageSource(640, 480, 100))
        
        g_in = cv.GMat()
        g_out = cv.gapi.medianBlur(g_in, 3)
        comp = cv.GComputation(g_in, g_out)
        
        compiled = comp.compileStreaming()
        compiled.setSource(cv.gin(source))
        compiled.start()
    """
    if not isinstance(source_instance, PyStreamSource):
        raise TypeError("source_instance must be a PyStreamSource subclass")
    
    # This will be implemented via Python bindings to call the C++ factory
    # For now, raise NotImplementedError to indicate this needs C++ bridge
    raise NotImplementedError(
        "make_python_src requires C++ Python bindings implementation. "
        "This function should be exposed via cv.gapi.wip.make_python_src()"
    )


if __name__ == "__main__":
    # Example usage and testing
    print("OpenCV G-API Custom Stream Sources")
    print("==================================")
    
    # Test RandomImageSource
    print("\nTesting RandomImageSource:")
    source = RandomImageSource(320, 240, 5)
    print(f"Metadata: {source.descr_of()}")
    
    for i in range(7):  # Test beyond count limit
        success, data = source.pull()
        if success:
            print(f"Frame {i}: Generated {data.shape} image")
        else:
            print(f"Frame {i}: Stream ended")
            break
    
    # Test CounterSource  
    print("\nTesting CounterSource:")
    counter = CounterSource(0, 5)
    print(f"Metadata: {counter.descr_of()}")
    
    for i in range(7):  # Test beyond count limit
        success, data = counter.pull()
        if success:
            print(f"Counter {i}: {data}")
        else:
            print(f"Counter {i}: Stream ended")
            break
    
    # Test ListSource
    print("\nTesting ListSource:")
    test_data = [
        np.array([[1, 2], [3, 4]], dtype=np.int32),
        np.array([[5, 6], [7, 8]], dtype=np.int32),
        np.array([[9, 10], [11, 12]], dtype=np.int32)
    ]
    list_source = ListSource(test_data)
    print(f"Metadata: {list_source.descr_of()}")
    
    for i in range(5):  # Test beyond list length
        success, data = list_source.pull()
        if success:
            print(f"List item {i}: {data.tolist()}")
        else:
            print(f"List item {i}: Stream ended")
            break
