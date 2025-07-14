# G-API Custom Stream Sources in Python - Design Document

## Issue #27276: Add support for custom stream sources in Python for G-API

### Problem Statement

Currently, OpenCV G-API supports custom stream sources in C++ through the `IStreamSource` interface, but Python users are limited to predefined sources like:
- `cv.gapi.wip.make_capture_src()` for video files/cameras
- `QueueSource` for programmatic data feeding

There's no straightforward way for Python developers to create custom streaming sources for scenarios like:
- Custom hardware device interfaces
- Network streaming protocols
- Database-backed data streams  
- Real-time sensor data
- Custom data transformations/generators

### Current Architecture

```cpp
// C++ IStreamSource interface
class IStreamSource: public std::enable_shared_from_this<IStreamSource>
{
public:
    using Ptr = std::shared_ptr<IStreamSource>;
    virtual bool pull(Data &data) = 0;
    virtual GMetaArg descr_of() const = 0;
    virtual void halt() = 0;
    virtual ~IStreamSource() = default;
};
```

Python currently only exposes:
```python
# Existing Python sources
source = cv.gapi.wip.make_capture_src(path)  # VideoCapture wrapper
# Limited to predefined implementations
```

### Proposed Solution

#### 1. Python Stream Source Interface

Create a Python-friendly interface that matches the C++ `IStreamSource` pattern:

```python
class PyStreamSource:
    """Base class for custom Python stream sources."""
    
    def pull(self):
        """
        Pull next data item from stream.
        
        Returns:
            tuple: (success: bool, data: Any) where data can be:
                   - cv.Mat for image streams
                   - tuple of values for multi-input streams
                   - None if stream ended
        """
        raise NotImplementedError
    
    def descr_of(self):
        """
        Return metadata description of stream output.
        
        Returns:
            cv.GMetaArg: Metadata describing the stream output type
        """
        raise NotImplementedError
    
    def halt(self):
        """Stop the stream source (optional override)."""
        pass
```

#### 2. C++ Bridge Implementation

Create a C++ wrapper that bridges Python implementations to `IStreamSource`:

```cpp
// modules/gapi/src/streaming/python_stream_source.hpp
class PythonStreamSource : public cv::gapi::wip::IStreamSource
{
private:
    cv::detail::PyObjectHolder m_python_source;
    cv::GMetaArg m_meta;
    
public:
    PythonStreamSource(PyObject* python_source);
    bool pull(cv::gapi::wip::Data& data) override;
    cv::GMetaArg descr_of() const override;
    void halt() override;
};
```

#### 3. Python Factory Function

Expose a factory function in Python:

```python
def make_python_src(source_instance):
    """
    Create a G-API stream source from Python object.
    
    Args:
        source_instance: Instance of PyStreamSource subclass
        
    Returns:
        Stream source compatible with G-API streaming compilation
    """
    return cv.gapi.wip.PythonStreamSource(source_instance)
```

### Implementation Files

#### File 1: Python Interface Definition
`modules/gapi/misc/python/pyopencv_custom_sources.hpp`

#### File 2: C++ Bridge Implementation  
`modules/gapi/src/streaming/python_stream_source.cpp`

#### File 3: Python Bindings
`modules/gapi/misc/python/shadow_gapi_custom.hpp`

#### File 4: CMake Integration
Updates to `modules/gapi/misc/python/CMakeLists.txt`

### Usage Examples

#### Example 1: Custom Image Generator
```python
class RandomImageSource(cv.gapi.PyStreamSource):
    def __init__(self, width, height, count):
        self.width = width
        self.height = height
        self.count = count
        self.generated = 0
    
    def pull(self):
        if self.generated >= self.count:
            return False, None
        
        img = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        self.generated += 1
        return True, img
    
    def descr_of(self):
        return cv.gapi.descr_of(np.zeros((self.height, self.width, 3), dtype=np.uint8))

# Usage
source = cv.gapi.wip.make_python_src(RandomImageSource(640, 480, 100))
```

#### Example 2: Network Stream Source
```python
class NetworkStreamSource(cv.gapi.PyStreamSource):
    def __init__(self, url):
        self.url = url
        self.connection = None
        self._connect()
    
    def _connect(self):
        # Custom network connection logic
        pass
    
    def pull(self):
        try:
            frame_data = self.connection.receive_frame()
            if frame_data is None:
                return False, None
            
            # Decode frame_data to cv.Mat
            frame = self._decode_frame(frame_data)
            return True, frame
        except Exception:
            return False, None
    
    def descr_of(self):
        # Return expected frame metadata
        return cv.gapi.descr_of(np.zeros((480, 640, 3), dtype=np.uint8))
    
    def halt(self):
        if self.connection:
            self.connection.close()
```

#### Example 3: Multi-Input Source  
```python
class MultiInputSource(cv.gapi.PyStreamSource):
    def __init__(self, image_source, metadata_source):
        self.image_source = image_source
        self.metadata_source = metadata_source
    
    def pull(self):
        img_success, img = self.image_source.get_next()
        meta_success, meta = self.metadata_source.get_next()
        
        if not (img_success and meta_success):
            return False, None
            
        return True, (img, meta)
    
    def descr_of(self):
        return cv.GIn(
            cv.gapi.descr_of(np.zeros((480, 640, 3), dtype=np.uint8)),
            cv.gapi.descr_of(np.zeros((10,), dtype=np.float32))
        )
```

### Integration with Existing G-API

The custom sources integrate seamlessly with existing G-API streaming:

```python
# Create custom source
source = cv.gapi.wip.make_python_src(MyCustomSource())

# Use in G-API pipeline
g_in = cv.GMat()
g_out = cv.gapi.medianBlur(g_in, 3)
comp = cv.GComputation(g_in, g_out)

# Compile for streaming
compiled = comp.compileStreaming()
compiled.setSource(cv.gin(source))
compiled.start()

# Process stream
while True:
    success, result = compiled.pull()
    if not success:
        break
    # Process result
```

### Benefits

1. **Flexibility**: Python developers can create sources for any data type or protocol
2. **Performance**: C++ bridge ensures minimal overhead  
3. **Compatibility**: Works with existing G-API streaming infrastructure
4. **Simplicity**: Pythonic interface that's easy to understand and implement
5. **Extensibility**: Foundation for community-contributed streaming sources

### Implementation Plan

1. **Phase 1**: Core infrastructure
   - C++ bridge implementation
   - Basic Python interface  
   - Simple example (random data generator)

2. **Phase 2**: Advanced features
   - Multi-input source support
   - Error handling improvements
   - Performance optimizations

3. **Phase 3**: Documentation and examples
   - Comprehensive documentation
   - Real-world usage examples
   - Performance benchmarks

### Testing Strategy

1. **Unit Tests**: Test Python-C++ bridge functionality
2. **Integration Tests**: Test with existing G-API streaming pipeline  
3. **Performance Tests**: Measure overhead vs native sources
4. **Examples**: Working examples for common use cases

This design provides a comprehensive solution for custom stream sources in Python G-API while maintaining compatibility with existing infrastructure and ensuring good performance.
