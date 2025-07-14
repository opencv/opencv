// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2024 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_PYTHON_STREAM_SOURCE_HPP
#define OPENCV_GAPI_STREAMING_PYTHON_STREAM_SOURCE_HPP

#ifdef HAVE_OPENCV_GAPI

#include <memory>
#include <opencv2/gapi/streaming/source.hpp>
#include <opencv2/gapi/gmetaarg.hpp>

// Forward declarations to avoid Python.h inclusion in header
struct _object;
typedef _object PyObject;

namespace cv {
namespace detail {
class PyObjectHolder; // Forward declaration
}

namespace gapi {
namespace wip {

/**
 * @brief C++ bridge for Python-implemented stream sources.
 * 
 * This class implements the IStreamSource interface and bridges calls
 * to a Python object that implements the PyStreamSource protocol.
 */
class GAPI_EXPORTS PythonStreamSource : public IStreamSource
{
public:
    /**
     * @brief Construct a new Python Stream Source object
     * 
     * @param python_source Python object implementing PyStreamSource protocol
     */
    explicit PythonStreamSource(PyObject* python_source);
    
    /**
     * @brief Pull data from the Python stream source
     * 
     * @param data Output data container
     * @return true if data was successfully pulled, false if stream ended
     */
    bool pull(Data& data) override;
    
    /**
     * @brief Get metadata description of the stream
     * 
     * @return GMetaArg Metadata describing the stream output
     */
    GMetaArg descr_of() const override;
    
    /**
     * @brief Request stream source to halt/stop
     */
    void halt() override;
    
    /**
     * @brief Destructor
     */
    virtual ~PythonStreamSource();

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief Factory function to create PythonStreamSource from Python object
 * 
 * @param python_source Python object implementing PyStreamSource protocol
 * @return IStreamSource::Ptr Shared pointer to the created stream source
 */
GAPI_EXPORTS IStreamSource::Ptr make_python_src(PyObject* python_source);

} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_OPENCV_GAPI

#endif // OPENCV_GAPI_STREAMING_PYTHON_STREAM_SOURCE_HPP
