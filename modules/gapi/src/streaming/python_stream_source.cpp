// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2024 Intel Corporation

#include "python_stream_source.hpp"

#ifdef HAVE_OPENCV_GAPI

#ifndef CV_PYTHON_ENABLED
// Stub implementation when Python is not available
namespace cv {
namespace gapi {
namespace wip {

PythonStreamSource::PythonStreamSource(PyObject*) {
    CV_Error(cv::Error::StsNotImplemented, "OpenCV was built without Python support");
}

bool PythonStreamSource::pull(Data&) {
    CV_Error(cv::Error::StsNotImplemented, "OpenCV was built without Python support");
}

GMetaArg PythonStreamSource::descr_of() const {
    CV_Error(cv::Error::StsNotImplemented, "OpenCV was built without Python support");
}

void PythonStreamSource::halt() {
    CV_Error(cv::Error::StsNotImplemented, "OpenCV was built without Python support");
}

PythonStreamSource::~PythonStreamSource() = default;

IStreamSource::Ptr make_python_src(PyObject*) {
    CV_Error(cv::Error::StsNotImplemented, "OpenCV was built without Python support");
}

} // namespace wip
} // namespace gapi
} // namespace cv

#else // CV_PYTHON_ENABLED

#include <Python.h>
#include <opencv2/gapi/streaming/gstreaming.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/core/cv_cpu_dispatch.h>
#include "misc/python/pyopencv_gapi.hpp" // For PyObjectHolder

namespace cv {
namespace gapi {
namespace wip {

/**
 * @brief Private implementation class for PythonStreamSource
 */
class PythonStreamSource::Impl
{
private:
    cv::detail::PyObjectHolder m_python_source;
    mutable cv::GMetaArg m_cached_meta;
    mutable bool m_meta_cached;
    
    // Python method names
    static constexpr const char* PULL_METHOD = "pull";
    static constexpr const char* DESCR_METHOD = "descr_of";
    static constexpr const char* HALT_METHOD = "halt";

public:
    explicit Impl(PyObject* python_source)
        : m_python_source(python_source, true)
        , m_meta_cached(false)
    {
        CV_Assert(python_source != nullptr);
        
        // Verify that the Python object has required methods
        PyObject* py_obj = m_python_source.get();
        
        if (!PyObject_HasAttrString(py_obj, PULL_METHOD)) {
            CV_Error(cv::Error::StsBadArg, "Python source object must have 'pull' method");
        }
        
        if (!PyObject_HasAttrString(py_obj, DESCR_METHOD)) {
            CV_Error(cv::Error::StsBadArg, "Python source object must have 'descr_of' method");
        }
        
        // halt method is optional - will be checked when called
    }
    
    bool pull(Data& data)
    {
        PyObject* py_obj = m_python_source.get();
        CV_Assert(py_obj != nullptr);
        
        // Call Python object's pull() method
        PyObject* py_result = PyObject_CallMethod(py_obj, PULL_METHOD, nullptr);
        
        if (py_result == nullptr) {
            PyErr_Print();
            CV_Error(cv::Error::StsError, "Failed to call pull() method on Python source");
        }
        
        // Expected return: (success: bool, data: Any)
        if (!PyTuple_Check(py_result) || PyTuple_Size(py_result) != 2) {
            Py_DECREF(py_result);
            CV_Error(cv::Error::StsBadArg, "Python source pull() must return (bool, data) tuple");
        }
        
        PyObject* py_success = PyTuple_GetItem(py_result, 0);
        PyObject* py_data = PyTuple_GetItem(py_result, 1);
        
        // Check success flag
        int success = PyObject_IsTrue(py_success);
        if (success == -1) {
            Py_DECREF(py_result);
            PyErr_Print();
            CV_Error(cv::Error::StsError, "Failed to evaluate success flag from Python source");
        }
        
        if (success == 0) {
            // Stream ended
            Py_DECREF(py_result);
            return false;
        }
        
        // Convert Python data to cv::gapi::wip::Data
        try {
            if (py_data == Py_None) {
                // No data available but success=True indicates continue
                Py_DECREF(py_result);
                return false;
            }
            
            // Try to convert py_data to cv::Mat first (most common case)
            cv::Mat mat;
            if (pyopencv_to(py_data, mat, cv::ArgInfo("data", false))) {
                data = Data{mat};
                Py_DECREF(py_result);
                return true;
            }
            
            // Try to convert to tuple of values (multi-input case)
            if (PyTuple_Check(py_data)) {
                cv::GRunArgs args;
                if (pyopencv_to(py_data, args, cv::ArgInfo("data", false))) {
                    data = Data{args};
                    Py_DECREF(py_result);
                    return true;
                }
            }
            
            // Try other common types
            cv::Scalar scalar;
            if (pyopencv_to(py_data, scalar, cv::ArgInfo("data", false))) {
                data = Data{scalar};
                Py_DECREF(py_result);
                return true;
            }
            
            // If we get here, unsupported data type
            Py_DECREF(py_result);
            CV_Error(cv::Error::StsError, "Unsupported data type returned from Python source");
            
        } catch (const cv::Exception& e) {
            Py_DECREF(py_result);
            throw;
        } catch (...) {
            Py_DECREF(py_result);
            CV_Error(cv::Error::StsError, "Unknown error converting Python data");
        }
        
        return false; // Should not reach here
    }
    
    cv::GMetaArg descr_of() const
    {
        if (m_meta_cached) {
            return m_cached_meta;
        }
        
        PyObject* py_obj = m_python_source.get();
        CV_Assert(py_obj != nullptr);
        
        // Call Python object's descr_of() method
        PyObject* py_result = PyObject_CallMethod(py_obj, DESCR_METHOD, nullptr);
        
        if (py_result == nullptr) {
            PyErr_Print();
            CV_Error(cv::Error::StsError, "Failed to call descr_of() method on Python source");
        }
        
        try {
            // Convert Python result to GMetaArg
            cv::GMetaArg meta;
            if (!pyopencv_to(py_result, meta, cv::ArgInfo("meta", false))) {
                Py_DECREF(py_result);
                CV_Error(cv::Error::StsError, "Failed to convert Python descr_of() result to GMetaArg");
            }
            
            Py_DECREF(py_result);
            
            // Cache the result
            m_cached_meta = meta;
            m_meta_cached = true;
            
            return meta;
            
        } catch (const cv::Exception& e) {
            Py_DECREF(py_result);
            throw;
        } catch (...) {
            Py_DECREF(py_result);
            CV_Error(cv::Error::StsError, "Unknown error converting Python metadata");
        }
    }
    
    void halt()
    {
        PyObject* py_obj = m_python_source.get();
        CV_Assert(py_obj != nullptr);
        
        // Check if halt method exists (it's optional)
        if (!PyObject_HasAttrString(py_obj, HALT_METHOD)) {
            return; // No halt method - that's OK
        }
        
        // Call Python object's halt() method
        PyObject* py_result = PyObject_CallMethod(py_obj, HALT_METHOD, nullptr);
        
        if (py_result == nullptr) {
            PyErr_Print();
            // Don't throw error for halt - just log warning
            CV_LOG_WARNING(nullptr, "Failed to call halt() method on Python source");
            return;
        }
        
        Py_DECREF(py_result);
    }
};

// PythonStreamSource implementation
PythonStreamSource::PythonStreamSource(PyObject* python_source)
    : m_impl(std::make_unique<Impl>(python_source))
{
}

bool PythonStreamSource::pull(Data& data)
{
    return m_impl->pull(data);
}

cv::GMetaArg PythonStreamSource::descr_of() const
{
    return m_impl->descr_of();
}

void PythonStreamSource::halt()
{
    m_impl->halt();
}

PythonStreamSource::~PythonStreamSource() = default;

// Factory function
IStreamSource::Ptr make_python_src(PyObject* python_source)
{
    auto src = std::make_shared<PythonStreamSource>(python_source);
    return src->ptr();
}

} // namespace wip
} // namespace gapi
} // namespace cv

#endif // CV_PYTHON_ENABLED

#endif // HAVE_OPENCV_GAPI
