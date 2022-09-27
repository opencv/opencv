#ifndef CV2_UTIL_HPP
#define CV2_UTIL_HPP

#include "cv2.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/utils/tls.hpp"
#include <vector>
#include <string>

//======================================================================================================================

bool isPythonBindingsDebugEnabled();
void emit_failmsg(PyObject * exc, const char *msg);
int failmsg(const char *fmt, ...);
PyObject* failmsgp(const char *fmt, ...);

//======================================================================================================================

class PyAllowThreads
{
public:
    PyAllowThreads() : _state(PyEval_SaveThread()) {}
    ~PyAllowThreads()
    {
        PyEval_RestoreThread(_state);
    }
private:
    PyThreadState* _state;
};

class PyEnsureGIL
{
public:
    PyEnsureGIL() : _state(PyGILState_Ensure()) {}
    ~PyEnsureGIL()
    {
        PyGILState_Release(_state);
    }
private:
    PyGILState_STATE _state;
};

/**
 * Light weight RAII wrapper for `PyObject*` owning references.
 * In comparisson to C++11 `std::unique_ptr` with custom deleter, it provides
 * implicit conversion functions that might be useful to initialize it with
 * Python functions those returns owning references through the `PyObject**`
 * e.g. `PyErr_Fetch` or directly pass it to functions those want to borrow
 * reference to object (doesn't extend object lifetime) e.g. `PyObject_Str`.
 */
class PySafeObject
{
public:
    PySafeObject() : obj_(NULL) {}

    explicit PySafeObject(PyObject* obj) : obj_(obj) {}

    ~PySafeObject()
    {
        Py_CLEAR(obj_);
    }

    operator PyObject*()
    {
        return obj_;
    }

    operator PyObject**()
    {
        return &obj_;
    }

    PyObject* release()
    {
        PyObject* obj = obj_;
        obj_ = NULL;
        return obj;
    }

private:
    PyObject* obj_;

    // Explicitly disable copy operations
    PySafeObject(const PySafeObject*); // = delete
    PySafeObject& operator=(const PySafeObject&); // = delete
};

//======================================================================================================================

extern PyObject* opencv_error;

void pyRaiseCVException(const cv::Exception &e);

#define ERRWRAP2(expr) \
try \
{ \
    PyAllowThreads allowThreads; \
    expr; \
} \
catch (const cv::Exception &e) \
{ \
    pyRaiseCVException(e); \
    return 0; \
} \
catch (const std::exception &e) \
{ \
    PyErr_SetString(opencv_error, e.what()); \
    return 0; \
} \
catch (...) \
{ \
    PyErr_SetString(opencv_error, "Unknown C++ exception from OpenCV code"); \
    return 0; \
}

//======================================================================================================================

extern cv::TLSData<std::vector<std::string> > conversionErrorsTLS;

inline void pyPrepareArgumentConversionErrorsStorage(std::size_t size)
{
    std::vector<std::string>& conversionErrors = conversionErrorsTLS.getRef();
    conversionErrors.clear();
    conversionErrors.reserve(size);
}

void pyRaiseCVOverloadException(const std::string& functionName);
void pyPopulateArgumentConversionErrors();

//======================================================================================================================

PyObject *pycvRedirectError(PyObject*, PyObject *args, PyObject *kw);

#endif // CV2_UTIL_HPP
