#include "cv2_util.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/utils/configuration.private.hpp"
#include "opencv2/core/utils/logger.hpp"

PyObject* opencv_error = NULL;
cv::TLSData<std::vector<std::string> > conversionErrorsTLS;

int cvDepthToNumpyType(int depth)
{
    switch (depth)
    {
    case CV_8U:   return NPY_UBYTE;
    case CV_8S:   return NPY_BYTE;
    case CV_16U:  return NPY_USHORT;
    case CV_16S:  return NPY_SHORT;
    case CV_32U:  return NPY_UINT32;
    case CV_32S:  return NPY_INT32;
    case CV_64U:  return NPY_UINT64;
    case CV_64S:  return NPY_INT64;
    case CV_32F:  return NPY_FLOAT;
    case CV_64F:  return NPY_DOUBLE;
    case CV_16F:  return NPY_HALF;
    // NumPy has no bfloat16 dtype; pyopencv_from() converts the values to float32.
    case CV_16BF: return NPY_FLOAT;
    case CV_Bool: return NPY_BOOL;
    default:
        CV_Error(cv::Error::StsNotImplemented,
                 cv::format("Mat depth %d has no corresponding NumPy dtype", depth));
    }
}

int numpyTypeToCvDepth(int typenum)
{
    // Only canonical NPY_* values may be used as case labels: the fixed-width
    // aliases (NPY_INT32, NPY_INT64, ...) expand to these and would collide.
    switch (typenum)
    {
    case NPY_UBYTE:     return CV_8U;
    case NPY_BYTE:      return CV_8S;
    case NPY_USHORT:    return CV_16U;
    case NPY_SHORT:     return CV_16S;
    case NPY_UINT:      return CV_32U;
    case NPY_INT:       return CV_32S;
    case NPY_ULONGLONG: return CV_64U;
    // 'long' is 64-bit on LP64 but 32-bit on LLP64, so decide by size, not by name.
    case NPY_ULONG:     return NPY_SIZEOF_LONG == 8 ? CV_64U : CV_32U;
    // Map the 32-bit case: it is numpy's int32 on LLP64, and output arrays cannot be cast.
    case NPY_LONG:      return NPY_SIZEOF_LONG == 4 ? CV_32S : -1;
    case NPY_HALF:      return CV_16F;
    case NPY_FLOAT:     return CV_32F;
    case NPY_DOUBLE:    return CV_64F;
    case NPY_BOOL:      return CV_Bool;
    default:            return -1;
    }
}

using namespace cv;

//======================================================================================================================

bool isPythonBindingsDebugEnabled()
{
    static bool param_debug = cv::utils::getConfigurationParameterBool("OPENCV_PYTHON_DEBUG", false);
    return param_debug;
}

void emit_failmsg(PyObject * exc, const char *msg)
{
    static bool param_debug = isPythonBindingsDebugEnabled();
    if (param_debug)
    {
        CV_LOG_WARNING(NULL, "Bindings conversion failed: " << msg);
    }
    PyErr_SetString(exc, msg);
}

int failmsg(const char *fmt, ...)
{
    char str[1000];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    emit_failmsg(PyExc_TypeError, str);
    return 0;
}

PyObject* failmsgp(const char *fmt, ...)
{
    char str[1000];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    emit_failmsg(PyExc_TypeError, str);
    return 0;
}

void pyRaiseCVException(const cv::Exception &e)
{
    PyObject* temp_obj = PyString_FromString(e.file.c_str());
    PyObject_SetAttrString(opencv_error, "file", temp_obj);
    Py_DECREF(temp_obj);
    temp_obj = PyString_FromString(e.func.c_str());
    PyObject_SetAttrString(opencv_error, "func", temp_obj);
    Py_DECREF(temp_obj);
    temp_obj = PyInt_FromLong(e.line);
    PyObject_SetAttrString(opencv_error, "line", temp_obj);
    Py_DECREF(temp_obj);
    temp_obj = PyInt_FromLong(e.code);
    PyObject_SetAttrString(opencv_error, "code", temp_obj);
    Py_DECREF(temp_obj);
    temp_obj = PyString_FromString(e.msg.c_str());
    PyObject_SetAttrString(opencv_error, "msg", temp_obj);
    Py_DECREF(temp_obj);
    temp_obj = PyString_FromString(e.err.c_str());
    PyObject_SetAttrString(opencv_error, "err", temp_obj);
    Py_DECREF(temp_obj);
    PyErr_SetString(opencv_error, e.what());
}

//======================================================================================================================

void pyRaiseCVOverloadException(const std::string& functionName)
{
    const std::vector<std::string>& conversionErrors = conversionErrorsTLS.getRef();
    const std::size_t conversionErrorsCount = conversionErrors.size();
    if (conversionErrorsCount > 0)
    {
        // In modern std libraries small string optimization is used = no dynamic memory allocations,
        // but it can be applied only for string with length < 18 symbols (in GCC)
        const std::string bullet = "\n - ";

        // Estimate required buffer size - save dynamic memory allocations = faster
        std::size_t requiredBufferSize = bullet.size() * conversionErrorsCount;
        for (std::size_t i = 0; i < conversionErrorsCount; ++i)
        {
            requiredBufferSize += conversionErrors[i].size();
        }

        // Only string concatenation is required so std::string is way faster than
        // std::ostringstream
        std::string errorMessage("Overload resolution failed:");
        errorMessage.reserve(errorMessage.size() + requiredBufferSize);
        for (std::size_t i = 0; i < conversionErrorsCount; ++i)
        {
            errorMessage += bullet;
            errorMessage += conversionErrors[i];
        }
        cv::Exception exception(Error::StsBadArg, errorMessage, functionName, "", -1);
        pyRaiseCVException(exception);
    }
    else
    {
        cv::Exception exception(Error::StsInternal, "Overload resolution failed, but no errors reported",
                                functionName, "", -1);
        pyRaiseCVException(exception);
    }
}

void pyPopulateArgumentConversionErrors()
{
    if (PyErr_Occurred())
    {
        PySafeObject exception_type;
        PySafeObject exception_value;
        PySafeObject exception_traceback;
        PyErr_Fetch(exception_type, exception_value, exception_traceback);
        PyErr_NormalizeException(exception_type, exception_value,
                                 exception_traceback);

        PySafeObject exception_message(PyObject_Str(exception_value));
        std::string message;
        getUnicodeString(exception_message, message);
        conversionErrorsTLS.getRef().push_back(std::move(message));
    }
}

//======================================================================================================================

static int OnError(int status, const char *func_name, const char *err_msg, const char *file_name, int line, void *userdata)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject *on_error = (PyObject*)userdata;
    PyObject *args = Py_BuildValue("isssi", status, func_name, err_msg, file_name, line);

    PyObject *r = PyObject_Call(on_error, args, NULL);
    if (r == NULL) {
        PyErr_Print();
    } else {
        Py_DECREF(r);
    }

    Py_DECREF(args);
    PyGILState_Release(gstate);

    return 0; // The return value isn't used
}

PyObject *pycvRedirectError(PyObject*, PyObject *args, PyObject *kw)
{
    const char *keywords[] = { "on_error", NULL };
    PyObject *on_error;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "O", (char**)keywords, &on_error))
        return NULL;

    if ((on_error != Py_None) && !PyCallable_Check(on_error))  {
        PyErr_SetString(PyExc_TypeError, "on_error must be callable");
        return NULL;
    }

    // Keep track of the previous handler parameter, so we can decref it when no longer used
    static PyObject* last_on_error = NULL;
    if (last_on_error) {
        Py_DECREF(last_on_error);
        last_on_error = NULL;
    }

    if (on_error == Py_None) {
        ERRWRAP2(redirectError(NULL));
    } else {
        last_on_error = on_error;
        Py_INCREF(last_on_error);
        ERRWRAP2(redirectError(OnError, last_on_error));
    }
    Py_RETURN_NONE;
}
