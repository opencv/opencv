#ifndef CV2_CONVERT_HPP
#define CV2_CONVERT_HPP

#include "cv2.hpp"
#include "cv2_util.hpp"
#include "cv2_numpy.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <type_traits>  // std::enable_if

extern PyTypeObject* pyopencv_Mat_TypePtr;

#define CV_HAS_CONVERSION_ERROR(x) (((x) == -1) && PyErr_Occurred())

inline bool isBool(PyObject* obj) CV_NOEXCEPT
{
    return PyArray_IsScalar(obj, Bool) || PyBool_Check(obj);
}

//======================================================================================================================


// exception-safe pyopencv_to
template<typename _Tp> static
bool pyopencv_to_safe(PyObject* obj, _Tp& value, const ArgInfo& info)
{
    try
    {
        return pyopencv_to(obj, value, info);
    }
    catch (const std::exception &e)
    {
        PyErr_SetString(opencv_error, cv::format("Conversion error: %s, what: %s", info.name, e.what()).c_str());
        return false;
    }
    catch (...)
    {
        PyErr_SetString(opencv_error, cv::format("Conversion error: %s", info.name).c_str());
        return false;
    }
}

//======================================================================================================================

template<typename T, class TEnable = void>  // TEnable is used for SFINAE checks
struct PyOpenCV_Converter
{
    //static inline bool to(PyObject* obj, T& p, const ArgInfo& info);
    //static inline PyObject* from(const T& src);
};

// --- Generic

template<typename T>
bool pyopencv_to(PyObject* obj, T& p, const ArgInfo& info) { return PyOpenCV_Converter<T>::to(obj, p, info); }

template<typename T>
PyObject* pyopencv_from(const T& src) { return PyOpenCV_Converter<T>::from(src); }

// --- Matx

template<typename _Tp, int m, int n>
bool pyopencv_to(PyObject* o, cv::Matx<_Tp, m, n>& mx, const ArgInfo& info)
{
    if (!o || o == Py_None) {
        return true;
    }

    cv::Mat tmp;
    if (!pyopencv_to(o, tmp, info)) {
        return false;
    }

    tmp.copyTo(mx);
    return true;
}

template<typename _Tp, int m, int n>
PyObject* pyopencv_from(const cv::Matx<_Tp, m, n>& matx)
{
    return pyopencv_from(cv::Mat(matx));
}

// --- bool
template<> bool pyopencv_to(PyObject* obj, bool& value, const ArgInfo& info);
template<> PyObject* pyopencv_from(const bool& value);

// --- Mat
template<> bool pyopencv_to(PyObject* o, cv::Mat& m, const ArgInfo& info);
template<> PyObject* pyopencv_from(const cv::Mat& m);

// --- Ptr
template<typename T>
struct PyOpenCV_Converter< cv::Ptr<T> >
{
    static PyObject* from(const cv::Ptr<T>& p)
    {
        if (!p)
            Py_RETURN_NONE;
        return pyopencv_from(*p);
    }
    static bool to(PyObject *o, cv::Ptr<T>& p, const ArgInfo& info)
    {
        if (!o || o == Py_None)
            return true;
        p = cv::makePtr<T>();
        return pyopencv_to(o, *p, info);
    }
};

// --- ptr
template<> bool pyopencv_to(PyObject* obj, void*& ptr, const ArgInfo& info);
PyObject* pyopencv_from(void*& ptr);

// --- Scalar
template<> bool pyopencv_to(PyObject *o, cv::Scalar& s, const ArgInfo& info);
template<> PyObject* pyopencv_from(const cv::Scalar& src);

// --- size_t
template<> bool pyopencv_to(PyObject* obj, size_t& value, const ArgInfo& info);
template<> PyObject* pyopencv_from(const size_t& value);

// --- int
template<> bool pyopencv_to(PyObject* obj, int& value, const ArgInfo& info);
template<> PyObject* pyopencv_from(const int& value);

// --- int64
template<> bool pyopencv_to(PyObject* obj, int64& value, const ArgInfo& info);
template<> PyObject* pyopencv_from(const int64& value);

// There is conflict between "size_t" and "unsigned int".
// They are the same type on some 32-bit platforms.
template<typename T>
struct PyOpenCV_Converter
    < T, typename std::enable_if< std::is_same<unsigned int, T>::value && !std::is_same<unsigned int, size_t>::value >::type >
{
    static inline PyObject* from(const unsigned int& value)
    {
        return PyLong_FromUnsignedLong(value);
    }

    static inline bool to(PyObject* obj, unsigned int& value, const ArgInfo& info)
    {
        CV_UNUSED(info);
        if(!obj || obj == Py_None)
            return true;
        if(PyInt_Check(obj))
            value = (unsigned int)PyInt_AsLong(obj);
        else if(PyLong_Check(obj))
            value = (unsigned int)PyLong_AsLong(obj);
        else
            return false;
        return value != (unsigned int)-1 || !PyErr_Occurred();
    }
};

// --- uchar
template<> bool pyopencv_to(PyObject* obj, uchar& value, const ArgInfo& info);
template<> PyObject* pyopencv_from(const uchar& value);

// --- char
template<> bool pyopencv_to(PyObject* obj, char& value, const ArgInfo& info);

// --- double
template<> bool pyopencv_to(PyObject* obj, double& value, const ArgInfo& info);
template<> PyObject* pyopencv_from(const double& value);

// --- float
template<> bool pyopencv_to(PyObject* obj, float& value, const ArgInfo& info);
template<> PyObject* pyopencv_from(const float& value);

// --- string
template<> bool pyopencv_to(PyObject* obj, cv::String &value, const ArgInfo& info);
template<> PyObject* pyopencv_from(const cv::String& value);
#if CV_VERSION_MAJOR == 3
template<> PyObject* pyopencv_from(const std::string& value);
#endif

// --- Size
template<> bool pyopencv_to(PyObject* obj, cv::Size& sz, const ArgInfo& info);
template<> PyObject* pyopencv_from(const cv::Size& sz);
template<> bool pyopencv_to(PyObject* obj, cv::Size_<float>& sz, const ArgInfo& info);
template<> PyObject* pyopencv_from(const cv::Size_<float>& sz);

// --- Rect
template<> bool pyopencv_to(PyObject* obj, cv::Rect& r, const ArgInfo& info);
template<> PyObject* pyopencv_from(const cv::Rect& r);
template<> bool pyopencv_to(PyObject* obj, cv::Rect2d& r, const ArgInfo& info);
template<> PyObject* pyopencv_from(const cv::Rect2d& r);

// --- RotatedRect
template<> bool pyopencv_to(PyObject* obj, cv::RotatedRect& dst, const ArgInfo& info);
template<> PyObject* pyopencv_from(const cv::RotatedRect& src);

// --- Range
template<> bool pyopencv_to(PyObject* obj, cv::Range& r, const ArgInfo& info);
template<> PyObject* pyopencv_from(const cv::Range& r);

// --- Point
template<> bool pyopencv_to(PyObject* obj, cv::Point& p, const ArgInfo& info);
template<> PyObject* pyopencv_from(const cv::Point& p);
template<> bool pyopencv_to(PyObject* obj, cv::Point2f& p, const ArgInfo& info);
template<> PyObject* pyopencv_from(const cv::Point2f& p);
template<> bool pyopencv_to(PyObject* obj, cv::Point2d& p, const ArgInfo& info);
template<> PyObject* pyopencv_from(const cv::Point2d& p);
template<> bool pyopencv_to(PyObject* obj, cv::Point3f& p, const ArgInfo& info);
template<> PyObject* pyopencv_from(const cv::Point3f& p);
template<> bool pyopencv_to(PyObject* obj, cv::Point3d& p, const ArgInfo& info);
template<> PyObject* pyopencv_from(const cv::Point3d& p);

// --- Vec
template<typename _Tp, int cn>
bool pyopencv_to(PyObject* o, cv::Vec<_Tp, cn>& vec, const ArgInfo& info)
{
    return pyopencv_to(o, (cv::Matx<_Tp, cn, 1>&)vec, info);
}
bool pyopencv_to(PyObject* obj, cv::Vec4d& v, ArgInfo& info);
PyObject* pyopencv_from(const cv::Vec4d& v);
bool pyopencv_to(PyObject* obj, cv::Vec4f& v, ArgInfo& info);
PyObject* pyopencv_from(const cv::Vec4f& v);
bool pyopencv_to(PyObject* obj, cv::Vec4i& v, ArgInfo& info);
PyObject* pyopencv_from(const cv::Vec4i& v);
bool pyopencv_to(PyObject* obj, cv::Vec3d& v, ArgInfo& info);
PyObject* pyopencv_from(const cv::Vec3d& v);
bool pyopencv_to(PyObject* obj, cv::Vec3f& v, ArgInfo& info);
PyObject* pyopencv_from(const cv::Vec3f& v);
bool pyopencv_to(PyObject* obj, cv::Vec3i& v, ArgInfo& info);
PyObject* pyopencv_from(const cv::Vec3i& v);
bool pyopencv_to(PyObject* obj, cv::Vec2d& v, ArgInfo& info);
PyObject* pyopencv_from(const cv::Vec2d& v);
bool pyopencv_to(PyObject* obj, cv::Vec2f& v, ArgInfo& info);
PyObject* pyopencv_from(const cv::Vec2f& v);
bool pyopencv_to(PyObject* obj, cv::Vec2i& v, ArgInfo& info);
PyObject* pyopencv_from(const cv::Vec2i& v);

// --- TermCriteria
template<> bool pyopencv_to(PyObject* obj, cv::TermCriteria& dst, const ArgInfo& info);
template<> PyObject* pyopencv_from(const cv::TermCriteria& src);

// --- Moments
template<> PyObject* pyopencv_from(const cv::Moments& m);

// --- pair
template<> PyObject* pyopencv_from(const std::pair<int, double>& src);

// --- vector
template <typename Tp>
struct pyopencvVecConverter;

template <typename Tp>
bool pyopencv_to(PyObject* obj, std::vector<Tp>& value, const ArgInfo& info)
{
    if (!obj || obj == Py_None)
    {
        return true;
    }
    return pyopencvVecConverter<Tp>::to(obj, value, info);
}

template <typename Tp>
PyObject* pyopencv_from(const std::vector<Tp>& value)
{
    return pyopencvVecConverter<Tp>::from(value);
}

template<typename K, typename V>
bool pyopencv_to(PyObject *obj, std::map<K,V> &map, const ArgInfo& info)
{
    if (!obj || obj == Py_None)
    {
        return true;
    }

    PyObject* py_key = nullptr;
    PyObject* py_value = nullptr;
    Py_ssize_t pos = 0;

    if (!PyDict_Check(obj)) {
        failmsg("Can't parse '%s'. Input argument isn't dict or"
                " an instance of subtype of the dict type", info.name);
        return false;
    }

    while(PyDict_Next(obj, &pos, &py_key, &py_value))
    {
        K cpp_key;
        if (!pyopencv_to(py_key, cpp_key, ArgInfo("key", false))) {
            failmsg("Can't parse dict key. Key on position %lu has a wrong type", pos);
            return false;
        }

        V cpp_value;
        if (!pyopencv_to(py_value, cpp_value, ArgInfo("value", false))) {
            failmsg("Can't parse dict value. Value on position %lu has a wrong type", pos);
            return false;
        }

        map.emplace(cpp_key, cpp_value);
    }
    return true;
}

template <typename Tp>
static bool pyopencv_to_generic_vec(PyObject* obj, std::vector<Tp>& value, const ArgInfo& info)
{
    if (!obj || obj == Py_None)
    {
        return true;
    }
    if (!PySequence_Check(obj))
    {
        failmsg("Can't parse '%s'. Input argument doesn't provide sequence protocol", info.name);
        return false;
    }
    const size_t n = static_cast<size_t>(PySequence_Size(obj));
    value.resize(n);
    for (size_t i = 0; i < n; i++)
    {
        SafeSeqItem item_wrap(obj, i);
        if (!pyopencv_to(item_wrap.item, value[i], info))
        {
            failmsg("Can't parse '%s'. Sequence item with index %lu has a wrong type", info.name, i);
            return false;
        }
    }
    return true;
}

template<> inline bool pyopencv_to_generic_vec(PyObject* obj, std::vector<bool>& value, const ArgInfo& info)
{
    if (!obj || obj == Py_None)
    {
        return true;
    }
    if (!PySequence_Check(obj))
    {
        failmsg("Can't parse '%s'. Input argument doesn't provide sequence protocol", info.name);
        return false;
    }
    const size_t n = static_cast<size_t>(PySequence_Size(obj));
    value.resize(n);
    for (size_t i = 0; i < n; i++)
    {
        SafeSeqItem item_wrap(obj, i);
        bool elem{};
        if (!pyopencv_to(item_wrap.item, elem, info))
        {
            failmsg("Can't parse '%s'. Sequence item with index %lu has a wrong type", info.name, i);
            return false;
        }
        value[i] = elem;
    }
    return true;
}

template <typename Tp>
static PyObject* pyopencv_from_generic_vec(const std::vector<Tp>& value)
{
    Py_ssize_t n = static_cast<Py_ssize_t>(value.size());
    PySafeObject seq(PyTuple_New(n));
    for (Py_ssize_t i = 0; i < n; i++)
    {
        PyObject* item = pyopencv_from(value[i]);
        // If item can't be assigned - PyTuple_SetItem raises exception and returns -1.
        if (!item || PyTuple_SetItem(seq, i, item) == -1)
        {
            return NULL;
        }
    }
    return seq.release();
}

template<> inline PyObject* pyopencv_from_generic_vec(const std::vector<bool>& value)
{
    Py_ssize_t n = static_cast<Py_ssize_t>(value.size());
    PySafeObject seq(PyTuple_New(n));
    for (Py_ssize_t i = 0; i < n; i++)
    {
        bool elem = value[i];
        PyObject* item = pyopencv_from(elem);
        // If item can't be assigned - PyTuple_SetItem raises exception and returns -1.
        if (!item || PyTuple_SetItem(seq, i, item) == -1)
        {
            return NULL;
        }
    }
    return seq.release();
}

namespace traits {

template <bool Value>
struct BooleanConstant
{
    static const bool value = Value;
    typedef BooleanConstant<Value> type;
};

typedef BooleanConstant<true> TrueType;
typedef BooleanConstant<false> FalseType;

template <class T>
struct VoidType {
    typedef void type;
};

template <class T, class DType = void>
struct IsRepresentableAsMatDataType : FalseType
{
};

template <class T>
struct IsRepresentableAsMatDataType<T, typename VoidType<typename cv::DataType<T>::channel_type>::type> : TrueType
{
};

// https://github.com/opencv/opencv/issues/20930
template <> struct IsRepresentableAsMatDataType<cv::RotatedRect, void> : FalseType {};

} // namespace traits

template <typename Tp>
struct pyopencvVecConverter
{
    typedef typename std::vector<Tp>::iterator VecIt;

    static bool to(PyObject* obj, std::vector<Tp>& value, const ArgInfo& info)
    {
        if (!PyArray_Check(obj))
        {
            return pyopencv_to_generic_vec(obj, value, info);
        }
        // If user passed an array it is possible to make faster conversions in several cases
        PyArrayObject* array_obj = reinterpret_cast<PyArrayObject*>(obj);
        const NPY_TYPES target_type = asNumpyType<Tp>();
        const NPY_TYPES source_type = static_cast<NPY_TYPES>(PyArray_TYPE(array_obj));
        if (target_type == NPY_OBJECT)
        {
            // Non-planar arrays representing objects (e.g. array of N Rect is an array of shape Nx4) have NPY_OBJECT
            // as their target type.
            return pyopencv_to_generic_vec(obj, value, info);
        }
        if (PyArray_NDIM(array_obj) > 1)
        {
            failmsg("Can't parse %dD array as '%s' vector argument", PyArray_NDIM(array_obj), info.name);
            return false;
        }
        if (target_type != source_type)
        {
            // Source type requires conversion
            // Allowed conversions for target type is handled in the corresponding pyopencv_to function
            return pyopencv_to_generic_vec(obj, value, info);
        }
        // For all other cases, all array data can be directly copied to std::vector data
        // Simple `memcpy` is not possible because NumPy array can reference a slice of the bigger array:
        // ```
        // arr = np.ones((8, 4, 5), dtype=np.int32)
        // convertible_to_vector_of_int = arr[:, 0, 1]
        // ```
        value.resize(static_cast<size_t>(PyArray_SIZE(array_obj)));
        const npy_intp item_step = PyArray_STRIDE(array_obj, 0) / PyArray_ITEMSIZE(array_obj);
        const Tp* data_ptr = static_cast<Tp*>(PyArray_DATA(array_obj));
        for (VecIt it = value.begin(); it != value.end(); ++it, data_ptr += item_step) {
            *it = *data_ptr;
        }
        return true;
    }

    static PyObject* from(const std::vector<Tp>& value)
    {
        if (value.empty())
        {
            return PyTuple_New(0);
        }
        return from(value, ::traits::IsRepresentableAsMatDataType<Tp>());
    }

private:
    static PyObject* from(const std::vector<Tp>& value, ::traits::FalseType)
    {
        // Underlying type is not representable as Mat Data Type
        return pyopencv_from_generic_vec(value);
    }

    static PyObject* from(const std::vector<Tp>& value, ::traits::TrueType)
    {
        // Underlying type is representable as Mat Data Type, so faster return type is available
        typedef cv::DataType<Tp> DType;
        typedef typename DType::channel_type UnderlyingArrayType;

        // If Mat is always exposed as NumPy array this code path can be reduced to the following snipped:
        //        Mat src(value);
        //        PyObject* array = pyopencv_from(src);
        //        return PyArray_Squeeze(reinterpret_cast<PyArrayObject*>(array));
        // This puts unnecessary restrictions on Mat object those might be avoided without losing the performance.
        // Moreover, this version is a bit faster, because it doesn't create temporary objects with reference counting.

        const NPY_TYPES target_type = asNumpyType<UnderlyingArrayType>();
        const int cols = DType::channels;
        PyObject* array = NULL;
        if (cols == 1)
        {
            npy_intp dims = static_cast<npy_intp>(value.size());
            array = PyArray_SimpleNew(1, &dims, target_type);
        }
        else
        {
            npy_intp dims[2] = {static_cast<npy_intp>(value.size()), cols};
            array = PyArray_SimpleNew(2, dims, target_type);
        }
        if(!array)
        {
            // NumPy arrays with shape (N, 1) and (N) are not equal, so correct error message should distinguish
            // them too.
            cv::String shape;
            if (cols > 1)
            {
                shape = cv::format("(%d x %d)", static_cast<int>(value.size()), cols);
            }
            else
            {
                shape = cv::format("(%d)", static_cast<int>(value.size()));
            }
            const cv::String error_message = cv::format("Can't allocate NumPy array for vector with dtype=%d and shape=%s",
                                                static_cast<int>(target_type), shape.c_str());
            emit_failmsg(PyExc_MemoryError, error_message.c_str());
            return array;
        }
        // Fill the array
        PyArrayObject* array_obj = reinterpret_cast<PyArrayObject*>(array);
        UnderlyingArrayType* array_data = static_cast<UnderlyingArrayType*>(PyArray_DATA(array_obj));
        // if Tp is representable as Mat DataType, so the following cast is pretty safe...
        const UnderlyingArrayType* value_data = reinterpret_cast<const UnderlyingArrayType*>(value.data());
        memcpy(array_data, value_data, sizeof(UnderlyingArrayType) * value.size() * static_cast<size_t>(cols));
        return array;
    }
};

// --- tuple
template<std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
convert_to_python_tuple(const std::tuple<Tp...>&, PyObject*) {  }

template<std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I < sizeof...(Tp), void>::type
convert_to_python_tuple(const std::tuple<Tp...>& cpp_tuple, PyObject* py_tuple)
{
    PyObject* item = pyopencv_from(std::get<I>(cpp_tuple));

    if (!item)
        return;

    PyTuple_SetItem(py_tuple, I, item);
    convert_to_python_tuple<I + 1, Tp...>(cpp_tuple, py_tuple);
}

template<typename... Ts>
PyObject* pyopencv_from(const std::tuple<Ts...>& cpp_tuple)
{
    size_t size = sizeof...(Ts);
    PyObject* py_tuple = PyTuple_New(size);
    convert_to_python_tuple(cpp_tuple, py_tuple);
    size_t actual_size = PyTuple_Size(py_tuple);

    if (actual_size < size)
    {
        Py_DECREF(py_tuple);
        return NULL;
    }

    return py_tuple;
}

#endif // CV2_CONVERT_HPP
