#ifdef HAVE_OPENCV_OBJDETECT

#include "opencv2/objdetect.hpp"

typedef QRCodeEncoder::Params QRCodeEncoder_Params;

typedef HOGDescriptor::HistogramNormType HOGDescriptor_HistogramNormType;
typedef HOGDescriptor::DescriptorStorageFormat HOGDescriptor_DescriptorStorageFormat;

class NativeByteArray
{
public:
    inline NativeByteArray& operator=(const std::string& from) {
        val = from;
        return *this;
    }
    std::string val;
};

class vector_NativeByteArray : public std::vector<std::string> {};

template<>
PyObject* pyopencv_from(const NativeByteArray& from)
{
    return PyBytes_FromStringAndSize(from.val.c_str(), from.val.size());
}

template<>
PyObject* pyopencv_from(const vector_NativeByteArray& results)
{
    PyObject* list = PyList_New(results.size());
    for(size_t i = 0; i < results.size(); ++i)
        PyList_SetItem(list, i, PyBytes_FromStringAndSize(results[i].c_str(), results[i].size()));
    return list;
}

#endif
