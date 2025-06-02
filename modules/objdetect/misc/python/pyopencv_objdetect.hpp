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

template<>
PyObject* pyopencv_from(const NativeByteArray& from)
{
    return PyBytes_FromStringAndSize(from.val.c_str(), from.val.size());
}

#endif
