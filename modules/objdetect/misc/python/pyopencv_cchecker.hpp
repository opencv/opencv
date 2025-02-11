#include "opencv2/mcc.hpp"

template <>
struct pyopencvVecConverter<Ptr<mcc::CChecker>>
{
    static bool to(PyObject *obj, std::vector<Ptr<mcc::CChecker>> &value,
                   const ArgInfo &info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject *from(const std::vector<Ptr<mcc::CChecker>> &value)
    {
        return pyopencv_from_generic_vec(value);
    }
};
typedef std::vector<cv::Ptr<mcc::CChecker>> vector_Ptr_CChecker;
typedef dnn::Net dnn_Net;
