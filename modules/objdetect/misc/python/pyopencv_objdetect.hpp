#ifdef HAVE_OPENCV_OBJDETECT

#include "opencv2/objdetect.hpp"

typedef QRCodeEncoder::Params QRCodeEncoder_Params;

typedef HOGDescriptor::HistogramNormType HOGDescriptor_HistogramNormType;
typedef HOGDescriptor::DescriptorStorageFormat HOGDescriptor_DescriptorStorageFormat;

// template<> bool pyopencv_to(PyObject* obj, cv::aruco::Board& board, const ArgInfo& info)
// {
//     Ptr<aruco::Board> * obj_getp = nullptr;
//     if (!pyopencv_aruco_Board_getp(obj, obj_getp))
//     {
//         return (failmsgp("Incorrect type of self (must be 'aruco::Board' or its derivative)") != nullptr);
//     }
//
//     board = **obj_getp;
//     return true;
// }
//
// template<> bool pyopencv_to(PyObject* obj, cv::aruco::CharucoBoard& board, const ArgInfo& info)
// {
//     Ptr<aruco::CharucoBoard> * obj_getp = nullptr;
//     if (!pyopencv_aruco_CharucoBoard_getp(obj, obj_getp))
//     {
//         return (failmsgp("Incorrect type of self (must be 'aruco::CharucoBoard' or its derivative)") != nullptr);
//     }
//
//     board = **obj_getp;
//     return true;
// }
//
// template<> bool pyopencv_to(PyObject* obj, cv::aruco::GridBoard& board, const ArgInfo& info)
// {
//     Ptr<aruco::GridBoard> * obj_getp = nullptr;
//     if (!pyopencv_aruco_GridBoard_getp(obj, obj_getp))
//     {
//         return (failmsgp("Incorrect type of self (must be 'aruco::GridBoard' or its derivative)") != nullptr);
//     }
//
//     board = **obj_getp;
//     return true;
// }

#endif
