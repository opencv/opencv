#ifdef HAVE_OPENCV_OBJDETECT

#include "opencv2/objdetect.hpp"

typedef QRCodeEncoder::Params QRCodeEncoder_Params;
typedef std::vector<cv::Ptr<mcc::CChecker>> vector_Ptr_CChecker;
#ifdef HAVE_OPENCV_DNN
typedef dnn::Net dnn_Net;
#endif

#endif
