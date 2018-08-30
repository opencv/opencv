#ifdef HAVE_OPENCV_CUDACODEC

#include "opencv2/cudacodec.hpp"

typedef cudacodec::EncoderCallBack::PicType EncoderCallBack_PicType;

CV_PY_TO_CLASS(cudacodec::EncoderParams);

CV_PY_TO_ENUM(cudacodec::EncoderCallBack::PicType);
CV_PY_TO_ENUM(cudacodec::SurfaceFormat);

CV_PY_FROM_CLASS(cudacodec::EncoderParams);

#endif
