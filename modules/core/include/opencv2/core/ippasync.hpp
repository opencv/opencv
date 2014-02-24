#ifndef __OPENCV_CORE_IPPASYNC_HPP__
#define __OPENCV_CORE_IPPASYNC_HPP__

#include "opencv2/core.hpp"
#include <ipp_async_op.h>
#include <ipp_async_accel.h>

namespace cv
{
    void DefaultDeleter<hppiMatrix>::operator () (hppiMatrix* p) const
    {
        hppiFreeMatrix(p);
    }

namespace hpp
{
    //convert OpenCV data type to hppDataType
    inline int toHppType(const int cvType)
    {
        int depth = CV_MAT_DEPTH(cvType);
        int hppType = depth == CV_8U ? HPP_DATA_TYPE_8U :
                     depth == CV_16U ? HPP_DATA_TYPE_16U :
                     depth == CV_16S ? HPP_DATA_TYPE_16S :
                     depth == CV_32S ? HPP_DATA_TYPE_32S :
                     depth == CV_32F ? HPP_DATA_TYPE_32F :
                     depth == CV_64F ? HPP_DATA_TYPE_64F : -1;
        CV_Assert( hppType >= 0 );
        return hppType; 
    }

    //convert hppDataType to OpenCV data type
    inline int toCvType(const int hppType)
    {
        int cvType = hppType == HPP_DATA_TYPE_8U ? CV_8U :
                    hppType == HPP_DATA_TYPE_16U ? CV_16U :
                    hppType == HPP_DATA_TYPE_16S ? CV_16S :
                    hppType == HPP_DATA_TYPE_32S ? CV_32S :
                    hppType == HPP_DATA_TYPE_32F ? CV_32F :
                    hppType == HPP_DATA_TYPE_64F ? CV_64F : -1;
        CV_Assert( cvType >= 0 );
        return cvType;
    }

    inline void copyHppToMat(hppiMatrix* src, Mat& dst, hppAccel accel, int cn)
    {
        hppDataType type;
        hpp32u width, height;
        hppStatus sts;

        CV_Assert(src!=NULL);

        sts = hppiInquireMatrix(src, &type, &width, &height);

        CV_Assert( sts == HPP_STATUS_NO_ERROR);

        int matType = CV_MAKETYPE(toCvType(type), cn);

        CV_Assert(width%cn == 0);

        width /= cn;

        dst.create((int)height, (int)width, (int)matType);

        size_t newSize = (size_t)(height*(hpp32u)(dst.step));

        sts = hppiGetMatrixData(accel,src,(hpp32u)(dst.step),dst.data,&newSize);

        CV_Assert( sts == HPP_STATUS_NO_ERROR);
    }

    //create cv::Mat from hppiMatrix
    inline Mat getMat(hppiMatrix* src, hppAccel accel, int cn)
    {
        Mat dst;
        copyHppToMat(src, dst, accel, cn);
        return dst;
    }

     //create hppiMatrix from cv::Mat
    inline Ptr<hppiMatrix> getHpp(const Mat& src)
    {
        int htype = toHppType(src.type());
        int cn = src.channels();

        CV_Assert(src.data);
        hppiMatrix *dst = hppiCreateMatrix(htype, src.cols*cn, src.rows, src.data, (hpp32s)(src.step));

        return Ptr<hppiMatrix>(dst);
    }
}}

#endif