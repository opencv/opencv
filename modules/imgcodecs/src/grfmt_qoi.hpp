// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _GRFMT_QOI_H_
#define _GRFMT_QOI_H_

#include "grfmt_base.hpp"

#ifdef HAVE_QOI

namespace cv
{

class QoiDecoder CV_FINAL : public BaseImageDecoder
{
public:
    QoiDecoder();
    virtual ~QoiDecoder() CV_OVERRIDE;

    bool  readData( Mat& img ) CV_OVERRIDE;
    bool  readHeader() CV_OVERRIDE;
    void  close();

    ImageDecoder newDecoder() const CV_OVERRIDE
    {
        return makePtr<QoiDecoder>();
    }
};

class QoiEncoder CV_FINAL : public BaseImageEncoder
{
public:
    QoiEncoder();
    virtual ~QoiEncoder() CV_OVERRIDE;

    bool  write( const Mat& img, const std::vector<int>& params ) CV_OVERRIDE;

    ImageEncoder newEncoder() const CV_OVERRIDE
    {
        return makePtr<QoiEncoder>();
    }
};

}

#endif // HAVE_QOI

#endif /*_GRFMT_QOI_H_*/
