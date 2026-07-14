// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _GRFMT_PFM_H_
#define _GRFMT_PFM_H_

#include "grfmt_base.hpp"
#include "bitstrm.hpp"

#ifdef HAVE_IMGCODEC_PFM
namespace cv
{

class PFMDecoder CV_FINAL : public BaseImageDecoder
{
public:
    PFMDecoder();
    virtual ~PFMDecoder() CV_OVERRIDE;

    bool  readData( Mat& img ) CV_OVERRIDE;
    bool  readHeader() CV_OVERRIDE;
    void  close();

    size_t signatureLength() const CV_OVERRIDE;
    bool checkSignature( const String& signature ) const CV_OVERRIDE;
    ImageDecoder newDecoder() const CV_OVERRIDE
    {
        return makePtr<PFMDecoder>();
    }

private:
    RLByteStream m_strm;
    double m_scale_factor;
    bool m_swap_byte_order;
};

class PFMEncoder CV_FINAL : public BaseImageEncoder
{
public:
    PFMEncoder();
    virtual ~PFMEncoder() CV_OVERRIDE;

    bool  isFormatSupported( int depth ) const CV_OVERRIDE;
    bool  write( const Mat& img, const std::vector<int>& params ) CV_OVERRIDE;

    ImageEncoder newEncoder() const CV_OVERRIDE
    {
        return makePtr<PFMEncoder>();
    }
};

}

#endif // HAVE_IMGCODEC_PXM

#endif/*_GRFMT_PFM_H_*/