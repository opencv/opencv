// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020, Stefan Brüns <stefan.bruens@rwth-aachen.de>

#ifndef _GRFMT_OPENJPEG_H_
#define _GRFMT_OPENJPEG_H_

#ifdef HAVE_OPENJPEG

#include "grfmt_base.hpp"
#include <openjpeg.h>

namespace cv
{

class Jpeg2KOpjDecoder CV_FINAL : public BaseImageDecoder
{
public:

    Jpeg2KOpjDecoder();
    virtual ~Jpeg2KOpjDecoder();

    bool readData( Mat& img ) CV_OVERRIDE;
    bool readHeader() CV_OVERRIDE;
    bool checkSignature( const String& signature ) const CV_OVERRIDE;
    ImageDecoder newDecoder() const CV_OVERRIDE;
    size_t signatureLength() const CV_OVERRIDE;

private:
    void setMessageHandlers();

    opj_stream_t* m_stream = nullptr;
    opj_codec_t* m_codec = nullptr;
    opj_image_t* m_image = nullptr;

    String m_errorMessage;
    OPJ_UINT32 m_maxPrec = 0;
};

class Jpeg2KOpjEncoder CV_FINAL : public BaseImageEncoder
{
public:
    Jpeg2KOpjEncoder();
    virtual ~Jpeg2KOpjEncoder();

    bool isFormatSupported( int depth ) const CV_OVERRIDE;
    bool write( const Mat& img, const std::vector<int>& params ) CV_OVERRIDE;
    ImageEncoder newEncoder() const CV_OVERRIDE;

private:
    String m_errorMessage;
};

}

#endif

#endif/*_GRFMT_OPENJPEG_H_*/
