/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

#ifdef HAVE_RAW
#define LIBRAW_NO_WINSOCK2
#define LIBRAW_NODLL
#include <../libraw/libraw.h>



#include "grfmt_raw.hpp"


namespace cv
{


    LibRawDecoder::LibRawDecoder()
    {
        rawProcessor = new LibRaw;
    }

    LibRawDecoder::~LibRawDecoder() {}

    size_t LibRawDecoder::signatureLength() const
    {
        return 2048;
    }

    bool LibRawDecoder::checkSignature(const String& signature) const
    {

        return true;
    }

    ImageDecoder LibRawDecoder::newDecoder() const
    {
        return makePtr<LibRawDecoder>();
    }

    bool LibRawDecoder::readHeader()
    {

        int status = ((LibRaw*)rawProcessor)->open_file(this->m_filename.c_str());
        if (status == LIBRAW_SUCCESS)
        {
            status = ((LibRaw*)rawProcessor)->unpack();
            if (status != LIBRAW_SUCCESS)
                CV_Error(Error::StsNotImplemented, libraw_strerror(status));
            int widthp, heightp, colorsp, bpp;
            ((LibRaw*)rawProcessor)->get_mem_image_format(&widthp, &heightp, &colorsp, &bpp);
            this->m_width = widthp;
            this->m_height = heightp;
            if (bpp <= 8)
                this->m_type = CV_8UC(colorsp);
            else if (bpp <= 16)
                this->m_type = CV_16UC(colorsp);
            else
                CV_Error(Error::StsNotImplemented, "only depth CV_8U and CV16_U are supported");
        }
        else
        {
            CV_Error(Error::StsNotImplemented, libraw_strerror(status));
        }
        return status == LIBRAW_SUCCESS; ;
    }

    bool LibRawDecoder::readData(Mat& img)
    {
        int status = ((LibRaw*)rawProcessor)->dcraw_process();
        if (status != LIBRAW_SUCCESS)
            CV_Error(Error::StsNotImplemented, libraw_strerror(status));
        status = ((LibRaw*)rawProcessor)->copy_mem_image(img.data, img.step, 1);
        if (status != LIBRAW_SUCCESS)
            CV_Error(Error::StsNotImplemented, libraw_strerror(status));
        return status == LIBRAW_SUCCESS;
    }


}

#endif
