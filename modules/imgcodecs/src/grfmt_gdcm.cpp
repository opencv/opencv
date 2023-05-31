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
#include "grfmt_gdcm.hpp"

#ifdef HAVE_GDCM

//#define DBG(...) printf(__VA_ARGS__)
#define DBG(...)

#include <gdcmImageReader.h>

static const size_t preamble_skip = 128;
static const size_t magic_len = 4;

inline cv::String getMagic()
{
    return cv::String("\x44\x49\x43\x4D", 4);
}

namespace cv
{

/************************ DICOM decoder *****************************/

DICOMDecoder::DICOMDecoder()
{
    // DICOM preamble is 128 bytes (can have any value, defaults to 0) + 4 bytes magic number (DICM)
    m_signature = String(preamble_skip, (char)'\x0') + getMagic();
    m_buf_supported = false;
}

bool DICOMDecoder::checkSignature( const String& signature ) const
{
    if (signature.size() >= preamble_skip + magic_len)
    {
        if (signature.substr(preamble_skip, magic_len) == getMagic())
        {
            return true;
        }
    }
    DBG("GDCM | Signature does not match\n");
    return false;
}

ImageDecoder DICOMDecoder::newDecoder() const
{
    return makePtr<DICOMDecoder>();
}

bool  DICOMDecoder::readHeader()
{
    gdcm::ImageReader csImageReader;
    csImageReader.SetFileName(m_filename.c_str());
    if(!csImageReader.Read())
    {
        DBG("GDCM | Failed to open DICOM file\n");
        return(false);
    }

    const gdcm::Image &csImage = csImageReader.GetImage();
    bool bOK = true;
    switch (csImage.GetPhotometricInterpretation().GetType())
    {
        case gdcm::PhotometricInterpretation::MONOCHROME1:
        case gdcm::PhotometricInterpretation::MONOCHROME2:
        {
            switch (csImage.GetPixelFormat().GetScalarType())
            {
                case gdcm::PixelFormat::INT8: m_type = CV_8SC1; break;
                case gdcm::PixelFormat::UINT8: m_type = CV_8UC1; break;
                case gdcm::PixelFormat::INT16: m_type = CV_16SC1; break;
                case gdcm::PixelFormat::UINT16: m_type = CV_16UC1; break;
                case gdcm::PixelFormat::INT32: m_type = CV_32SC1; break;
                case gdcm::PixelFormat::FLOAT32: m_type = CV_32FC1; break;
                case gdcm::PixelFormat::FLOAT64: m_type = CV_64FC1; break;
                default: bOK = false; DBG("GDCM | Monochrome scalar type not supported\n"); break;
            }
            break;
        }

        case gdcm::PhotometricInterpretation::RGB:
        {
            switch (csImage.GetPixelFormat().GetScalarType())
            {
                case gdcm::PixelFormat::UINT8: m_type = CV_8UC3; break;
                default: bOK = false; DBG("GDCM | RGB scalar type not supported\n"); break;
            }
            break;
        }

        default:
        {
            bOK = false;
            DBG("GDCM | PI not supported: %s\n", csImage.GetPhotometricInterpretation().GetString());
            break;
        }
    }

    if(bOK)
    {
        unsigned int ndim = csImage.GetNumberOfDimensions();
        if (ndim != 2)
        {
            DBG("GDCM | Invalid dimensions number: %d\n", ndim);
            bOK = false;
        }
    }
    if (bOK)
    {
        const unsigned int *piDimension = csImage.GetDimensions();
        m_height = piDimension[0];
        m_width = piDimension[1];
        if( ( m_width <=0 )  || ( m_height <=0 ) )
        {
            DBG("GDCM | Invalid dimensions: %d x %d\n", piDimension[0], piDimension[1]);
            bOK = false;
        }
    }

    return(bOK);
}


bool  DICOMDecoder::readData( Mat& csImage )
{
    csImage.create(m_width,m_height,m_type);

    gdcm::ImageReader csImageReader;
    csImageReader.SetFileName(m_filename.c_str());
    if(!csImageReader.Read())
    {
        DBG("GDCM | Failed to Read\n");
        return false;
    }

    const gdcm::Image &img = csImageReader.GetImage();

    unsigned long len = img.GetBufferLength();
    if (len > csImage.elemSize() * csImage.total())
    {
        DBG("GDCM | Buffer is bigger than Mat: %ld > %ld * %ld\n", len, csImage.elemSize(), csImage.total());
        return false;
    }

    if (!img.GetBuffer((char*)csImage.ptr()))
    {
        DBG("GDCM | Failed to GetBuffer\n");
        return false;
    }
    DBG("GDCM | Read OK\n");
    return true;
}

}

#endif // HAVE_GDCM