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
#include "gdcm_dicom.hpp"

#include <gdcmImageReader.h>

namespace cv
{

/************************ DICOM decoder *****************************/

DICOMDecoder::DICOMDecoder()
{
    /// DICOM preable is 128 bytes (can have any vale, defaults to x00) + 4 bytes magic number (DICM)
    m_signature = "";
    for(int iSize=0; iSize<128; iSize++)
    {
        m_signature = m_signature + "\xFF";
    }

    m_signature = m_signature + "\x44\x49\x43\x4D";

    m_buf_supported = false;
}


DICOMDecoder::~DICOMDecoder()
{
}

bool DICOMDecoder::checkSignature( const String& signature ) const
{
    size_t len = signatureLength();
    bool bOK = signature.size() >= len;
    for(int iIndex = 128; iIndex < len; iIndex++)
    {
        if(signature[iIndex] == m_signature[iIndex])
        {
            continue;
        }
        else
        {
            bOK = false;
            break;
        }
    }

    return(bOK);
}

void  DICOMDecoder::close()
{
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
        return(false);
    }

    bool bOK = true;

    const gdcm::Image &csImage = csImageReader.GetImage();
    if(        ( csImage.GetPhotometricInterpretation() == gdcm::PhotometricInterpretation::MONOCHROME1 )
        ||    ( csImage.GetPhotometricInterpretation() == gdcm::PhotometricInterpretation::MONOCHROME2 )
        )
    {
        gdcm::PixelFormat ePixelFormat = csImage.GetPixelFormat();
        if( ePixelFormat == gdcm::PixelFormat::INT8)
        {
            m_type = CV_8SC1;
        }
        else if( ePixelFormat == gdcm::PixelFormat::UINT8)
        {
            m_type = CV_8UC1;
        }
        else if( ePixelFormat == gdcm::PixelFormat::INT16)
        {
            m_type = CV_16SC1;
        }
        else if( ePixelFormat == gdcm::PixelFormat::UINT16)
        {
            m_type = CV_16UC1;
        }
        else if( ePixelFormat == gdcm::PixelFormat::INT32)
        {
            m_type = CV_32SC1;
        }
        else if( ePixelFormat == gdcm::PixelFormat::FLOAT32)
        {
            m_type = CV_32FC1;
        }
        else if( ePixelFormat == gdcm::PixelFormat::FLOAT64)
        {
            m_type = CV_64FC1;
        }
        else if( ePixelFormat == gdcm::PixelFormat::INT12)
        {
            bOK = false;
        }
        else if( ePixelFormat == gdcm::PixelFormat::UINT12)
        {
            bOK = false;
        }
        else if( ePixelFormat == gdcm::PixelFormat::UINT32)
        {
            bOK = false;
        }
        else if( ePixelFormat == gdcm::PixelFormat::SINGLEBIT)
        {
            bOK = false;
        }
        else
        {
            bOK = false;
        }
    }
    else if( csImage.GetPhotometricInterpretation() == gdcm::PhotometricInterpretation::RGB )
    {
        gdcm::PixelFormat ePixelFormat = csImage.GetPixelFormat();
        if( ePixelFormat == gdcm::PixelFormat::UINT8)
        {
            m_type = CV_8UC3;
        }
        else
        {
            bOK = false;
        }
    }
    else
    {
        bOK = false;
    }

    if(bOK)
    {
        const unsigned int *piDimension = csImage.GetDimensions();
        m_width = piDimension[0];
        m_height = piDimension[1];
        if( ( m_width <=0 )  || ( m_height <=0 ) )
        {
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
        return(false);
    }

    bool bOK = true;
    const gdcm::Image &csGDCMImage = csImageReader.GetImage();
    bOK = csGDCMImage.GetBuffer((char*)csImage.ptr());

    return(bOK);
}
}
