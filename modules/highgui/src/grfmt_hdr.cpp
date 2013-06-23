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

#include "grfmt_hdr.hpp"
#include "rgbe.hpp"

namespace cv
{
HdrDecoder::HdrDecoder()
{
	m_signature = "#?RGBE";
	m_signature_alt = "#?RADIANCE";
	file = NULL;
	m_type = CV_32FC3;
}

HdrDecoder::~HdrDecoder()
{
}

size_t HdrDecoder::signatureLength() const
{
	return m_signature.size() > m_signature_alt.size() ?
		   m_signature.size() : m_signature_alt.size();
}

bool  HdrDecoder::readHeader()
{
	file = fopen(m_filename.c_str(), "rb");
	if(!file) {
		CV_Error(Error::StsError, "HDR decoder: can't open file");
	}
	RGBE_ReadHeader(file, &m_width, &m_height, NULL);
	if(m_width <= 0 || m_height <= 0) {
		CV_Error(Error::StsError, "HDR decoder: invalid image size");
	}
	return true;
}

bool HdrDecoder::readData(Mat& img)
{
	if(!file) {
		readHeader();
	}
	if(img.cols != m_width || img.rows != m_height ||
	   img.type() != CV_32FC3) {
		CV_Error(Error::StsError, "HDR decoder: bad mat");
	}
	RGBE_ReadPixels_RLE(file, const_cast<float*>(img.ptr<float>()), img.cols, img.rows);
	fclose(file); file = NULL;
	return true;
}

bool HdrDecoder::checkSignature( const String& signature ) const
{
	if(signature.size() >= (m_signature.size()) && 
	   !memcmp(signature.c_str(), m_signature.c_str(), m_signature.size()))
	   return true;
	if(signature.size() >= (m_signature.size()) && 
	   !memcmp(signature.c_str(), m_signature_alt.c_str(), m_signature_alt.size()))
	   return true;
	return false;
}

ImageDecoder HdrDecoder::newDecoder() const
{
	return new HdrDecoder;
}

HdrEncoder::HdrEncoder()
{
	m_description = "Radiance HDR (*.hdr;*.pic)";
}

HdrEncoder::~HdrEncoder()
{
}

bool HdrEncoder::write( const Mat& img, const std::vector<int>& params )
{
	if(img.type() != CV_32FC3) {
		CV_Error(Error::StsBadArg, "HDR encoder: need 32FC3 mat");
	}
	if(!(params.empty() || params[0] == HDR_NONE || params[0] == HDR_RLE)) {
		CV_Error(Error::StsBadArg, "HDR encoder: wrong compression param");
	}

	FILE *fout = fopen(m_filename.c_str(), "wb");
	if(!fout) {
		CV_Error(Error::StsError, "HDR encoder: can't open file");
	}

	RGBE_WriteHeader(fout, img.cols, img.rows, NULL);
	if(params.empty() || params[0] == HDR_RLE) {
		RGBE_WritePixels_RLE(fout, const_cast<float*>(img.ptr<float>()), img.cols, img.rows);
	} else {
		RGBE_WritePixels(fout, const_cast<float*>(img.ptr<float>()), img.cols * img.rows);
	}

	fclose(fout);
	return true;
}

ImageEncoder HdrEncoder::newEncoder() const
{
	return new HdrEncoder;
}

bool HdrEncoder::isFormatSupported( int depth ) const {
	return depth == CV_32F;
}

}
