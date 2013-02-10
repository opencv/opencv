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

#ifdef HAVE_WEBP

#include <webp/decode.h>
#include <stdio.h>

#include "precomp.hpp"
#include "grfmt_webp.hpp"

namespace cv
{

WebPDecoder::WebPDecoder()
{
	m_signature = "RIFF....WEBPVP8 ";
	m_buf_supported = true;
}

WebPDecoder::~WebPDecoder()
{
}

ImageDecoder WebPDecoder::newDecoder() const
{
    return new WebPDecoder;
}

bool WebPDecoder::checkSignature( const string& signature ) const
{
	size_t len = signatureLength();
	bool ret = false;
	
	if(signature.size() >= len)
	{
		ret = ( (memcmp(signature.c_str(), m_signature.c_str(), 4) == 0) &&
			(memcmp(signature.c_str() + 8, m_signature.c_str() + 8, 8) == 0) );
	}
	
	return ret;
}

bool WebPDecoder::readHeader()
{
	bool header_read = false;
	
	FILE *webp_file = NULL;
	webp_file = fopen(m_filename.c_str(), "rb");
	
	if(webp_file != NULL)
	{	
		uint8_t *webp_file_data = NULL;
		size_t webp_file_data_size = 0;
	
		fseek(webp_file, 0, SEEK_END);
		webp_file_data_size = ftell(webp_file);
		fseek(webp_file, 0, SEEK_SET);
	
		webp_file_data = (uint8_t *) malloc (webp_file_data_size);
	
		if(webp_file_data != NULL)
		{
			size_t data_read_size = fread(webp_file_data, 1, webp_file_data_size, webp_file);
			if(data_read_size == webp_file_data_size)
			{
				if(WebPGetInfo(webp_file_data, webp_file_data_size, &m_width, &m_height) == 1)
				{
					header_read = true;
					
					/*
					 * According to VP8 bitstream format these are the default
					 * values.
					 */
					m_type = CV_8UC3;			
					// m_origin = IPL_ORIGIN_TL;	// TODO Check if this is correct.
					// m_bpp = 24;
				}
				else
				{
					printf("Cannot get width, height information about %s\n",
						m_filename.c_str());
				}
			}
			else
			{
				printf("Not able to read %lu bytes from file %s", webp_file_data_size
					, m_filename.c_str());
			}
		
			free(webp_file_data); webp_file_data = NULL;
		}
	
		fclose(webp_file); webp_file = NULL;
	}
	
	return header_read;
}

bool WebPDecoder::readData(Mat &img)
{
	bool data_read = false;

	uint8_t *webp_file_data = NULL;
	size_t webp_file_data_size = 0;

	FILE *webp_file = NULL;
	webp_file = fopen(m_filename.c_str(), "rb");

	if(webp_file != NULL)
	{
		fseek(webp_file, 0, SEEK_END);
		webp_file_data_size = ftell(webp_file);
		fseek(webp_file, 0, SEEK_SET);

		webp_file_data = (uint8_t *) malloc (webp_file_data_size);

		if(webp_file_data != NULL)
		{
			size_t data_read_size = fread(webp_file_data, 1, webp_file_data_size,
											webp_file);
			if( (data_read_size == webp_file_data_size) &&
				(m_width > 0 && m_height > 0) )
			{
				uchar* out_data = img.data;
				unsigned int out_data_size = m_width * m_height * 3 * sizeof(uchar);
				WebPDecodeBGRInto(webp_file_data, webp_file_data_size, out_data,
									out_data_size, m_width * 3);
			}
			else
			{
				printf("Not able to read %lu bytes from file %s", webp_file_data_size
					, m_filename.c_str());
			}
	
			free(webp_file_data); webp_file_data = NULL;
		}

		fclose(webp_file); webp_file = NULL;
	}

	return data_read;
}

}

#endif
