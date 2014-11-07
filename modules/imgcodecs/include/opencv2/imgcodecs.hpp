/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
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

#ifndef __OPENCV_IMGCODECS_HPP__
#define __OPENCV_IMGCODECS_HPP__

#include "opencv2/core.hpp"

//////////////////////////////// image codec ////////////////////////////////
namespace cv
{

enum { IMREAD_UNCHANGED  = -1, // 8bit, color or not
       IMREAD_GRAYSCALE  = 0,  // 8bit, gray
       IMREAD_COLOR      = 1,  // ?, color
       IMREAD_ANYDEPTH   = 2,  // any depth, ?
       IMREAD_ANYCOLOR   = 4,  // ?, any color
       IMREAD_LOAD_GDAL  = 8   // Use gdal driver
     };

enum { IMWRITE_JPEG_QUALITY        = 1,
       IMWRITE_JPEG_PROGRESSIVE    = 2,
       IMWRITE_JPEG_OPTIMIZE       = 3,
       IMWRITE_JPEG_RST_INTERVAL   = 4,
       IMWRITE_JPEG_LUMA_QUALITY   = 5,
       IMWRITE_JPEG_CHROMA_QUALITY = 6,
       IMWRITE_PNG_COMPRESSION     = 16,
       IMWRITE_PNG_STRATEGY        = 17,
       IMWRITE_PNG_BILEVEL         = 18,
       IMWRITE_PXM_BINARY          = 32,
       IMWRITE_WEBP_QUALITY        = 64
     };

enum { IMWRITE_PNG_STRATEGY_DEFAULT      = 0,
       IMWRITE_PNG_STRATEGY_FILTERED     = 1,
       IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY = 2,
       IMWRITE_PNG_STRATEGY_RLE          = 3,
       IMWRITE_PNG_STRATEGY_FIXED        = 4
     };

CV_EXPORTS_W Mat imread( const String& filename, int flags = IMREAD_COLOR );

CV_EXPORTS_W bool imwrite( const String& filename, InputArray img,
              const std::vector<int>& params = std::vector<int>());

CV_EXPORTS_W Mat imdecode( InputArray buf, int flags );

CV_EXPORTS Mat imdecode( InputArray buf, int flags, Mat* dst);

CV_EXPORTS_W bool imencode( const String& ext, InputArray img,
                            CV_OUT std::vector<uchar>& buf,
                            const std::vector<int>& params = std::vector<int>());

} // cv

#endif //__OPENCV_IMGCODECS_HPP__
