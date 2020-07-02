// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_IMGCODECS_LEGACY_CONSTANTS_H
#define OPENCV_IMGCODECS_LEGACY_CONSTANTS_H

/* duplicate of "ImreadModes" enumeration for better compatibility with OpenCV 3.x */
enum
{
/* 8bit, color or not */
    CV_LOAD_IMAGE_UNCHANGED  =-1,
/* 8bit, gray */
    CV_LOAD_IMAGE_GRAYSCALE  =0,
/* ?, color */
    CV_LOAD_IMAGE_COLOR      =1,
/* any depth, ? */
    CV_LOAD_IMAGE_ANYDEPTH   =2,
/* ?, any color */
    CV_LOAD_IMAGE_ANYCOLOR   =4,
/* ?, no rotate */
    CV_LOAD_IMAGE_IGNORE_ORIENTATION  =128
};

/* duplicate of "ImwriteFlags" enumeration for better compatibility with OpenCV 3.x */
enum
{
    CV_IMWRITE_JPEG_QUALITY =1,
    CV_IMWRITE_JPEG_PROGRESSIVE =2,
    CV_IMWRITE_JPEG_OPTIMIZE =3,
    CV_IMWRITE_JPEG_RST_INTERVAL =4,
    CV_IMWRITE_JPEG_LUMA_QUALITY =5,
    CV_IMWRITE_JPEG_CHROMA_QUALITY =6,
    CV_IMWRITE_PNG_COMPRESSION =16,
    CV_IMWRITE_PNG_STRATEGY =17,
    CV_IMWRITE_PNG_BILEVEL =18,
    CV_IMWRITE_PNG_STRATEGY_DEFAULT =0,
    CV_IMWRITE_PNG_STRATEGY_FILTERED =1,
    CV_IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY =2,
    CV_IMWRITE_PNG_STRATEGY_RLE =3,
    CV_IMWRITE_PNG_STRATEGY_FIXED =4,
    CV_IMWRITE_PXM_BINARY =32,
    CV_IMWRITE_EXR_TYPE = 48,
    CV_IMWRITE_WEBP_QUALITY =64,
    CV_IMWRITE_PAM_TUPLETYPE = 128,
    CV_IMWRITE_PAM_FORMAT_NULL = 0,
    CV_IMWRITE_PAM_FORMAT_BLACKANDWHITE = 1,
    CV_IMWRITE_PAM_FORMAT_GRAYSCALE = 2,
    CV_IMWRITE_PAM_FORMAT_GRAYSCALE_ALPHA = 3,
    CV_IMWRITE_PAM_FORMAT_RGB = 4,
    CV_IMWRITE_PAM_FORMAT_RGB_ALPHA = 5,
};

#endif // OPENCV_IMGCODECS_LEGACY_CONSTANTS_H
