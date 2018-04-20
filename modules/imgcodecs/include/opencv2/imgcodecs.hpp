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

#ifndef OPENCV_IMGCODECS_HPP
#define OPENCV_IMGCODECS_HPP

#include "opencv2/core.hpp"

/**
  @defgroup imgcodecs Image file reading and writing
  @{
    @defgroup imgcodecs_c C API
    @defgroup imgcodecs_ios iOS glue
  @}
*/

//////////////////////////////// image codec ////////////////////////////////
namespace cv
{

//! @addtogroup imgcodecs
//! @{

//! Imread flags
enum ImreadModes {
       IMREAD_UNCHANGED            = -1, //!< If set, return the loaded image as is (with alpha channel, otherwise it gets cropped).
       IMREAD_GRAYSCALE            = 0,  //!< If set, always convert image to the single channel grayscale image.
       IMREAD_COLOR                = 1,  //!< If set, always convert image to the 3 channel BGR color image.
       IMREAD_ANYDEPTH             = 2,  //!< If set, return 16-bit/32-bit image when the input has the corresponding depth, otherwise convert it to 8-bit.
       IMREAD_ANYCOLOR             = 4,  //!< If set, the image is read in any possible color format.
       IMREAD_LOAD_GDAL            = 8,  //!< If set, use the gdal driver for loading the image.
       IMREAD_REDUCED_GRAYSCALE_2  = 16, //!< If set, always convert image to the single channel grayscale image and the image size reduced 1/2.
       IMREAD_REDUCED_COLOR_2      = 17, //!< If set, always convert image to the 3 channel BGR color image and the image size reduced 1/2.
       IMREAD_REDUCED_GRAYSCALE_4  = 32, //!< If set, always convert image to the single channel grayscale image and the image size reduced 1/4.
       IMREAD_REDUCED_COLOR_4      = 33, //!< If set, always convert image to the 3 channel BGR color image and the image size reduced 1/4.
       IMREAD_REDUCED_GRAYSCALE_8  = 64, //!< If set, always convert image to the single channel grayscale image and the image size reduced 1/8.
       IMREAD_REDUCED_COLOR_8      = 65, //!< If set, always convert image to the 3 channel BGR color image and the image size reduced 1/8.
       IMREAD_IGNORE_ORIENTATION   = 128 //!< If set, do not rotate the image according to EXIF's orientation flag.
     };

//! Imwrite flags
enum ImwriteFlags {
       IMWRITE_JPEG_QUALITY        = 1,  //!< For JPEG, it can be a quality from 0 to 100 (the higher is the better). Default value is 95.
       IMWRITE_JPEG_PROGRESSIVE    = 2,  //!< Enable JPEG features, 0 or 1, default is False.
       IMWRITE_JPEG_OPTIMIZE       = 3,  //!< Enable JPEG features, 0 or 1, default is False.
       IMWRITE_JPEG_RST_INTERVAL   = 4,  //!< JPEG restart interval, 0 - 65535, default is 0 - no restart.
       IMWRITE_JPEG_LUMA_QUALITY   = 5,  //!< Separate luma quality level, 0 - 100, default is 0 - don't use.
       IMWRITE_JPEG_CHROMA_QUALITY = 6,  //!< Separate chroma quality level, 0 - 100, default is 0 - don't use.
       IMWRITE_PNG_COMPRESSION     = 16, //!< For PNG, it can be the compression level from 0 to 9. A higher value means a smaller size and longer compression time. If specified, strategy is changed to IMWRITE_PNG_STRATEGY_DEFAULT (Z_DEFAULT_STRATEGY). Default value is 1 (best speed setting).
       IMWRITE_PNG_STRATEGY        = 17, //!< One of cv::ImwritePNGFlags, default is IMWRITE_PNG_STRATEGY_RLE.
       IMWRITE_PNG_BILEVEL         = 18, //!< Binary level PNG, 0 or 1, default is 0.
       IMWRITE_PXM_BINARY          = 32, //!< For PPM, PGM, or PBM, it can be a binary format flag, 0 or 1. Default value is 1.
       IMWRITE_EXR_TYPE            = (3 << 4) + 0, /* 48 */ //!< override EXR storage type (FLOAT (FP32) is default)
       IMWRITE_WEBP_QUALITY        = 64, //!< For WEBP, it can be a quality from 1 to 100 (the higher is the better). By default (without any parameter) and for quality above 100 the lossless compression is used.
       IMWRITE_PAM_TUPLETYPE       = 128,//!< For PAM, sets the TUPLETYPE field to the corresponding string value that is defined for the format
       IMWRITE_TIFF_RESUNIT = 256,//!< For TIFF, use to specify which DPI resolution unit to set; see libtiff documentation for valid values
       IMWRITE_TIFF_XDPI = 257,//!< For TIFF, use to specify the X direction DPI
       IMWRITE_TIFF_YDPI = 258 //!< For TIFF, use to specify the Y direction DPI
     };

enum ImwriteEXRTypeFlags {
       /*IMWRITE_EXR_TYPE_UNIT = 0, //!< not supported */
       IMWRITE_EXR_TYPE_HALF = 1,   //!< store as HALF (FP16)
       IMWRITE_EXR_TYPE_FLOAT = 2   //!< store as FP32 (default)
     };

//! Imwrite PNG specific flags used to tune the compression algorithm.
/** These flags will be modify the way of PNG image compression and will be passed to the underlying zlib processing stage.

-   The effect of IMWRITE_PNG_STRATEGY_FILTERED is to force more Huffman coding and less string matching; it is somewhat intermediate between IMWRITE_PNG_STRATEGY_DEFAULT and IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY.
-   IMWRITE_PNG_STRATEGY_RLE is designed to be almost as fast as IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY, but give better compression for PNG image data.
-   The strategy parameter only affects the compression ratio but not the correctness of the compressed output even if it is not set appropriately.
-   IMWRITE_PNG_STRATEGY_FIXED prevents the use of dynamic Huffman codes, allowing for a simpler decoder for special applications.
*/
enum ImwritePNGFlags {
       IMWRITE_PNG_STRATEGY_DEFAULT      = 0, //!< Use this value for normal data.
       IMWRITE_PNG_STRATEGY_FILTERED     = 1, //!< Use this value for data produced by a filter (or predictor).Filtered data consists mostly of small values with a somewhat random distribution. In this case, the compression algorithm is tuned to compress them better.
       IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY = 2, //!< Use this value to force Huffman encoding only (no string match).
       IMWRITE_PNG_STRATEGY_RLE          = 3, //!< Use this value to limit match distances to one (run-length encoding).
       IMWRITE_PNG_STRATEGY_FIXED        = 4  //!< Using this value prevents the use of dynamic Huffman codes, allowing for a simpler decoder for special applications.
     };

//! Imwrite PAM specific tupletype flags used to define the 'TUPETYPE' field of a PAM file.
enum ImwritePAMFlags {
       IMWRITE_PAM_FORMAT_NULL = 0,
       IMWRITE_PAM_FORMAT_BLACKANDWHITE = 1,
       IMWRITE_PAM_FORMAT_GRAYSCALE = 2,
       IMWRITE_PAM_FORMAT_GRAYSCALE_ALPHA = 3,
       IMWRITE_PAM_FORMAT_RGB = 4,
       IMWRITE_PAM_FORMAT_RGB_ALPHA = 5,
     };

/** @brief Loads an image from a file.

@anchor imread

The function imread loads an image from the specified file and returns it. If the image cannot be
read (because of missing file, improper permissions, unsupported or invalid format), the function
returns an empty matrix ( Mat::data==NULL ).

Currently, the following file formats are supported:

-   Windows bitmaps - \*.bmp, \*.dib (always supported)
-   JPEG files - \*.jpeg, \*.jpg, \*.jpe (see the *Notes* section)
-   JPEG 2000 files - \*.jp2 (see the *Notes* section)
-   Portable Network Graphics - \*.png (see the *Notes* section)
-   WebP - \*.webp (see the *Notes* section)
-   Portable image format - \*.pbm, \*.pgm, \*.ppm \*.pxm, \*.pnm (always supported)
-   Sun rasters - \*.sr, \*.ras (always supported)
-   TIFF files - \*.tiff, \*.tif (see the *Notes* section)
-   OpenEXR Image files - \*.exr (see the *Notes* section)
-   Radiance HDR - \*.hdr, \*.pic (always supported)
-   Raster and Vector geospatial data supported by Gdal (see the *Notes* section)

@note

-   The function determines the type of an image by the content, not by the file extension.
-   In the case of color images, the decoded images will have the channels stored in **B G R** order.
-   On Microsoft Windows\* OS and MacOSX\*, the codecs shipped with an OpenCV image (libjpeg,
    libpng, libtiff, and libjasper) are used by default. So, OpenCV can always read JPEGs, PNGs,
    and TIFFs. On MacOSX, there is also an option to use native MacOSX image readers. But beware
    that currently these native image loaders give images with different pixel values because of
    the color management embedded into MacOSX.
-   On Linux\*, BSD flavors and other Unix-like open-source operating systems, OpenCV looks for
    codecs supplied with an OS image. Install the relevant packages (do not forget the development
    files, for example, "libjpeg-dev", in Debian\* and Ubuntu\*) to get the codec support or turn
    on the OPENCV_BUILD_3RDPARTY_LIBS flag in CMake.
-   In the case you set *WITH_GDAL* flag to true in CMake and @ref IMREAD_LOAD_GDAL to load the image,
    then [GDAL](http://www.gdal.org) driver will be used in order to decode the image by supporting
    the following formats: [Raster](http://www.gdal.org/formats_list.html),
    [Vector](http://www.gdal.org/ogr_formats.html).
-   If EXIF information are embedded in the image file, the EXIF orientation will be taken into account
    and thus the image will be rotated accordingly except if the flag @ref IMREAD_IGNORE_ORIENTATION is passed.
@param filename Name of file to be loaded.
@param flags Flag that can take values of cv::ImreadModes
*/
CV_EXPORTS_W Mat imread( const String& filename, int flags = IMREAD_COLOR );

/** @brief Loads a multi-page image from a file.

The function imreadmulti loads a multi-page image from the specified file into a vector of Mat objects.
@param filename Name of file to be loaded.
@param flags Flag that can take values of cv::ImreadModes, default with cv::IMREAD_ANYCOLOR.
@param mats A vector of Mat objects holding each page, if more than one.
@sa cv::imread
*/
CV_EXPORTS_W bool imreadmulti(const String& filename, CV_OUT std::vector<Mat>& mats, int flags = IMREAD_ANYCOLOR);

/** @brief Saves an image to a specified file.

The function imwrite saves the image to the specified file. The image format is chosen based on the
filename extension (see cv::imread for the list of extensions). Only 8-bit (or 16-bit unsigned (CV_16U)
in case of PNG, JPEG 2000, and TIFF) single-channel or 3-channel (with 'BGR' channel order) images
can be saved using this function. If the format, depth or channel order is different, use
Mat::convertTo , and cv::cvtColor to convert it before saving. Or, use the universal FileStorage I/O
functions to save the image to XML or YAML format.

It is possible to store PNG images with an alpha channel using this function. To do this, create
8-bit (or 16-bit) 4-channel image BGRA, where the alpha channel goes last. Fully transparent pixels
should have alpha set to 0, fully opaque pixels should have alpha set to 255/65535.

The sample below shows how to create such a BGRA image and store to PNG file. It also demonstrates how to set custom
compression parameters :
@include snippets/imgcodecs_imwrite.cpp
@param filename Name of the file.
@param img Image to be saved.
@param params Format-specific parameters encoded as pairs (paramId_1, paramValue_1, paramId_2, paramValue_2, ... .) see cv::ImwriteFlags
*/
CV_EXPORTS_W bool imwrite( const String& filename, InputArray img,
              const std::vector<int>& params = std::vector<int>());

/** @brief Reads an image from a buffer in memory.

The function imdecode reads an image from the specified buffer in the memory. If the buffer is too short or
contains invalid data, the function returns an empty matrix ( Mat::data==NULL ).

See cv::imread for the list of supported formats and flags description.

@note In the case of color images, the decoded images will have the channels stored in **B G R** order.
@param buf Input array or vector of bytes.
@param flags The same flags as in cv::imread, see cv::ImreadModes.
*/
CV_EXPORTS_W Mat imdecode( InputArray buf, int flags );

/** @overload
@param buf
@param flags
@param dst The optional output placeholder for the decoded matrix. It can save the image
reallocations when the function is called repeatedly for images of the same size.
*/
CV_EXPORTS Mat imdecode( InputArray buf, int flags, Mat* dst);

/** @brief Encodes an image into a memory buffer.

The function imencode compresses the image and stores it in the memory buffer that is resized to fit the
result. See cv::imwrite for the list of supported formats and flags description.

@param ext File extension that defines the output format.
@param img Image to be written.
@param buf Output buffer resized to fit the compressed image.
@param params Format-specific parameters. See cv::imwrite and cv::ImwriteFlags.
*/
CV_EXPORTS_W bool imencode( const String& ext, InputArray img,
                            CV_OUT std::vector<uchar>& buf,
                            const std::vector<int>& params = std::vector<int>());

//! @} imgcodecs

} // cv

#endif //OPENCV_IMGCODECS_HPP
