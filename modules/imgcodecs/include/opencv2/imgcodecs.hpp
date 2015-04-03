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

/** @brief Loads an image from a file.

@anchor imread

@param filename Name of file to be loaded.
@param flags Flags specifying the color type of a loaded image:
-   CV_LOAD_IMAGE_ANYDEPTH - If set, return 16-bit/32-bit image when the input has the
    corresponding depth, otherwise convert it to 8-bit.
-   CV_LOAD_IMAGE_COLOR - If set, always convert image to the color one
-   CV_LOAD_IMAGE_GRAYSCALE - If set, always convert image to the grayscale one
-   **\>0** Return a 3-channel color image.

@note In the current implementation the alpha channel, if any, is stripped from the output image.
Use negative value if you need the alpha channel.

-   **=0** Return a grayscale image.
-   **\<0** Return the loaded image as is (with alpha channel).

The function imread loads an image from the specified file and returns it. If the image cannot be
read (because of missing file, improper permissions, unsupported or invalid format), the function
returns an empty matrix ( Mat::data==NULL ). Currently, the following file formats are supported:

-   Windows bitmaps - \*.bmp, \*.dib (always supported)
-   JPEG files - \*.jpeg, \*.jpg, \*.jpe (see the *Notes* section)
-   JPEG 2000 files - \*.jp2 (see the *Notes* section)
-   Portable Network Graphics - \*.png (see the *Notes* section)
-   WebP - \*.webp (see the *Notes* section)
-   Portable image format - \*.pbm, \*.pgm, \*.ppm (always supported)
-   Sun rasters - \*.sr, \*.ras (always supported)
-   TIFF files - \*.tiff, \*.tif (see the *Notes* section)

@note

-   The function determines the type of an image by the content, not by the file extension.
-   On Microsoft Windows\* OS and MacOSX\*, the codecs shipped with an OpenCV image (libjpeg,
    libpng, libtiff, and libjasper) are used by default. So, OpenCV can always read JPEGs, PNGs,
    and TIFFs. On MacOSX, there is also an option to use native MacOSX image readers. But beware
    that currently these native image loaders give images with different pixel values because of
    the color management embedded into MacOSX.
-   On Linux\*, BSD flavors and other Unix-like open-source operating systems, OpenCV looks for
    codecs supplied with an OS image. Install the relevant packages (do not forget the development
    files, for example, "libjpeg-dev", in Debian\* and Ubuntu\*) to get the codec support or turn
    on the OPENCV_BUILD_3RDPARTY_LIBS flag in CMake.

@note In the case of color images, the decoded images will have the channels stored in B G R order.
 */
CV_EXPORTS_W Mat imread( const String& filename, int flags = IMREAD_COLOR );

/** @brief Loads a multi-page image from a file. (see imread for details.)

@param filename Name of file to be loaded.
@param flags Flags specifying the color type of a loaded image (see imread).
            Defaults to IMREAD_ANYCOLOR, as each page may be different.
@param mats A vector of Mat objects holding each page, if more than one.

*/
CV_EXPORTS_W bool imreadmulti(const String& filename, std::vector<Mat>& mats, int flags = IMREAD_ANYCOLOR);

/** @brief Saves an image to a specified file.

@param filename Name of the file.
@param img Image to be saved.
@param params Format-specific save parameters encoded as pairs
paramId_1, paramValue_1, paramId_2, paramValue_2, ... . The following parameters are currently
supported:
-   For JPEG, it can be a quality ( CV_IMWRITE_JPEG_QUALITY ) from 0 to 100 (the higher is
    the better). Default value is 95.
-   For WEBP, it can be a quality ( CV_IMWRITE_WEBP_QUALITY ) from 1 to 100 (the higher is
    the better). By default (without any parameter) and for quality above 100 the lossless
    compression is used.
-   For PNG, it can be the compression level ( CV_IMWRITE_PNG_COMPRESSION ) from 0 to 9. A
    higher value means a smaller size and longer compression time. Default value is 3.
-   For PPM, PGM, or PBM, it can be a binary format flag ( CV_IMWRITE_PXM_BINARY ), 0 or 1.
    Default value is 1.

The function imwrite saves the image to the specified file. The image format is chosen based on the
filename extension (see imread for the list of extensions). Only 8-bit (or 16-bit unsigned (CV_16U)
in case of PNG, JPEG 2000, and TIFF) single-channel or 3-channel (with 'BGR' channel order) images
can be saved using this function. If the format, depth or channel order is different, use
Mat::convertTo , and cvtColor to convert it before saving. Or, use the universal FileStorage I/O
functions to save the image to XML or YAML format.

It is possible to store PNG images with an alpha channel using this function. To do this, create
8-bit (or 16-bit) 4-channel image BGRA, where the alpha channel goes last. Fully transparent pixels
should have alpha set to 0, fully opaque pixels should have alpha set to 255/65535. The sample below
shows how to create such a BGRA image and store to PNG file. It also demonstrates how to set custom
compression parameters :
@code
    #include <vector>
    #include <stdio.h>
    #include <opencv2/opencv.hpp>

    using namespace cv;
    using namespace std;

    void createAlphaMat(Mat &mat)
    {
        CV_Assert(mat.channels() == 4);
        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                Vec4b& bgra = mat.at<Vec4b>(i, j);
                bgra[0] = UCHAR_MAX; // Blue
                bgra[1] = saturate_cast<uchar>((float (mat.cols - j)) / ((float)mat.cols) * UCHAR_MAX); // Green
                bgra[2] = saturate_cast<uchar>((float (mat.rows - i)) / ((float)mat.rows) * UCHAR_MAX); // Red
                bgra[3] = saturate_cast<uchar>(0.5 * (bgra[1] + bgra[2])); // Alpha
            }
        }
    }

    int main(int argv, char **argc)
    {
        // Create mat with alpha channel
        Mat mat(480, 640, CV_8UC4);
        createAlphaMat(mat);

        vector<int> compression_params;
        compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);

        try {
            imwrite("alpha.png", mat, compression_params);
        }
        catch (runtime_error& ex) {
            fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
            return 1;
        }

        fprintf(stdout, "Saved PNG file with alpha data.\n");
        return 0;
    }
@endcode
 */
CV_EXPORTS_W bool imwrite( const String& filename, InputArray img,
              const std::vector<int>& params = std::vector<int>());

/** @overload */
CV_EXPORTS_W Mat imdecode( InputArray buf, int flags );

/** @brief Reads an image from a buffer in memory.

@param buf Input array or vector of bytes.
@param flags The same flags as in imread .
@param dst The optional output placeholder for the decoded matrix. It can save the image
reallocations when the function is called repeatedly for images of the same size.

The function reads an image from the specified buffer in the memory. If the buffer is too short or
contains invalid data, the empty matrix/image is returned.

See imread for the list of supported formats and flags description.

@note In the case of color images, the decoded images will have the channels stored in B G R order.
 */
CV_EXPORTS Mat imdecode( InputArray buf, int flags, Mat* dst);

/** @brief Encodes an image into a memory buffer.

@param ext File extension that defines the output format.
@param img Image to be written.
@param buf Output buffer resized to fit the compressed image.
@param params Format-specific parameters. See imwrite .

The function compresses the image and stores it in the memory buffer that is resized to fit the
result. See imwrite for the list of supported formats and flags description.

@note cvEncodeImage returns single-row matrix of type CV_8UC1 that contains encoded image as array
of bytes.
 */
CV_EXPORTS_W bool imencode( const String& ext, InputArray img,
                            CV_OUT std::vector<uchar>& buf,
                            const std::vector<int>& params = std::vector<int>());

//! @} imgcodecs

} // cv

#endif //__OPENCV_IMGCODECS_HPP__
