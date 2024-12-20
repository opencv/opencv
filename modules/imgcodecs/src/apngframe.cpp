// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html

/****************************************************************************\
 *
 *  this file includes some modified part of apngasm
 *
 ****************************************************************************/


 /*  apngasm
 *
 *  The next generation of apngasm, the APNG Assembler.
 *  The apngasm CLI tool and library can assemble and disassemble APNG image files.
 *
 *  https://github.com/apngasm/apngasm
 *
 * zlib license
 * ------------
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 */

#include "apngframe.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <png.h>

#if defined _MSC_VER && _MSC_VER >= 1200
 // interaction between '_setjmp' and C++ object destruction is non-portable
#pragma warning( disable: 4611 )
#pragma warning( disable: 4244 )
#endif

// the following defines are a hack to avoid multiple problems with frame pointer handling and setjmp
// see http://gcc.gnu.org/ml/gcc/2011-10/msg00324.html for some details
#define mingw_getsp(...) 0
#define __builtin_frame_address(...) 0

namespace cv {

    APNGFrame::APNGFrame()
    {
        _pixels = NULL;
        _width = 0;
        _height = 0;
        _colorType = 0;
        _paletteSize = 0;
        _transparencySize = 0;
        _delayNum = DEFAULT_FRAME_NUMERATOR;
        _delayDen = DEFAULT_FRAME_DENOMINATOR;
        _rows = NULL;
    }

    APNGFrame::~APNGFrame() {}

    bool APNGFrame::setMat(const cv::Mat& src, unsigned delayNum, unsigned delayDen)
    {
        _delayNum = delayNum;
        _delayDen = delayDen;

        if (!src.empty())
        {
            png_uint_32 rowbytes = src.cols * src.channels();

            _width = src.cols;
            _height = src.rows;
            _colorType = src.channels() == 1 ? PNG_COLOR_TYPE_GRAY : src.channels() == 3 ? PNG_COLOR_TYPE_RGB : PNG_COLOR_TYPE_RGB_ALPHA;
            _pixels = src.data;
            _rows = new png_bytep[_height * sizeof(png_bytep)];

            for (unsigned int i = 0; i < _height; ++i)
                _rows[i] = _pixels + i * rowbytes;
            return true;
        }
        return false;
    }

    void APNGFrame::setWidth(unsigned int setWidth) {
        _width = setWidth;
    }

    void APNGFrame::setHeight(unsigned int setHeight) {
        _height = setHeight;
    }

    void APNGFrame::setColorType(unsigned char setColorType) {
        _colorType = setColorType;
    }

    void APNGFrame::setPalette(const rgb* setPalette) {
        std::copy(setPalette, setPalette + 256, _palette);
    }

    void APNGFrame::setTransparency(const unsigned char* setTransparency) {
        std::copy(setTransparency, setTransparency + 256, _transparency);
    }

    void APNGFrame::setPaletteSize(int setPaletteSize) {
        _paletteSize = setPaletteSize;
    }

    void APNGFrame::setTransparencySize(int setTransparencySize) {
        _transparencySize = setTransparencySize;
    }

    void APNGFrame::setDelayNum(unsigned int setDelayNum) {
        _delayNum = setDelayNum;
    }

    void APNGFrame::setDelayDen(unsigned int setDelayDen) {
        _delayDen = setDelayDen;
    }

    void APNGFrame::setPixels(unsigned char* setPixels) {
        _pixels = setPixels;
    }

    void APNGFrame::setRows(unsigned char** setRows) {
        _rows = setRows;
    }

    // Save frame to a PNG image file.
    // Return true if save succeeded.
    bool APNGFrame::save(const std::string& outPath) const
    {
        FILE* f;
        if ((f = fopen(outPath.c_str(), "wb")) != 0) {
            png_structp png_ptr =
                png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
            png_infop info_ptr = png_create_info_struct(png_ptr);
            if (png_ptr && info_ptr) {
                if (setjmp(png_jmpbuf(png_ptr))) {
                    png_destroy_read_struct(&png_ptr, &info_ptr, 0);
                    fclose(f);
                    return false;
                }
                png_init_io(png_ptr, f);
                png_set_compression_level(png_ptr, 9);
                png_set_IHDR(png_ptr, info_ptr, _width, _height, 8, _colorType, 0, 0, 0);
                if (_paletteSize > 0) {
                    png_color palette[PNG_MAX_PALETTE_LENGTH];
                    memcpy(palette, _palette, _paletteSize * 3);
                    png_set_PLTE(png_ptr, info_ptr, palette, _paletteSize);
                }
                if (_transparencySize > 0) {
                    png_color_16 trans_color;
                    if (_colorType == PNG_COLOR_TYPE_GRAY) {
                        trans_color.gray = _transparency[1];
                        png_set_tRNS(png_ptr, info_ptr, NULL, 0, &trans_color);
                    }
                    else if (_colorType == PNG_COLOR_TYPE_RGB) {
                        trans_color.red = _transparency[1];
                        trans_color.green = _transparency[3];
                        trans_color.blue = _transparency[5];
                        png_set_tRNS(png_ptr, info_ptr, NULL, 0, &trans_color);
                    }
                    else if (_colorType == PNG_COLOR_TYPE_PALETTE)
                        png_set_tRNS(png_ptr, info_ptr,
                            const_cast<unsigned char*>(_transparency),
                            _transparencySize, NULL);
                }
                png_write_info(png_ptr, info_ptr);
                png_write_image(png_ptr, _rows);
                png_write_end(png_ptr, info_ptr);
                return true;
            }
            else
                png_destroy_write_struct(&png_ptr, &info_ptr);
            fclose(f);
        }
        return false;
    }

} // namespace cv
