// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

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

#ifndef _APNGFRAME_H_
#define _APNGFRAME_H_

#include "precomp.hpp"
#include <string>

namespace cv {

typedef struct {
  unsigned char r, g, b;
} rgb;
typedef struct {
  unsigned char r, g, b, a;
} rgba;

class APNGFrame {
public:

    APNGFrame();

    // Destructor
    ~APNGFrame();

    bool setMat(const cv::Mat& src, unsigned delayNum = 1, unsigned delayDen = 1000);

    // Getters and Setters
    unsigned char* getPixels() const { return _pixels; }
    void setPixels(unsigned char* setPixels);

    unsigned int getWidth() const { return _width; }
    void setWidth(unsigned int setWidth);

    unsigned int getHeight() const { return _height; }
    void setHeight(unsigned int setHeight);

    unsigned char getColorType() const { return _colorType; }
    void setColorType(unsigned char setColorType);

    rgb* getPalette() { return _palette; }
    void setPalette(const rgb* setPalette);

    unsigned char* getTransparency() { return _transparency; }
    void setTransparency(const unsigned char* setTransparency);

    int getPaletteSize() const { return _paletteSize; }
    void setPaletteSize(int setPaletteSize);

    int getTransparencySize() const { return _transparencySize; }
    void setTransparencySize(int setTransparencySize);

    unsigned int getDelayNum() const { return _delayNum; }
    void setDelayNum(unsigned int setDelayNum);

    unsigned int getDelayDen() const { return _delayDen; }
    void setDelayDen(unsigned int setDelayDen);

    unsigned char** getRows() const { return _rows; }
    void setRows(unsigned char** setRows);

private:
    unsigned char* _pixels;
    unsigned int _width;
    unsigned int _height;
    unsigned char _colorType;
    rgb _palette[256];
    unsigned char _transparency[256];
    int _paletteSize;
    int _transparencySize;
    unsigned int _delayNum;
    unsigned int _delayDen;
    unsigned char** _rows;
};

} // namespace apngasm

#endif /* _APNGFRAME_H_ */
