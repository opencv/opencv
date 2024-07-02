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

#ifndef _APNGFRAME_H_
#define _APNGFRAME_H_

#include <string>

namespace cv {

const unsigned DEFAULT_FRAME_NUMERATOR =
    100; //!< @brief The default numerator for the frame delay fraction.
const unsigned DEFAULT_FRAME_DENOMINATOR =
    1000; //!< @brief The default denominator for the frame delay fraction.

typedef struct {
  unsigned char r, g, b;
} rgb;
typedef struct {
  unsigned char r, g, b, a;
} rgba;

// Individual APNG frame
class APNGFrame {
public:
  // Raw pixel data
  unsigned char *pixels(unsigned char *setPixels = NULL);
  unsigned char *_pixels;

  // Width and Height
  unsigned int width(unsigned int setWidth = 0);
  unsigned int height(unsigned int setHeight = 0);
  unsigned int _width;
  unsigned int _height;

  // PNG color type
  unsigned char colorType(unsigned char setColorType = 255);
  unsigned char _colorType;

  // Palette into
  rgb *palette(rgb *setPalette = NULL);
  rgb _palette[256];

  // Transparency info
  unsigned char *transparency(unsigned char *setTransparency = NULL);
  unsigned char _transparency[256];

  // Sizes for palette and transparency records
  int paletteSize(int setPaletteSize = 0);
  int _paletteSize;

  int transparencySize(int setTransparencySize = 0);
  int _transparencySize;

  // Delay is numerator/denominator ratio, in seconds
  unsigned int delayNum(unsigned int setDelayNum = 0);
  unsigned int _delayNum;

  unsigned int delayDen(unsigned int setDelayDen = 0);
  unsigned int _delayDen;

  unsigned char **rows(unsigned char **setRows = NULL);
  unsigned char **_rows;

  /**
   * @brief Creates an empty APNGFrame.
   */
  APNGFrame();

  /**
   * @brief Creates an APNGFrame from a PNG file.
   * @param filePath The relative or absolute path to an image file.
   * @param delayNum The delay numerator for this frame (defaults to
   * DEFAULT_FRAME_NUMERATOR).
   * @param delayDen The delay denominator for this frame (defaults to
   * DEFAULT_FRAME_DENMINATOR).
   */
  APNGFrame(const std::string &filePath,
            unsigned delayNum = DEFAULT_FRAME_NUMERATOR,
            unsigned delayDen = DEFAULT_FRAME_DENOMINATOR);

  /**
   * @brief Creates an APNGFrame from a bitmapped array of RBG pixel data.
   * @param pixels The RGB pixel data.
   * @param width The width of the pixel data.
   * @param height The height of the pixel data.
   * @param delayNum The delay numerator for this frame (defaults to
   * DEFAULT_FRAME_NUMERATOR).
   * @param delayDen The delay denominator for this frame (defaults to
   * DEFAULT_FRAME_DENMINATOR).
   */
  APNGFrame(rgb *pixels, unsigned int width, unsigned int height,
            unsigned delayNum = DEFAULT_FRAME_NUMERATOR,
            unsigned delayDen = DEFAULT_FRAME_DENOMINATOR);

  /**
   * @brief Creates an APNGFrame from a bitmapped array of RBG pixel data.
   * @param pixels The RGB pixel data.
   * @param width The width of the pixel data.
   * @param height The height of the pixel data.
   * @param trns_color An array of transparency data.
   * @param delayNum The delay numerator for this frame (defaults to
   * DEFAULT_FRAME_NUMERATOR).
   * @param delayDen The delay denominator for this frame (defaults to
   * DEFAULT_FRAME_DENMINATOR).
   */
  APNGFrame(rgb *pixels, unsigned int width, unsigned int height,
            rgb *trns_color = NULL, unsigned delayNum = DEFAULT_FRAME_NUMERATOR,
            unsigned delayDen = DEFAULT_FRAME_DENOMINATOR);

  /**
   * @brief Creates an APNGFrame from a bitmapped array of RBGA pixel data.
   * @param pixels The RGBA pixel data.
   * @param width The width of the pixel data.
   * @param height The height of the pixel data.
   * @param delayNum The delay numerator for this frame (defaults to
   * DEFAULT_FRAME_NUMERATOR).
   * @param delayDen The delay denominator for this frame (defaults to
   * DEFAULT_FRAME_DENMINATOR).
   */
  APNGFrame(rgba *pixels, unsigned int width, unsigned int height,
            unsigned delayNum = DEFAULT_FRAME_NUMERATOR,
            unsigned delayDen = DEFAULT_FRAME_DENOMINATOR);

  /**
   * @brief Saves this frame as a single PNG file.
   * @param outPath The relative or absolute path to save the image file to.
   * @return Returns true if save was successful.
   */
  bool save(const std::string &outPath) const;

private:
}; // class APNGFrame

} // namespace apngasm

#endif /* _APNGFRAME_H_ */
