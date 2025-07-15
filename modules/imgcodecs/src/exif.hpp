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


#ifndef _OPENCV_EXIF_HPP_
#define _OPENCV_EXIF_HPP_

#include <cstdio>
#include <map>
#include <utility>
#include <algorithm>
#include <string>
#include <vector>
#include <iostream>

namespace cv
{

/**
 * @brief Base Exif tags used by IFD0 (main image)
 */
enum ExifTagName
{
    IMAGE_DESCRIPTION       = 0x010E,   ///< Image Description: ASCII string
    MAKE                    = 0x010F,   ///< Description of manufacturer: ASCII string
    MODEL                   = 0x0110,   ///< Description of camera model: ASCII string
    ORIENTATION             = 0x0112,   ///< Orientation of the image: unsigned short
    XRESOLUTION             = 0x011A,   ///< Resolution of the image across X axis: unsigned rational
    YRESOLUTION             = 0x011B,   ///< Resolution of the image across Y axis: unsigned rational
    RESOLUTION_UNIT         = 0x0128,   ///< Resolution units. '1' no-unit, '2' inch, '3' centimeter
    SOFTWARE                = 0x0131,   ///< Shows firmware(internal software of digicam) version number
    DATE_TIME               = 0x0132,   ///< Date/Time of image was last modified
    WHITE_POINT             = 0x013E,   ///< Chromaticity of white point of the image
    PRIMARY_CHROMATICIES    = 0x013F,   ///< Chromaticity of the primaries of the image
    Y_CB_CR_COEFFICIENTS    = 0x0211,   ///< constant to translate an image from YCbCr to RGB format
    Y_CB_CR_POSITIONING     = 0x0213,   ///< Chroma sample point of subsampling pixel array
    REFERENCE_BLACK_WHITE   = 0x0214,   ///< Reference value of black point/white point
    COPYRIGHT               = 0x8298,   ///< Copyright information
    EXIF_OFFSET             = 0x8769,   ///< Offset to Exif Sub IFD
    INVALID_TAG             = 0xFFFF    ///< Shows that the tag was not recognized
};

enum Endianness_t
{
    INTEL = 0x49,
    MOTO = 0x4D,
    NONE = 0x00
};

typedef std::pair<uint32_t, uint32_t> u_rational_t;

/**
 * @brief Entry which contains possible values for different exif tags
 */
struct ExifEntry_t
{
    ExifEntry_t();

    std::vector<u_rational_t> field_u_rational; ///< vector of rational fields
    std::string field_str;                      ///< any kind of textual information

    float  field_float;                         ///< Currently is not used
    double field_double;                        ///< Currently is not used

    uint32_t field_u32;                         ///< Unsigned 32-bit value
    int32_t  field_s32;                         ///< Signed 32-bit value

    uint16_t tag;                               ///< Tag number

    uint16_t field_u16;                         ///< Unsigned 16-bit value
    int16_t  field_s16;                         ///< Signed 16-bit value
    uint8_t  field_u8;                          ///< Unsigned 8-bit value
    int8_t   field_s8;                          ///< Signed 8-bit value
};

/**
 * @brief Picture orientation which may be taken from EXIF
 *      Orientation usually matters when the picture is taken by
 *      smartphone or other camera with orientation sensor support
 *      Corresponds to EXIF 2.3 Specification
 */
enum ImageOrientation
{
    IMAGE_ORIENTATION_TL = 1, ///< Horizontal (normal)
    IMAGE_ORIENTATION_TR = 2, ///< Mirrored horizontal
    IMAGE_ORIENTATION_BR = 3, ///< Rotate 180
    IMAGE_ORIENTATION_BL = 4, ///< Mirrored vertical
    IMAGE_ORIENTATION_LT = 5, ///< Mirrored horizontal & rotate 270 CW
    IMAGE_ORIENTATION_RT = 6, ///< Rotate 90 CW
    IMAGE_ORIENTATION_RB = 7, ///< Mirrored horizontal & rotate 90 CW
    IMAGE_ORIENTATION_LB = 8  ///< Rotate 270 CW
};

/**
 * @brief Reading exif information from Jpeg file
 *
 * Usage example for getting the orientation of the image:
 *
 *      @code
 *      std::ifstream stream(filename,std::ios_base::in | std::ios_base::binary);
 *      ExifReader reader(stream);
 *      if( reader.parse() )
 *      {
 *          int orientation = reader.getTag(Orientation).field_u16;
 *      }
 *      @endcode
 *
 */
class ExifReader
{
public:
    /**
     * @brief ExifReader constructor. Constructs an object of exif reader
     */
    ExifReader();
    ~ExifReader();

    bool processRawProfile(const char* profile, size_t profile_len);

    /**
     * @brief Parse the file with exif info
     *
     * @param [in] data The data buffer to read EXIF data starting with endianness
     * @param [in] size The size of the data buffer
     *
     * @return true if successful parsing
     *         false if parsing error
     */

    bool parseExif(unsigned char* data, const size_t size);

    /**
     * @brief Get tag info by tag number
     *
     * @param [in] tag The tag number
     * @return ExifEntru_t structure. Caller has to know what tag it calls in order to extract proper field from the structure ExifEntry_t
     */
    ExifEntry_t getTag( const ExifTagName tag ) const;

    /**
     * @brief Get the whole exif buffer
     */
    const std::vector<unsigned char>& getData() const;

private:
    std::vector<unsigned char> m_data;
    std::map<int, ExifEntry_t > m_exif;
    Endianness_t m_format;

    void parseExif();
    bool checkTagMark() const;

    size_t getNumDirEntry( const size_t offsetNumDir ) const;
    uint32_t getStartOffset() const;
    uint16_t getExifTag( const size_t offset ) const;
    uint16_t getU16( const size_t offset ) const;
    uint32_t getU32( const size_t offset ) const;
    uint16_t getOrientation( const size_t offset ) const;
    uint16_t getResolutionUnit( const size_t offset ) const;
    uint16_t getYCbCrPos( const size_t offset ) const;

    Endianness_t getFormat() const;

    ExifEntry_t parseExifEntry( const size_t offset );

    u_rational_t getURational( const size_t offset ) const;

    std::string getString( const size_t offset ) const;
    std::vector<u_rational_t> getResolution( const size_t offset ) const;
    std::vector<u_rational_t> getWhitePoint( const size_t offset ) const;
    std::vector<u_rational_t> getPrimaryChromaticies( const size_t offset ) const;
    std::vector<u_rational_t> getYCbCrCoeffs( const size_t offset ) const;
    std::vector<u_rational_t> getRefBW( const size_t offset ) const;

private:
    static const uint16_t tagMarkRequired = 0x2A;

    //max size of data in tag.
    //'DDDDDDDD' contains the value of that Tag. If its size is over 4bytes,
    //'DDDDDDDD' contains the offset to data stored address.
    static const size_t maxDataSize = 4;

    //bytes per tag field
    static const size_t tiffFieldSize = 12;

    //number of primary chromaticies components
    static const size_t primaryChromaticiesComponents = 6;

    //number of YCbCr coefficients in field
    static const size_t ycbcrCoeffs = 3;

    //number of Reference Black&White components
    static const size_t refBWComponents = 6;
};


}

#endif /* _OPENCV_EXIF_HPP_ */
