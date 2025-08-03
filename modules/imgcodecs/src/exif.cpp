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

#include "precomp.hpp"
#include "exif.hpp"
#include "opencv2/core/utils/logger.hpp"

namespace {

    class ExifParsingError {
    };
}


namespace cv
{

static std::string HexStringToBytes(const char* hexstring, size_t expected_length);

// Converts the NULL terminated 'hexstring' which contains 2-byte character
// representations of hex values to raw data.
// 'hexstring' may contain values consisting of [A-F][a-f][0-9] in pairs,
// e.g., 7af2..., separated by any number of newlines.
// 'expected_length' is the anticipated processed size.
// On success the raw buffer is returned with its length equivalent to
// 'expected_length'. NULL is returned if the processed length is less than
// 'expected_length' or any character aside from those above is encountered.
// The returned buffer must be freed by the caller.
static std::string HexStringToBytes(const char* hexstring,
    size_t expected_length) {
    const char* src = hexstring;
    size_t actual_length = 0;
    std::string raw_data;
    raw_data.resize(expected_length);
    char* dst = const_cast<char*>(raw_data.data());

    for (; actual_length < expected_length && *src != '\0'; ++src) {
        char* end;
        char val[3];
        if (*src == '\n') continue;
        val[0] = *src++;
        val[1] = *src;
        val[2] = '\0';
        *dst++ = static_cast<uint8_t>(strtol(val, &end, 16));
        if (end != val + 2) break;
        ++actual_length;
    }

    if (actual_length != expected_length) {
        raw_data.clear();
    }
    return raw_data;
}

ExifEntry_t::ExifEntry_t() :
    field_float(0), field_double(0), field_u32(0), field_s32(0),
    tag(INVALID_TAG), field_u16(0), field_s16(0), field_u8(0), field_s8(0)
{
}

/**
 * @brief ExifReader constructor
 */
ExifReader::ExifReader() : m_format(NONE)
{
}

/**
 * @brief ExifReader destructor
 */
ExifReader::~ExifReader()
{
}


/**
 *  @brief Get tag value by tag number
 *
 *  @param [in] tag The tag number
 *
 *  @return ExifEntru_t structure. Caller has to know what tag it calls in order to extract proper field from the structure ExifEntry_t
 *
 */
ExifEntry_t ExifReader::getTag(const ExifTagName tag) const
{
    ExifEntry_t entry;
    std::map<int, ExifEntry_t>::const_iterator it = m_exif.find(tag);

    if( it != m_exif.end() )
    {
        entry = it->second;
    }
    return entry;
}

const std::vector<unsigned char>& ExifReader::getData() const
{
    return m_data;
}

bool ExifReader::processRawProfile(const char* profile, size_t profile_len) {
    const char* src = profile;
    char* end;
    int expected_length;

    if (profile == nullptr || profile_len == 0) return false;

    // ImageMagick formats 'raw profiles' as
    // '\n<name>\n<length>(%8lu)\n<hex payload>\n'.
    if (*src != '\n') {
        CV_LOG_WARNING(NULL, cv::format("Malformed raw profile, expected '\\n' got '\\x%.2X'", *src));
        return false;
    }
    ++src;
    // skip the profile name and extract the length.
    while (*src != '\0' && *src++ != '\n') {}
    expected_length = static_cast<int>(strtol(src, &end, 10));
    if (*end != '\n') {
        CV_LOG_WARNING(NULL, cv::format("Malformed raw profile, expected '\\n' got '\\x%.2X'", *src));
        return false;
    }
    ++end;

    // 'end' now points to the profile payload.
    std::string payload = HexStringToBytes(end, expected_length);
    if (payload.size() == 0) return false;

    return parseExif((unsigned char*)payload.c_str() + 6, expected_length - 6);
}

/**
 * @brief Parsing the exif data buffer and prepare (internal) exif directory
 *
 * @param [in] data The data buffer to read EXIF data starting with endianness
 * @param [in] size The size of the data buffer
 *
 * @return  true if parsing was successful
 *          false in case of unsuccessful parsing
 */
bool ExifReader::parseExif(unsigned char* data, const size_t size)
{
    // Populate m_data, then call parseExif() (private)
    if( data && size > 0 )
    {
        m_data.assign(data, data + size);
    }
    else
    {
        return false;
    }

    try {
        parseExif();
        if( !m_exif.empty() )
        {
            return true;
        }
        return false;
    }
    catch( ExifParsingError& ) {
        return false;
    }
}

/**
 * @brief Filling m_exif member with exif directory elements
 *          This is internal function and is not exposed to client
 *
 *  The function doesn't return any value. In case of unsuccessful parsing
 *      the m_exif member is not filled up
 */
void ExifReader::parseExif()
{
    m_format = getFormat();

    if( !checkTagMark() )
    {
        return;
    }

    uint32_t offset = getStartOffset();

    size_t numEntry = getNumDirEntry( offset );

    offset += 2; //go to start of tag fields

    for( size_t entry = 0; entry < numEntry; entry++ )
    {
        ExifEntry_t exifEntry = parseExifEntry( offset );
        m_exif.insert( std::make_pair( exifEntry.tag, exifEntry ) );
        offset += tiffFieldSize;
    }
}

/**
 * @brief Get endianness of exif information
 *          This is internal function and is not exposed to client
 *
 * @return INTEL, MOTO or NONE
 */
Endianness_t ExifReader::getFormat() const
{
    if (m_data.size() < 1)
        return NONE;

    if( m_data.size() > 1 && m_data[0] != m_data[1] )
    {
        return NONE;
    }

    if( m_data[0] == 'I' )
    {
        return INTEL;
    }

    if( m_data[0] == 'M' )
    {
        return MOTO;
    }

    return NONE;
}

/**
 * @brief Checking whether Tag Mark (0x002A) correspond to one contained in the Jpeg file
 *          This is internal function and is not exposed to client
 *
 * @return true if tag mark equals 0x002A, false otherwise
 */
bool ExifReader::checkTagMark() const
{
    uint16_t tagMark = getU16( 2 );

    if( tagMark != tagMarkRequired )
    {
        return false;
    }
    return true;
}

/**
 * @brief The utility function for extracting actual offset exif IFD0 info is started from
 *          This is internal function and is not exposed to client
 *
 * @return offset of IFD0 field
 */
uint32_t ExifReader::getStartOffset() const
{
    return getU32( 4 );
}

/**
 * @brief Get the number of Directory Entries in Jpeg file
 *
 * @return The number of directory entries
 */
size_t ExifReader::getNumDirEntry(const size_t offsetNumDir) const
{
    return getU16( offsetNumDir );
}

/**
 * @brief Parsing particular entry in exif directory
 *          This is internal function and is not exposed to client
 *
 *      Entries are divided into 12-bytes blocks each
 *      Each block corresponds the following structure:
 *
 *      +------+-------------+-------------------+------------------------+
 *      | Type | Data format | Num of components | Data or offset to data |
 *      +======+=============+===================+========================+
 *      | TTTT | ffff        | NNNNNNNN          | DDDDDDDD               |
 *      +------+-------------+-------------------+------------------------+
 *
 *      Details can be found here: http://www.media.mit.edu/pia/Research/deepview/exif.html
 *
 * @param [in] offset Offset to entry in bytes inside raw exif data
 * @return ExifEntry_t structure which corresponds to particular entry
 *
 */
ExifEntry_t ExifReader::parseExifEntry(const size_t offset)
{
    ExifEntry_t entry;
    uint16_t tagNum = getExifTag( offset );
    entry.tag = tagNum;

    switch( tagNum )
    {
        case IMAGE_DESCRIPTION:
            entry.field_str = getString( offset );
            break;
        case MAKE:
            entry.field_str = getString( offset );
            break;
        case MODEL:
            entry.field_str = getString( offset );
            break;
        case ORIENTATION:
            entry.field_u16 = getOrientation( offset );
            break;
        case XRESOLUTION:
            entry.field_u_rational = getResolution( offset );
            break;
        case YRESOLUTION:
            entry.field_u_rational = getResolution( offset );
            break;
        case RESOLUTION_UNIT:
            entry.field_u16 = getResolutionUnit( offset );
            break;
        case SOFTWARE:
            entry.field_str = getString( offset );
            break;
        case DATE_TIME:
            entry.field_str = getString( offset );
            break;
        case WHITE_POINT:
            entry.field_u_rational = getWhitePoint( offset );
            break;
        case PRIMARY_CHROMATICIES:
            entry.field_u_rational = getPrimaryChromaticies( offset );
            break;
        case Y_CB_CR_COEFFICIENTS:
            entry.field_u_rational = getYCbCrCoeffs( offset );
            break;
        case Y_CB_CR_POSITIONING:
            entry.field_u16 = getYCbCrPos( offset );
            break;
        case REFERENCE_BLACK_WHITE:
            entry.field_u_rational = getRefBW( offset );
            break;
        case COPYRIGHT:
            entry.field_str = getString( offset );
            break;
        case EXIF_OFFSET:
            break;
        default:
            entry.tag = INVALID_TAG;
            break;
    }
    return entry;
}

/**
 * @brief Get tag number from raw exif data
 *          This is internal function and is not exposed to client
 * @param [in] offset Offset to entry in bytes inside raw exif data
 * @return tag number
 */
uint16_t ExifReader::getExifTag(const size_t offset) const
{
    return getU16( offset );
}

/**
 * @brief Get string information from raw exif data
 *          This is internal function and is not exposed to client
 * @param [in] offset Offset to entry in bytes inside raw exif data
 * @return string value
 */
std::string ExifReader::getString(const size_t offset) const
{
    size_t size = getU32( offset + 4 );
    size_t dataOffset = 8; // position of data in the field
    if( size > maxDataSize )
    {
        dataOffset = getU32( offset + 8 );
    }
    if (dataOffset > m_data.size() || dataOffset + size > m_data.size()) {
        throw ExifParsingError();
    }
    std::vector<uint8_t>::const_iterator it = m_data.begin() + dataOffset;
    std::string result( it, it + size ); //copy vector content into result

    return result;
}

/**
 * @brief Get unsigned short data from raw exif data
 *          This is internal function and is not exposed to client
 * @param [in] offset Offset to entry in bytes inside raw exif data
 * @return Unsigned short data
 */
uint16_t ExifReader::getU16(const size_t offset) const
{
    if (offset + 1 >= m_data.size())
        throw ExifParsingError();

    if( m_format == INTEL )
    {
        return m_data[offset] + ( m_data[offset + 1] << 8 );
    }
    return ( m_data[offset] << 8 ) + m_data[offset + 1];
}

/**
 * @brief Get unsigned 32-bit data from raw exif data
 *          This is internal function and is not exposed to client
 * @param [in] offset Offset to entry in bytes inside raw exif data
 * @return Unsigned 32-bit data
 */
uint32_t ExifReader::getU32(const size_t offset) const
{
    if (offset + 3 >= m_data.size())
        throw ExifParsingError();

    if( m_format == INTEL )
    {
        return m_data[offset] +
                ( m_data[offset + 1] << 8 ) +
                ( m_data[offset + 2] << 16 ) +
                ( m_data[offset + 3] << 24 );
    }

    return ( m_data[offset] << 24 ) +
            ( m_data[offset + 1] << 16 ) +
            ( m_data[offset + 2] << 8 ) +
            m_data[offset + 3];
}

/**
 * @brief Get unsigned rational data from raw exif data
 *          This is internal function and is not exposed to client
 * @param [in] offset Offset to entry in bytes inside raw exif data
 * @return Unsigned rational data
 *
 * "rational" means a fractional value, it contains 2 signed/unsigned long integer value,
 *  and the first represents the numerator, the second, the denominator.
 */
u_rational_t ExifReader::getURational(const size_t offset) const
{
    uint32_t numerator = getU32( offset );
    uint32_t denominator = getU32( offset + 4 );

    return std::make_pair( numerator, denominator );

}

/**
 * @brief Get orientation information from raw exif data
 *          This is internal function and is not exposed to client
 * @param [in] offset Offset to entry in bytes inside raw exif data
 * @return orientation number
 */
uint16_t ExifReader::getOrientation(const size_t offset) const
{
    return getU16( offset + 8 );
}

/**
 * @brief Get resolution information from raw exif data
 *          This is internal function and is not exposed to client
 * @param [in] offset Offset to entry in bytes inside raw exif data
 * @return resolution value
 */
std::vector<u_rational_t> ExifReader::getResolution(const size_t offset) const
{
    std::vector<u_rational_t> result;
    uint32_t rationalOffset = getU32( offset + 8 );
    result.push_back( getURational( rationalOffset ) );

    return result;
}

/**
 * @brief Get resolution unit from raw exif data
 *          This is internal function and is not exposed to client
 * @param [in] offset Offset to entry in bytes inside raw exif data
 * @return resolution unit value
 */
uint16_t ExifReader::getResolutionUnit(const size_t offset) const
{
    return getU16( offset + 8 );
}

/**
 * @brief Get White Point information from raw exif data
 *          This is internal function and is not exposed to client
 * @param [in] offset Offset to entry in bytes inside raw exif data
 * @return White Point value
 *
 * If the image uses CIE Standard Illumination D65(known as international
 * standard of 'daylight'), the values are '3127/10000,3290/10000'.
 */
std::vector<u_rational_t> ExifReader::getWhitePoint(const size_t offset) const
{
    std::vector<u_rational_t> result;
    uint32_t rationalOffset = getU32( offset + 8 );
    result.push_back( getURational( rationalOffset ) );
    result.push_back( getURational( rationalOffset + 8 ) );

    return result;
}

/**
 * @brief Get Primary Chromaticies information from raw exif data
 *          This is internal function and is not exposed to client
 * @param [in] offset Offset to entry in bytes inside raw exif data
 * @return vector with primary chromaticies values
 *
 */
std::vector<u_rational_t> ExifReader::getPrimaryChromaticies(const size_t offset) const
{
    std::vector<u_rational_t> result;
    uint32_t rationalOffset = getU32( offset + 8 );
    for( size_t i = 0; i < primaryChromaticiesComponents; i++ )
    {
        result.push_back( getURational( rationalOffset ) );
        rationalOffset += 8;
    }
    return result;
}

/**
 * @brief Get YCbCr Coefficients information from raw exif data
 *          This is internal function and is not exposed to client
 * @param [in] offset Offset to entry in bytes inside raw exif data
 * @return vector with YCbCr coefficients values
 *
 */
std::vector<u_rational_t> ExifReader::getYCbCrCoeffs(const size_t offset) const
{
    std::vector<u_rational_t> result;
    uint32_t rationalOffset = getU32( offset + 8 );
    for( size_t i = 0; i < ycbcrCoeffs; i++ )
    {
        result.push_back( getURational( rationalOffset ) );
        rationalOffset += 8;
    }
    return result;
}

/**
 * @brief Get YCbCr Positioning information from raw exif data
 *          This is internal function and is not exposed to client
 * @param [in] offset Offset to entry in bytes inside raw exif data
 * @return vector with YCbCr positioning value
 *
 */
uint16_t ExifReader::getYCbCrPos(const size_t offset) const
{
    return getU16( offset + 8 );
}

/**
 * @brief Get Reference Black&White point information from raw exif data
 *          This is internal function and is not exposed to client
 * @param [in] offset Offset to entry in bytes inside raw exif data
 * @return vector with reference BW points
 *
 * In case of YCbCr format, first 2 show black/white of Y, next 2 are Cb,
 * last 2 are Cr. In case of RGB format, first 2 show black/white of R,
 * next 2 are G, last 2 are B.
 *
 */
std::vector<u_rational_t> ExifReader::getRefBW(const size_t offset) const
{
    const size_t rationalFieldSize = 8;
    std::vector<u_rational_t> result;
    uint32_t rationalOffset = getU32( offset + rationalFieldSize );
    for( size_t i = 0; i < refBWComponents; i++ )
    {
        result.push_back( getURational( rationalOffset ) );
        rationalOffset += rationalFieldSize;
    }
    return result;
}

} //namespace cv
