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

namespace {

    class ExifParsingError {
    };
}


namespace cv
{

ExifEntry_t::ExifEntry_t() :
    field_float(0), field_double(0), field_u32(0), field_s32(0),
    tag(INVALID_TAG), field_u16(0), field_s16(0), field_u8(0), field_s8(0)
{
}

/**
 * @brief ExifReader constructor
 */
ExifReader::ExifReader(std::istream& stream) : m_stream(stream), m_format(NONE)
{
}

/**
 * @brief ExifReader destructor
 */
ExifReader::~ExifReader()
{
}

/**
 * @brief Parsing the file and prepare (internally) exif directory structure
 * @return  true if parsing was successful and exif information exists in JpegReader object
 *          false in case of unsuccessful parsing
 */
bool ExifReader::parse()
{
    try {
        m_exif = getExif();
        if( !m_exif.empty() )
        {
            return true;
        }
        return false;
    } catch (ExifParsingError&) {
        return false;
    }
}


/**
 *  @brief Get tag value by tag number
 *
 *  @param [in] tag The tag number
 *
 *  @return ExifEntru_t structure. Caller has to know what tag it calls in order to extract proper field from the structure ExifEntry_t
 *
 */
ExifEntry_t ExifReader::getTag(const ExifTagName tag)
{
    ExifEntry_t entry;
    std::map<int, ExifEntry_t>::iterator it = m_exif.find(tag);

    if( it != m_exif.end() )
    {
        entry = it->second;
    }
    return entry;
}


/**
 * @brief Get exif directory structure contained in file (if any)
 *          This is internal function and is not exposed to client
 *
 *  @return Map where key is tag number and value is ExifEntry_t structure
 */
std::map<int, ExifEntry_t > ExifReader::getExif()
{
    const std::streamsize markerSize = 2;
    const std::streamsize offsetToTiffHeader = 6; //bytes from Exif size field to the first TIFF header
    unsigned char appMarker[markerSize];
    m_exif.erase( m_exif.begin(), m_exif.end() );

    std::streamsize count;

    bool exifFound = false, stopSearch = false;
    while( ( !m_stream.eof() ) && !exifFound && !stopSearch )
    {
        m_stream.read( reinterpret_cast<char*>(appMarker), markerSize );
        count = m_stream.gcount();
        if( count < markerSize )
        {
            break;
        }
        unsigned char marker = appMarker[1];
        size_t bytesToSkip;
        size_t exifSize;
        switch( marker )
        {
            //For all the markers just skip bytes in file pointed by followed two bytes (field size)
            case SOF0: case SOF2: case DHT: case DQT: case DRI: case SOS:
            case RST0: case RST1: case RST2: case RST3: case RST4: case RST5: case RST6: case RST7:
            case APP0: case APP2: case APP3: case APP4: case APP5: case APP6: case APP7: case APP8:
            case APP9: case APP10: case APP11: case APP12: case APP13: case APP14: case APP15:
            case COM:
                bytesToSkip = getFieldSize();
                if (bytesToSkip < markerSize) {
                    throw ExifParsingError();
                }
                m_stream.seekg( static_cast<long>( bytesToSkip - markerSize ), m_stream.cur );
                if ( m_stream.fail() ) {
                    throw ExifParsingError();
                }
                break;

            //SOI and EOI don't have the size field after the marker
            case SOI: case EOI:
                break;

            case APP1: //actual Exif Marker
                exifSize = getFieldSize();
                if (exifSize <= offsetToTiffHeader) {
                    throw ExifParsingError();
                }
                m_data.resize( exifSize - offsetToTiffHeader );
                m_stream.seekg( static_cast<long>( offsetToTiffHeader ), m_stream.cur );
                if ( m_stream.fail() ) {
                    throw ExifParsingError();
                }
                m_stream.read( reinterpret_cast<char*>(&m_data[0]), exifSize - offsetToTiffHeader );
                exifFound = true;
                break;

            default: //No other markers are expected according to standard. May be a signal of error
                stopSearch = true;
                break;
        }
    }

    if( !exifFound )
    {
        return m_exif;
    }

    parseExif();

    return m_exif;
}

/**
 * @brief Get the size of exif field (required to properly ready whole exif from the file)
 *          This is internal function and is not exposed to client
 *
 *  @return size of exif field in the file
 */
size_t ExifReader::getFieldSize ()
{
    unsigned char fieldSize[2];
    m_stream.read( reinterpret_cast<char*>(fieldSize), 2 );
    std::streamsize count = m_stream.gcount();
    if (count < 2)
    {
        return 0;
    }
    return ( fieldSize[0] << 8 ) + fieldSize[1];
}

/**
 * @brief Filling m_exif member with exif directory elements
 *          This is internal function and is not exposed to client
 *
 *  @return The function doesn't return any value. In case of unsuccessful parsing
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
Endianess_t ExifReader::getFormat() const
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
