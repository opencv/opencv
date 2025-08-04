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
#include <iomanip>

namespace {

    class ExifParsingError {
    };
}


namespace cv
{

static uint32_t getExifTagTypeSize(uint16_t type)
{
    return
        type == TAG_TYPE_NOTYPE ? 0 :
        type == TAG_TYPE_BYTE ? 1 :
        type == TAG_TYPE_ASCII ? 1 :
        type == TAG_TYPE_SHORT ? 2 :
        type == TAG_TYPE_LONG ? 4 :
        type == TAG_TYPE_RATIONAL ? 8 :
        type == TAG_TYPE_SBYTE ? 1 :
        type == TAG_TYPE_UNDEFINED ? 1 :
        type == TAG_TYPE_SSHORT ? 2 :
        type == TAG_TYPE_SLONG ? 4 :
        type == TAG_TYPE_SRATIONAL ? 8 :
        type == TAG_TYPE_FLOAT ? 4 :
        type == TAG_TYPE_DOUBLE ? 8 :
        type == TAG_TYPE_IFD ? 0 :
        type == TAG_TYPE_LONG8 ? 8 :
        type == TAG_TYPE_SLONG8 ? 8 :
        type == TAG_TYPE_IFD8 ? 0 : 0;
}

static uint32_t exifEntryValuetoUInt32(ExifEntry entry)
{
    switch (entry.type)
    {
    case TAG_TYPE_BYTE:
    case TAG_TYPE_UNDEFINED:
        return (entry.count > 0) ? entry.value.field_u8 : 0;
    case TAG_TYPE_SBYTE:
        return (entry.count > 0) ? static_cast<uint32_t>(entry.value.field_s8) : 0;
    case TAG_TYPE_SHORT:
        return (entry.count > 0) ? entry.value.field_u16 : 0;
    case TAG_TYPE_SSHORT:
        return (entry.count > 0) ? static_cast<uint32_t>(entry.value.field_s16) : 0;
    case TAG_TYPE_LONG:
    case TAG_TYPE_SLONG:
        return (entry.count > 0) ? entry.value.field_u32 : 0;
    case TAG_TYPE_ASCII:
        return 0;
    case TAG_TYPE_RATIONAL:
    case TAG_TYPE_SRATIONAL:
        return 0;
    default:
        return 0;
    }
}

static std::vector<uchar> exifEntryValuetoBytes(ExifEntry entry)
{
    std::vector<uchar> bytes;
    switch (entry.type)
    {
    case TAG_TYPE_ASCII:
    {
        const char* str = entry.value.field_str.c_str();
        int len = (int)strlen(str);
        bytes.insert(bytes.end(), str, str + len);
        if (len < entry.count) // pad with nulls if needed
            bytes.insert(bytes.end(), entry.count - len, 0x00);
    }
    break;
    case TAG_TYPE_UNDEFINED:
    {
        bytes.insert(bytes.end(), entry.value.field_u8, entry.value.field_u8 + entry.count);
    }
    break;
    case TAG_TYPE_RATIONAL:
    {
        for (int i = 0; i < entry.count; ++i)
        {
            uint32_t numerator = entry.value.field_urational[i].num;
            uint32_t denominator = entry.value.field_urational[i].denom;
            bytes.insert(bytes.end(), reinterpret_cast<uchar*>(&numerator), reinterpret_cast<uchar*>(&numerator) + 4);
            bytes.insert(bytes.end(), reinterpret_cast<uchar*>(&denominator), reinterpret_cast<uchar*>(&denominator) + 4);
        }
    }
    break;
    case TAG_TYPE_SRATIONAL:
    {
        for (int i = 0; i < entry.count; ++i)
        {
            uint32_t numerator = entry.value.field_srational[i].num;
            uint32_t denominator = entry.value.field_srational[i].denom;
            bytes.insert(bytes.end(), reinterpret_cast<uchar*>(&numerator), reinterpret_cast<uchar*>(&numerator) + 4);
            bytes.insert(bytes.end(), reinterpret_cast<uchar*>(&denominator), reinterpret_cast<uchar*>(&denominator) + 4);
        }
    }
    break;
    default:
        // other types are handled inline in toUInt32()
        break;
    }

    return bytes;
}

bool encodeExif(const std::vector<std::vector<ExifEntry>>& exif_entries, std::vector<uchar>& data)
{
    data.clear();

    if (exif_entries.empty() || exif_entries[0].empty())
        return false;

    // TIFF Header
    size_t tiffHeaderOffset = data.size();
    data.push_back('I'); data.push_back('I'); // Little Endian
    data.push_back(0x2A); data.push_back(0x00); // 0x002A
    uint32_t firstIFDOffset = 8; // IFD starts after TIFF header
    data.insert(data.end(), (uchar*)&firstIFDOffset, (uchar*)&firstIFDOffset + 4);


    for (const auto& ifdEntries : exif_entries)
    {
        // Temporary buffer for Value Area (big data blocks)
        std::vector<uchar> valueArea;

        uint16_t entryCount = static_cast<uint16_t>(ifdEntries.size());

        // Entry count
        data.insert(data.end(), (uchar*)&entryCount, (uchar*)&entryCount + 2);

        // Placeholders for Entries
        size_t entryStart = data.size();
        data.resize(data.size() + entryCount * 12);

        // Next IFD Offset (set to 0)
        uint32_t nextIFDOffset = 0;
        data.insert(data.end(), (uchar*)&nextIFDOffset, (uchar*)&nextIFDOffset + 4);

        // Process Entries
        size_t entryOffset = entryStart;
        for (const auto& entry : ifdEntries)
        {
            uint16_t tagId = entry.tagId;
            uint16_t type = entry.type;
            uint32_t count = entry.count;

            // Write Tag ID
            std::memcpy(&data[entryOffset], &tagId, 2); entryOffset += 2;
            // Write Type
            std::memcpy(&data[entryOffset], &type, 2); entryOffset += 2;
            // Write Count
            std::memcpy(&data[entryOffset], &count, 4); entryOffset += 4;

            uint32_t valueSize = count * getExifTagTypeSize(type);
            if (valueSize <= 4)
            {
                uint32_t val = exifEntryValuetoUInt32(entry);
                std::memcpy(&data[entryOffset], &val, 4);
            }
            else
            {
                uint32_t offset = static_cast<uint32_t>(tiffHeaderOffset + data.size() + valueArea.size());
                std::memcpy(&data[entryOffset], &offset, 4);
                std::vector<uchar> valData = exifEntryValuetoBytes(entry);

                valueArea.insert(valueArea.end(), valData.begin(), valData.end());

                if (valData.size() % 2 != 0) // Align to 2 bytes
                    valueArea.push_back(0x00);
            }
            entryOffset += 4;

        }

        // Append Value Area after IFDs
        data.insert(data.end(), valueArea.begin(), valueArea.end());
    }


    return true;
}


bool decodeExif(const std::vector<uchar>& data, std::vector< std::vector<ExifEntry> >& exif_entries)
{
    ExifReader reader;
    return reader.parseExif(data.data(), data.size(), exif_entries);
}

std::string exifTagIdToString(ExifTagId tag);
std::string tagTypeToString(ExifTagType type);

template<typename _Tp> void dumpScalar(std::ostream& strm, _Tp v)
{
    strm << v;
}

template<> void dumpScalar(std::ostream& strm, int64_t v)
{
    strm << v;
}

template<> void dumpScalar(std::ostream& strm, double v)
{
    strm << cv::format("%.8g", v);
}

template<> void dumpScalar(std::ostream& strm, srational64_t v)
{
    strm << v.num << "/" << v.denom << cv::format(" (%.4f)", (double)v.num / v.denom);
}

template<> void dumpScalar(std::ostream& strm, urational64_t v)
{
    strm << v.num << "/" << v.denom;
    if (v.denom != 0)
        strm << cv::format(" (%.4f)", static_cast<double>(v.num) / v.denom);
    else
        strm << " (NaN)";
}

template <typename _Tp> void dumpVector(std::ostream& strm, const std::vector<_Tp>& v)
{
    size_t i, nvalues = v.size();
    if (nvalues > 1)
        strm << '[';
    for (i = 0; i < nvalues; i++) {
        if (i > 0)
            strm << ", ";
        if (i >= 3 && i + 6 < nvalues) {
            strm << "... ";
            i = nvalues - 3;
        }
        dumpScalar(strm, v[i]);
    }
    if (nvalues > 1)
        strm << ']';
}

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

/**
 * @brief ExifReader constructor
 */
ExifReader::ExifReader() : m_format(ExifEndianness::NONE)
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
ExifEntry ExifReader::getEntrybyTagId(const ExifTagId tag) const
{
    ExifEntry entry;
    std::map<int, ExifEntry>::const_iterator it = m_exif.find(tag);

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
bool ExifReader::parseExif(const unsigned char* data, const size_t size)
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
        ExifEntry exifEntry = parseExifEntry( offset );
        m_exif.insert( std::make_pair( exifEntry.tagId, exifEntry ) );
        offset += tiffFieldSize;
    }
}

bool ExifReader::parseExif(const unsigned char* data, const size_t size, std::vector< std::vector<ExifEntry> >& exif_entries_vec)
{
    if (data && size > 0)
    {
        m_data.assign(data, data + size);
    }
    else
    {
        return false;
    }

    m_format = getFormat();

    if (!checkTagMark())
    {
        return false;
    }

    std::vector<uint32_t> ifd_offsets;
    ifd_offsets.push_back(getStartOffset());
    size_t current_ifd = 0;
    while (current_ifd < ifd_offsets.size())
    {
        uint32_t offset = ifd_offsets[current_ifd];

        size_t numEntry = getNumDirEntry(offset);
        offset += 2; //go to start of tag fields

        std::vector<ExifEntry> exif_entries;
        for (size_t i = 0; i < numEntry; ++i)
        {
            ExifEntry exifEntry = parseExifEntry(offset);
            exif_entries.push_back(exifEntry);
            if (exifEntry.tagId == 0x8769 || exifEntry.tagId == 0x8825) // Exif or GPS IFD pointer
            {
                uint32_t sub_ifd_offset = exifEntry.value.field_u32;
                if (sub_ifd_offset < m_data.size())
                    ifd_offsets.push_back(sub_ifd_offset);
            }
            offset += tiffFieldSize;
        }
        exif_entries_vec.push_back(exif_entries);
        // Handle IFD1 (Next IFD offset at the end of current IFD)
        if (offset + 4 <= m_data.size())
        {
            uint32_t next_ifd_offset = getU32(offset);
            if (next_ifd_offset != 0 && next_ifd_offset < m_data.size())
                ifd_offsets.push_back(next_ifd_offset);
        }
        current_ifd++;
    }
    return true;
}

/**
 * @brief Get endianness of exif information
 *          This is internal function and is not exposed to client
 *
 * @return INTEL, MOTO or NONE
 */
int ExifReader::getFormat() const
{
    if (m_data.size() < 1)
        return ExifEndianness::NONE;

    if( m_data.size() > 1 && m_data[0] != m_data[1] )
    {
        return ExifEndianness::NONE;
    }

    if( m_data[0] == 'I' )
    {
        return ExifEndianness::INTEL;
    }

    if( m_data[0] == 'M' )
    {
        return ExifEndianness::MOTO;
    }

    return ExifEndianness::NONE;
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
 * @return ExifEntry structure which corresponds to particular entry
 *
 */
ExifEntry ExifReader::parseExifEntry(const size_t offset)
{
    ExifEntry exifentry;
    exifentry.tagId = ExifTagId(getU16(offset));
    exifentry.type = ExifTagType(getU16(offset + 2));
    exifentry.count = getU32(offset + 4);

    switch (exifentry.type)
    {
    case TAG_TYPE_BYTE:
    case TAG_TYPE_SBYTE:
        exifentry.value.field_u8 = m_data[offset + 8];
        break;
    case TAG_TYPE_ASCII:
        exifentry.value.field_str = getString(offset);
        break;

    case TAG_TYPE_SHORT:
        exifentry.value.field_u16 = getU16(offset + 8);
        break;

    case TAG_TYPE_UNDEFINED:
    case TAG_TYPE_LONG:
        exifentry.value.field_u32 = getU32(offset + 8);
        break;

    case TAG_TYPE_SSHORT:
        exifentry.value.field_s16 = (int16_t)getU16(offset + 8);
        break;

    case TAG_TYPE_SLONG:
        exifentry.value.field_s32 = (int32_t)getU32(offset + 8);
        break;

    case TAG_TYPE_RATIONAL:
        exifentry.value.field_urational = getURational(offset);
        break;
    case TAG_TYPE_SRATIONAL:
        exifentry.value.field_srational = getSRational(offset);
        break;
    default:
        CV_LOG_WARNING(NULL, "Undefined ExifTagValue type " << (int)exifentry.type);
        break;
    }

    return exifentry;
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
    size_t dataOffset = offset + 8; // position of data in the field
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

    if( m_format == ExifEndianness::INTEL )
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

    if( m_format == ExifEndianness::INTEL )
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
std::vector<urational64_t> ExifReader::getURational(const size_t offset) const
{
    std::vector<urational64_t> result;
    size_t dataOffset = getU32(offset + 8);
    if (dataOffset > m_data.size() || dataOffset + 8 > m_data.size()) {
        throw ExifParsingError();
    }
    for (uint32_t count = getU32(offset + 4); count > 0; count--)
    {
        urational64_t item;
        item.num = getU32(dataOffset);
        item.denom = getU32(dataOffset + 4);
        result.push_back(item);
        dataOffset += 8;
    }
    return result;
}

std::vector<srational64_t> ExifReader::getSRational(const size_t offset) const
{
    std::vector<srational64_t> result;
    size_t dataOffset = getU32(offset + 8);
    if (dataOffset > m_data.size() || dataOffset + 8 > m_data.size()) {
        throw ExifParsingError();
    }
    for (uint32_t count = getU32(offset + 4); count > 0; count--)
    {
        srational64_t item;
        item.num = getU32(dataOffset);
        item.denom = getU32(dataOffset + 4);
        result.push_back(item);
        dataOffset += 8;
    }
    return result;
}

std::string tagTypeToString(ExifTagType type)
{
    const char* typestr =
        type == TAG_TYPE_NOTYPE ? "NoType" :
        type == TAG_TYPE_BYTE ? "Byte" :
        type == TAG_TYPE_ASCII ? "ASCII" :
        type == TAG_TYPE_SHORT ? "Short" :
        type == TAG_TYPE_LONG ? "Long" :
        type == TAG_TYPE_RATIONAL ? "Rational" :
        type == TAG_TYPE_SBYTE ? "SByte" :
        type == TAG_TYPE_UNDEFINED ? "Undefined" :
        type == TAG_TYPE_SSHORT ? "SShort" :
        type == TAG_TYPE_SLONG ? "SLong" :
        type == TAG_TYPE_SRATIONAL ? "SRational" :
        type == TAG_TYPE_FLOAT ? "Float" :
        type == TAG_TYPE_DOUBLE ? "Double" :
        type == TAG_TYPE_IFD ? "IFD" :
        type == TAG_TYPE_LONG8 ? "Long8" :
        type == TAG_TYPE_SLONG8 ? "SLong8" :
        type == TAG_TYPE_IFD8 ? "IFD8" : nullptr;
    return typestr ? std::string(typestr) : cv::format("Unkhown type <%d>", (int)type);
}

std::string exifTagIdToString(ExifTagId tag)
{
    const char* tagstr =
        tag == TAG_EMPTY ? "<empty>" :
        tag == TAG_SUB_FILETYPE ? "SubFileType" :
        tag == TAG_IMAGE_WIDTH ? "Image Width" :
        tag == TAG_IMAGE_LENGTH ? "Image Height" :
        tag == TAG_BITS_PER_SAMPLE ? "BitsPerSample" :
        tag == TAG_COMPRESSION ? "Compression" :
        tag == TAG_PHOTOMETRIC ? "Photometric" :
        tag == TAG_IMAGEDESCRIPTION ? "Image Description" :
        tag == TAG_MAKE ? "Make" :
        tag == TAG_MODEL ? "Model" :
        tag == TAG_STRIP_OFFSET ? "StripOffset" :
        tag == TAG_SAMPLES_PER_PIXEL ? "SamplesPerPixel" :
        tag == TAG_ROWS_PER_STRIP ? "RowsPerStrip" :
        tag == TAG_STRIP_BYTE_COUNTS ? "StripByteCounts" :
        tag == TAG_PLANAR_CONFIG ? "PlanarConfig" :
        tag == TAG_ORIENTATION ? "Orientation" :
        tag == TAG_XRESOLUTION ? "XResolution" :
        tag == TAG_YRESOLUTION ? "YResolution" :
        tag == TAG_RESOLUTION_UNIT ? "ResolutionUnit" :
        tag == TAG_SOFTWARE ? "Software" :
        tag == TAG_MODIFYDATE ? "ModifyDate" :
        tag == TAG_SAMPLEFORMAT ? "SampleFormat" :
        tag == TAG_YCBCRPOSITIONING ? "YCbCrPositioning" :
        tag == TAG_JPGFROMRAWSTART ? "JpgFromRawStart " :
        tag == TAG_JPGFROMRAWLENGTH ? "JpgFromRawLength" :
        tag == TAG_CFA_REPEAT_PATTERN_DIM ? "CFARepeatPatternDim" :
        tag == TAG_CFA_PATTERN ? "CFAPattern" :

        tag == TAG_COMPONENTSCONFIGURATION ? "Components Configuration" :

        tag == TAG_COPYRIGHT ? "Copyright" :
        tag == TAG_EXPOSURE_TIME ? "Exposure Time" :
        tag == TAG_FNUMBER ? "FNumber" :

        tag == TAG_EXIF_OFFSET ? "Exif Offset" :

        tag == TAG_EXPOSUREPROGRAM ? "Exposure Program" :
        tag == TAG_GPSINFO ? "GPS Info" :
        tag == TAG_ISOSPEED ? "ISO Speed" :

        tag == TAG_DATETIME_CREATE ? "CreateDate" :
        tag == TAG_DATETIME_ORIGINAL ? "DateTimeOriginal" :

        tag == TAG_FLASH ? "Flash" :
        tag == TAG_FOCALLENGTH ? "FocalLength" :
        tag == TAG_EP_STANDARD_ID ? "TIFF/EPStandardID" :

        tag == TAG_SHUTTER_SPEED ? "Shutter Speed" :
        tag == TAG_APERTURE_VALUE ? "Aperture Value" :

        tag == TAG_EXPOSUREBIASVALUE ? "ExposureBiasValue" :
        tag == TAG_MAXAPERTUREVALUE ? "MaxApertureValue" :
        tag == TAG_SUBJECTDISTANCE ? "SubjectDistance" :
        tag == TAG_METERINGMODE ? "MeteringMode" :
        tag == TAG_LIGHTSOURCE ? "Light Source" :
        tag == TAG_FLASH ? "Flash" :

        tag == TAG_MAKERNOTE ? "MakerNote" :
        tag == TAG_USERCOMMENT ? "User Comment" :

        tag == TAG_SUBSECTIME ? "SubSec Time" :
        tag == TAG_SUBSECTIME_ORIGINAL ? "SubSec Original Time" :
        tag == TAG_SUBSECTIME_DIGITIZED ? "SubSec Digitized Time" :

        tag == TAG_FLASHPIXVERSION ? "Flashpix Version" :
        tag == TAG_COLORSPACE ? "ColorSpace" :
        tag == TAG_EXIF_IMAGE_WIDTH ? "Exif Image Width" :
        tag == TAG_EXIF_IMAGE_HEIGHT ? "Exif Image Height" :
        tag == TAG_WHITE_BALANCE ? "White Balance" :

        tag == TAG_EXIF_VERSION ? "Exif Version" :

        tag == TAG_FOCALPLANEXRESOLUTION ? "FocalPlaneXResolution" :
        tag == TAG_FOCALPLANEYRESOLUTION ? "FocalPlaneYResolution" :
        tag == TAG_FOCALPLANERESOLUTIONUNIT ? "FocalPlaneResolutionUnit" :

        tag == TAG_CUSTOMRENDERED ? "CustomRendered" :
        tag == TAG_EXPOSUREMODE ? "ExposureMode" :

        tag == TAG_BODYSERIALNUMBER ? "BodySerialNumber" :
        tag == TAG_LENSSPECIFICATION ? "LensSpecification" :
        tag == TAG_LENSMODEL ? "LensModel" :
        tag == TAG_SCENECAPTURETYPE ? "SceneCaptureType" :

        tag == TAG_DNG_VERSION ? "DNGVersion" :
        tag == TAG_DNG_BACKWARD_VERSION ? "DNGBackwardVersion" :
        tag == TAG_UNIQUE_CAMERA_MODEL ? "UniqueCameraModel" :
        tag == TAG_CHROMA_BLUR_RADIUS ? "ChromaBlurRadius" :
        tag == TAG_CFA_PLANECOLOR ? "CFAPlaneColor" :
        tag == TAG_CFA_LAYOUT ? "CFALayout" :
        tag == TAG_BLACK_LEVEL_REPEAT_DIM ? "BlackLevelRepeatDim" :
        tag == TAG_BLACK_LEVEL ? "BlackLevel" :
        tag == TAG_WHITE_LEVEL ? "WhiteLevel" :
        tag == TAG_DEFAULT_SCALE ? "DefaultScale" :
        tag == TAG_DEFAULT_CROP_ORIGIN ? "DefaultCropOrigin" :
        tag == TAG_DEFAULT_CROP_SIZE ? "DefaultCropSize" :
        tag == TAG_COLOR_MATRIX1 ? "ColorMatrix1" :
        tag == TAG_COLOR_MATRIX2 ? "ColorMatrix2" :
        tag == TAG_CAMERA_CALIBRATION1 ? "CameraCalibration1" :
        tag == TAG_CAMERA_CALIBRATION2 ? "CameraCalibration2" :
        tag == TAG_ANALOG_BALANCE ? "AnalogBalance" :
        tag == TAG_AS_SHOT_NEUTRAL ? "AsShotNeutral" :
        tag == TAG_AS_SHOT_WHITE_XY ? "AsShotWhiteXY" :
        tag == TAG_BASELINE_EXPOSURE ? "BaselineExposure" :
        tag == TAG_CALIBRATION_ILLUMINANT1 ? "CalibrationIlluminant1" :
        tag == TAG_CALIBRATION_ILLUMINANT2 ? "CalibrationIlluminant2" :
        tag == TAG_EXTRA_CAMERA_PROFILES ? "ExtraCameraProfiles" :
        tag == TAG_PROFILE_NAME ? "ProfileName" :
        tag == TAG_AS_SHOT_PROFILE_NAME ? "AsShotProfileName" :
        tag == TAG_PREVIEW_COLORSPACE ? "PreviewColorspace" :
        tag == TAG_OPCODE_LIST2 ? "OpCodeList2" :
        tag == TAG_NOISE_PROFILE ? "NoiseProfile" :
        tag == TAG_DEFAULT_BLACK_RENDER ? "BlackRender" :
        tag == TAG_ACTIVE_AREA ? "ActiveArea" :
        tag == TAG_FORWARD_MATRIX1 ? "ForwardMatrix1" :
        tag == TAG_FORWARD_MATRIX2 ? "ForwardMatrix2" : nullptr;
    return tagstr ? std::string(tagstr) : cv::format("<unknown tag>(%d)", (int)tag);
};

std::ostream& ExifEntry::dump(std::ostream& strm) const
{
    if (empty()) {
        strm << "<empty>";
        return strm;
    }

    strm << exifTagIdToString(tagId) << ": ";

    switch (type) {
    case TAG_TYPE_ASCII:
        strm << "\"" << value.field_str << "\"";
        break;
    case TAG_TYPE_BYTE:
        strm << static_cast<int>(value.field_u8);
        break;
    case TAG_TYPE_SHORT:
        strm << value.field_u16;
        break;
    case TAG_TYPE_LONG:
        strm << value.field_u32;
        break;
    case TAG_TYPE_FLOAT:
        strm << value.field_float;
        break;
    case TAG_TYPE_DOUBLE:
        strm << value.field_double;
        break;
    case TAG_TYPE_RATIONAL:
        dumpVector(strm, value.field_urational);
        break;
    case TAG_TYPE_SRATIONAL:
        dumpVector(strm, value.field_srational);
        break;
    case TAG_TYPE_UNDEFINED:
    {
        uint32_t raw = value.field_u32;
        strm << "[ ";
        for (int i = 3; i >= 0; --i) {
            uint8_t byte = static_cast<uint8_t>((raw >> (i * 8)) & 0xFF);
            strm << "0x" << std::hex << std::uppercase
                << std::setw(2) << std::setfill('0')
                << static_cast<int>(byte);
            if (i > 0)
                strm << ", ";
        }
        strm << " ]";
        strm << std::dec; // restore decimal output
        break;
    }
    default:
        break;
    }

    strm << std::endl;
    return strm;
}

} //namespace cv
