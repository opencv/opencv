// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "exif.hpp"
#include "opencv2/core/utils/logger.hpp"
#include <iomanip>
#include <set>

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
        return (entry.count > 0) ? entry.getValueAsInt() : 0;
    case TAG_TYPE_SBYTE:
        return (entry.count > 0) ? static_cast<uint32_t>(entry.getValueAsInt()) : 0;
    case TAG_TYPE_SHORT:
        return (entry.count > 0) ? entry.getValueAsInt() : 0;
    case TAG_TYPE_SSHORT:
        return (entry.count > 0) ? static_cast<uint32_t>(entry.getValueAsInt()) : 0;
    case TAG_TYPE_LONG:
    case TAG_TYPE_SLONG:
        return (entry.count > 0) ? entry.getValueAsInt() : 0;
    case TAG_TYPE_ASCII:
        return 0;
    case TAG_TYPE_RATIONAL:
    case TAG_TYPE_SRATIONAL:
        return 0;
    default:
        return 0;
    }
}

static std::vector<uchar> exifEntryValuetoBytes(const ExifEntry& entry)
{
    std::vector<uchar> bytes;
    switch (entry.type)
    {
    case TAG_TYPE_ASCII:
    {
        std::string str = entry.getValueAsString();
        size_t len = str.size();

        // Insert string data
        bytes.insert(bytes.end(),
            reinterpret_cast<const uchar*>(str.data()),
            reinterpret_cast<const uchar*>(str.data()) + len);

        // Pad with nulls if needed
        if (len < static_cast<size_t>(entry.count))
            bytes.insert(bytes.end(), entry.count - len, 0x00);
    }
    break;
    case TAG_TYPE_UNDEFINED:
    {
        return entry.getValueAsRaw();
    }
    break;
    case TAG_TYPE_SRATIONAL:
    case TAG_TYPE_RATIONAL:
    {
        std::vector<SRational> srational_vec = entry.getValueAsRational();
        for (size_t i = 0; i < srational_vec.size(); ++i)
        {
            uint32_t numerator = srational_vec[i].num;
            uint32_t denominator = srational_vec[i].denom;
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
            uint16_t tagId = (uint16_t)entry.tagId;
            uint16_t type = (uint16_t)entry.type;
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
                if (entry.type == TAG_TYPE_UNDEFINED || entry.type == TAG_TYPE_ASCII)
                    std::memcpy(&data[entryOffset], exifEntryValuetoBytes(entry).data(), 4);
                else
                {
                    uint32_t val = entry.getValueAsInt();
                    std::memcpy(&data[entryOffset], &val, 4);
                }
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

template<> void dumpScalar(std::ostream& strm, SRational v)
{
    strm << v.num << "/" << v.denom << cv::format(" (%.4f)", (double)v.num / v.denom);
}

template<> void dumpScalar(std::ostream& strm, Rational v)
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

// Helper: endian-aware offset extraction
uint32_t ExifReader::extractIFDOffset(const ExifEntry& entry) const
{
    uint32_t offsetVal = 0;

    // If the type is LONG or SLONG, the offset is directly the 32-bit value
    if (entry.type == TAG_TYPE_LONG || entry.type == TAG_TYPE_SLONG)
    {
            offsetVal = static_cast<uint32_t>(entry.getValueAsInt());
    }
    else
    {
        // For other types, you may store offsets in a different union member
        // or directly in a separate struct field when parsing
        // This fallback assumes the offset fits in uint32_t
        offsetVal = static_cast<uint32_t>(exifEntryValuetoUInt32(entry));
    }

    // Apply endian conversion if needed
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    uint8_t tmp[4];
    std::memcpy(tmp, &offsetVal, 4);
    std::reverse(tmp, tmp + 4);
    std::memcpy(&offsetVal, tmp, 4);
#endif

    return offsetVal;
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
                uint32_t sub_ifd_offset = exifEntry.getValueAsInt();
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
    case TAG_TYPE_UNDEFINED:
        exifentry.setValueAsRaw(getRaw(offset));
        break;
    case TAG_TYPE_ASCII:
        exifentry.setValueAsString(getString(offset));
        break;

    case TAG_TYPE_SHORT:
        exifentry.setValueAsInt(getU16(offset + 8));
        break;

    case TAG_TYPE_LONG:
        exifentry.setValueAsInt(getU32(offset + 8));
        break;

    case TAG_TYPE_SSHORT:
        exifentry.setValueAsInt((int16_t)getU16(offset + 8));
        break;

    case TAG_TYPE_SLONG:
        exifentry.setValueAsInt((int32_t)getU32(offset + 8));
        break;

    case TAG_TYPE_RATIONAL:
    case TAG_TYPE_SRATIONAL:
        exifentry.setValueAsRational(getSRational(offset));
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
    std::string result( it, it + size - 1 ); //copy vector content into result
    return result;
}

std::vector<uchar> ExifReader::getRaw(const size_t offset) const
{
    size_t size = getU32(offset + 4);
    size_t dataOffset = offset + 8; // position of data in the field
    if (size > maxDataSize + 4)
    {
        dataOffset = getU32(offset + 8);
    }

    if (dataOffset > m_data.size() || dataOffset + size > m_data.size()) {
        throw ExifParsingError();
    }
    std::vector<uchar>::const_iterator it = m_data.begin() + dataOffset;
    std::vector<uchar> result(it, it + size); //copy vector content into result
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
std::vector<Rational> ExifReader::getRational(const size_t offset) const
{
    std::vector<Rational> result;
    size_t dataOffset = getU32(offset + 8);
    if (dataOffset > m_data.size() || dataOffset + 8 > m_data.size()) {
        throw ExifParsingError();
    }
    for (uint32_t count = getU32(offset + 4); count > 0; count--)
    {
        Rational item;
        item.num = getU32(dataOffset);
        item.denom = getU32(dataOffset + 4);
        result.push_back(item);
        dataOffset += 8;
    }
    return result;
}

std::vector<SRational> ExifReader::getSRational(const size_t offset) const
{
    std::vector<SRational> result;
    size_t dataOffset = getU32(offset + 8);
    if (dataOffset > m_data.size() || dataOffset + 8 > m_data.size()) {
        throw ExifParsingError();
    }
    for (uint32_t count = getU32(offset + 4); count > 0; count--)
    {
        SRational item;
        item.num = getU32(dataOffset);
        item.denom = getU32(dataOffset + 4);
        result.push_back(item);
        dataOffset += 8;
    }
    return result;
}

std::string ExifEntry::getTagTypeAsString() const
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

std::string ExifEntry::getTagIdAsString() const
{
    int tag = tagId;
    const char* tagstr =
        tag == TAG_EMPTY ? "<empty>" :
        tag == TAG_SUB_FILE_TYPE ? "Sub File Type" :
        tag == TAG_IMAGE_WIDTH ? "Image Width" :
        tag == TAG_IMAGE_LENGTH ? "Image Height" :
        tag == TAG_BITS_PER_SAMPLE ? "Bits Per Sample" :
        tag == TAG_COMPRESSION ? "Compression" :
        tag == TAG_PHOTOMETRIC ? "Photometric" :
        tag == TAG_IMAGE_DESCRIPTION ? "Image Description" :
        tag == TAG_MAKE ? "Make" :
        tag == TAG_MODEL ? "Model" :
        tag == TAG_STRIP_OFFSET ? "Strip Offset" :
        tag == TAG_SAMPLES_PER_PIXEL ? "Samples Per Pixel" :
        tag == TAG_ROWS_PER_STRIP ? "Rows Per Strip" :
        tag == TAG_STRIP_BYTE_COUNTS ? "Strip Byte Counts" :
        tag == TAG_PLANAR_CONFIG ? "Planar Config" :
        tag == TAG_ORIENTATION ? "Orientation" :
        tag == TAG_X_RESOLUTION ? "X Resolution" :
        tag == TAG_Y_RESOLUTION ? "Y Resolution" :
        tag == TAG_RESOLUTION_UNIT ? "Resolution Unit" :
        tag == TAG_SOFTWARE ? "Software" :
        tag == TAG_MODIFY_DATE ? "Modify Date" :
        tag == TAG_ARTIST ? "Artist" :
        tag == TAG_HOST_COMPUTER ? "Host Computer" :

        tag == TAG_SAMPLE_FORMAT ? "Sample Format" :
        tag == TAG_YCBCRPOSITIONING ? "YCbCr Positioning" :
        tag == TAG_JPGFROMRAWSTART ? "Jpg From Raw Start " :
        tag == TAG_JPGFROMRAWLENGTH ? "Jpg From Raw Length" :
        tag == TAG_CFA_REPEAT_PATTERN_DIM ? "CFA Repeat Pattern Dim" :
        tag == TAG_CFA_PATTERN ? "CFA Pattern" :

        tag == TAG_COMPONENTS_CONFIGURATION ? "Components Configuration" :

        tag == TAG_COPYRIGHT ? "Copyright" :
        tag == TAG_EXPOSURE_TIME ? "Exposure Time" :
        tag == TAG_F_NUMBER ? "F Number" :

        tag == TAG_EXIF_OFFSET ? "Exif Offset" :

        tag == TAG_EXPOSURE_PROGRAM ? "Exposure Program" :
        tag == TAG_GPS_INFO ? "GPS Info" :
        tag == TAG_ISO_SPEED ? "ISO Speed" :

        tag == TAG_DATETIME_CREATE ? "Create Date" :
        tag == TAG_DATETIME_ORIGINAL ? "DateTime Original" :

        tag == TAG_OFFSETTIME ? "Offset Time" :
        tag == TAG_OFFSETTIME_ORIGINAL ? "Offset Time Original" :
        tag == TAG_OFFSETTIME_DIGITIZED ? "Offset Time Digitized" :

        tag == TAG_COMPRESSED_BITS_PER_PIXEL ? "CompressedBitsPerPixel" :

        tag == TAG_FLASH ? "Flash" :
        tag == TAG_FOCAL_LENGTH ? "Focal Length" :
        tag == TAG_EP_STANDARD_ID ? "TIFF/EPStandard ID" :

        tag == TAG_SHUTTER_SPEED ? "Shutter Speed" :
        tag == TAG_APERTURE_VALUE ? "Aperture Value" :
        tag == TAG_BRIGHTNESS_VALUE ? "Brightness Value" :
        tag == TAG_EXPOSURE_BIAS_VALUE ? "Exposure Bias Value" :
        tag == TAG_MAX_APERTURE_VALUE ? "Max Aperture Value" :
        tag == TAG_SUBJECT_DISTANCE ? "Subject Distance" :
        tag == TAG_METERING_MODE ? "Metering Mode" :
        tag == TAG_LIGHT_SOURCE ? "Light Source" :
        tag == TAG_FLASH ? "Flash" :
        tag == TAG_SUBJECT_AREA ? "Subject Area" :
        tag == TAG_MAKER_NOTE ? "Maker Note" :
        tag == TAG_USER_COMMENT ? "User Comment" :

        tag == TAG_SUBSECTIME ? "SubSec Time" :
        tag == TAG_SUBSECTIME_ORIGINAL ? "SubSec Original Time" :
        tag == TAG_SUBSECTIME_DIGITIZED ? "SubSec Digitized Time" :

        tag == TAG_FLASH_PIX_VERSION ? "Flashpix Version" :
        tag == TAG_COLOR_SPACE ? "Color Space" :
        tag == TAG_EXIF_IMAGE_WIDTH ? "Exif Image Width" :
        tag == TAG_EXIF_IMAGE_HEIGHT ? "Exif Image Height" :
        tag == TAG_WHITE_BALANCE ? "White Balance" :

        tag == TAG_EXIF_VERSION ? "Exif Version" :

        tag == TAG_FOCAL_PLANE_X_RESOLUTION ? "Focal Plane X Resolution" :
        tag == TAG_FOCAL_PLANE_Y_RESOLUTION ? "Focal Plane Y Resolution" :
        tag == TAG_FOCAL_PLANE_RESOLUTION_UNIT ? "Focal Plane Resolution Unit" :

        tag == TAG_SCENE_TYPE ? "Scene Type" :

        tag == TAG_CUSTOM_RENDERED ? "Custom Rendered" :
        tag == TAG_EXPOSURE_MODE ? "Exposure Mode" :

        tag == TAG_BODY_SERIAL_NUMBER ? "Body Serial Number" :
        tag == TAG_LENS_SPECIFICATION ? "Lens Specification" :
        tag == TAG_LENS_MAKE ? "Lens Make" :
        tag == TAG_LENS_MODEL ? "Lens Model" :
        tag == TAG_SCENE_CAPTURE_TYPE ? "Scene Capture Type" :

        tag == TAG_DNG_VERSION ? "DNG Version" :
        tag == TAG_DNG_BACKWARD_VERSION ? "DNG Backward Version" :
        tag == TAG_UNIQUE_CAMERA_MODEL ? "Unique Camera Model" :
        tag == TAG_CHROMA_BLUR_RADIUS ? "Chroma Blur Radius" :
        tag == TAG_CFA_PLANECOLOR ? "CFA Plane Color" :
        tag == TAG_CFA_LAYOUT ? "CFA Layout" :
        tag == TAG_BLACK_LEVEL_REPEAT_DIM ? "Black Level Repeat Dim" :
        tag == TAG_BLACK_LEVEL ? "Black Level" :
        tag == TAG_WHITE_LEVEL ? "White Level" :
        tag == TAG_DEFAULT_SCALE ? "Default Scale" :
        tag == TAG_DEFAULT_CROP_ORIGIN ? "Default Crop Origin" :
        tag == TAG_DEFAULT_CROP_SIZE ? "Default Crop Size" :
        tag == TAG_COLOR_MATRIX1 ? "Color Matrix1" :
        tag == TAG_COLOR_MATRIX2 ? "Color Matrix2" :
        tag == TAG_CAMERA_CALIBRATION1 ? "Camera Calibration1" :
        tag == TAG_CAMERA_CALIBRATION2 ? "Camera Calibration2" :
        tag == TAG_ANALOG_BALANCE ? "Analog Balance" :
        tag == TAG_AS_SHOT_NEUTRAL ? "As Shot Neutral" :
        tag == TAG_AS_SHOT_WHITE_XY ? "As Shot White XY" :
        tag == TAG_BASELINE_EXPOSURE ? "Baseline Exposure" :
        tag == TAG_CALIBRATION_ILLUMINANT1 ? "Calibration Illuminant 1" :
        tag == TAG_CALIBRATION_ILLUMINANT2 ? "Calibration Illuminant 2" :
        tag == TAG_EXTRA_CAMERA_PROFILES ? "Extra Camera Profiles" :
        tag == TAG_PROFILE_NAME ? "Profile Name" :
        tag == TAG_AS_SHOT_PROFILE_NAME ? "As Shot Profile Name" :
        tag == TAG_PREVIEW_COLORSPACE ? "Preview Colorspace" :
        tag == TAG_OPCODE_LIST2 ? "Op Code List 2" :
        tag == TAG_NOISE_PROFILE ? "Noise Profile" :
        tag == TAG_DEFAULT_BLACK_RENDER ? "Black Render" :
        tag == TAG_ACTIVE_AREA ? "Active Area" :
        tag == TAG_FORWARD_MATRIX1 ? "Forward Matrix 1" :
        tag == TAG_FORWARD_MATRIX2 ? "Forward Matrix 2" : nullptr;
    return tagstr ? std::string(tagstr) : cv::format("<unknown tag>(%d)", (int)tag);
};

std::string ExifEntry::dumpAsString() const {
    std::ostringstream oss;
    dump(oss);
    return oss.str();
}

std::ostream& ExifEntry::dump(std::ostream& strm) const
{
    if (empty()) {
        strm << "<empty>";
        return strm;
    }

    strm << getTagIdAsString() << ": ";

    switch (type) {
    case TAG_TYPE_ASCII:
        strm << "\"" << getValueAsString() << "\"";
        break;
    case TAG_TYPE_BYTE:
    case TAG_TYPE_SHORT:
    case TAG_TYPE_LONG:
        strm << getValueAsInt();
        break;
    case TAG_TYPE_FLOAT:

        strm << getValueAsInt();
        break;
    case TAG_TYPE_DOUBLE:
        strm << getValueAsInt();
        break;
    case TAG_TYPE_RATIONAL:
    case TAG_TYPE_SRATIONAL:
        dumpVector(strm, getValueAsRational());
        break;
    case TAG_TYPE_UNDEFINED:
    {
        strm << "[";
        for (size_t i = 0; i < value_raw.size(); ++i) {
            if (i) strm << " ";
            strm << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<int>(value_raw[i]);
        }
        strm << "]" << std::dec;
        break;
    }
    default:
        break;
    }

    return strm;
}

} //namespace cv
