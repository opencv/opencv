// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


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

enum ExifEndianness
{
    INTEL = 0x49,
    MOTO = 0x4D,
    NONE = 0x00
};

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

    bool parseExif(const unsigned char* data, const size_t size);
    bool parseExif(const unsigned char* data, const size_t size, std::vector< std::vector<ExifEntry> >& exif_entries);
    /**
     * @brief Get tag info by tag number
     *
     * @param [in] tag The tag number
     * @return ExifEntru_t structure. Caller has to know what tag it calls in order to extract proper field from the structure ExifEntry_t
     */
    ExifEntry getEntrybyTagId( const ExifTagId tag ) const;

    /**
     * @brief Get the whole exif buffer
     */
    const std::vector<unsigned char>& getData() const;

private:
    std::vector<unsigned char> m_data;
    std::map<int, ExifEntry > m_exif;
    int m_format;

    void parseExif();
    bool checkTagMark() const;

    size_t      getNumDirEntry( const size_t offsetNumDir ) const;
    uint32_t    getStartOffset() const;
    ExifTagId   getExifTagId( const size_t offset ) const;
    uint16_t    getU16( const size_t offset ) const;
    uint32_t    getU32( const size_t offset ) const;
    std::string getString(const size_t offset) const;
    uint16_t    getOrientation( const size_t offset ) const;
    int         getFormat() const;
    ExifEntry   parseExifEntry( const size_t offset );
    std::vector<urational64_t> getURational(const size_t offset) const;
    std::vector<srational64_t> getSRational(const size_t offset) const;
    uint32_t extractIFDOffset(const ExifEntry& entry) const;


private:
    template <typename RationalT, typename IntReader>
    std::vector<RationalT> getRational(size_t offset, IntReader readInt32) const;

    static const uint16_t tagMarkRequired = 0x2A;

    //max size of data in tag.
    //'DDDDDDDD' contains the value of that Tag. If its size is over 4bytes,
    //'DDDDDDDD' contains the offset to data stored address.
    static const size_t maxDataSize = 4;

    //bytes per tag field
    static const size_t tiffFieldSize = 12;
};

}

#endif /* _OPENCV_EXIF_HPP_ */
