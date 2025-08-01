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

    size_t        getNumDirEntry( const size_t offsetNumDir ) const;
    uint32_t      getStartOffset() const;
    ExifTagId     getExifTagId( const size_t offset ) const;
    uint16_t      getU16( const size_t offset ) const;
    uint32_t      getU32( const size_t offset ) const;
    urational64_t getURational(const size_t offset) const;
    srational64_t getSRational(const size_t offset) const;
    std::string   getString(const size_t offset) const;
    uint16_t      getOrientation( const size_t offset ) const;
    int           getFormat() const;
    ExifEntry     parseExifEntry( const size_t offset );



private:
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
