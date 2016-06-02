/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#ifndef __GRFMT_GDAL_HPP__
#define __GRFMT_GDAL_HPP__

/// OpenCV FMT Base Type
#include "grfmt_base.hpp"

/// Macro to make sure we specified GDAL in CMake
#ifdef HAVE_GDAL

/// C++ Libraries
#include <iostream>

/// Geospatial Data Abstraction Library
#include <cpl_conv.h>
#include <gdal_priv.h>
#include <gdal.h>


/// Start of CV Namespace
namespace cv {

/**
 * Convert GDAL Pixel Range to OpenCV Pixel Range
*/
double range_cast( const GDALDataType& gdalType,
                   const int& cvDepth,
                   const double& value );

/**
 * Convert GDAL Palette Interpretation to OpenCV Pixel Type
*/
int gdalPaletteInterpretation2OpenCV( GDALPaletteInterp const& paletteInterp,
                                      GDALDataType const& gdalType );

/**
 * Convert a GDAL Raster Type to OpenCV Type
*/
int gdal2opencv( const GDALDataType& gdalType, const int& channels );

/**
 * Write an image to pixel
*/
void write_pixel( const double& pixelValue,
                  GDALDataType const& gdalType,
                  const int& gdalChannels,
                  Mat& image,
                  const int& row,
                  const int& col,
                  const int& channel );

/**
 * Write a color table pixel to the image
*/
void write_ctable_pixel( const double& pixelValue,
                         const GDALDataType& gdalType,
                         const GDALColorTable* gdalColorTable,
                         Mat& image,
                         const int& y,
                         const int& x,
                         const int& c );

/**
 * Loader for GDAL
*/
class GdalDecoder : public BaseImageDecoder{

    public:

        /**
         * Default Constructor
        */
        GdalDecoder();

        /**
         * Destructor
        */
        ~GdalDecoder();

        /**
         * Read image data
        */
        bool readData( Mat& img );

        /**
         * Read the image header
        */
        bool readHeader();

        /**
         * Close the module
        */
        void close();

        /**
         * Create a new decoder
        */
        ImageDecoder newDecoder() const;

        /**
         * Test the file signature
         *
         * In general, this should be avoided as the user should specifically request GDAL.
         * The reason is that GDAL tends to overlap with other image formats and it is probably
         * safer to use other formats first.
        */
        virtual bool checkSignature( const String& signature ) const;

    protected:

        /// GDAL Dataset
        GDALDataset* m_dataset;

        /// GDAL Driver
        GDALDriver* m_driver;

        /// Check if we are reading from a color table
        bool hasColorTable;

}; /// End of GdalDecoder Class

} /// End of Namespace cv

#endif/*HAVE_GDAL*/

#endif/*__GRFMT_GDAL_HPP__*/
