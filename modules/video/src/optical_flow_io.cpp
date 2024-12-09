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
#include<iostream>
#include<fstream>

namespace cv {

const float FLOW_TAG_FLOAT = 202021.25f;
const char *FLOW_TAG_STRING = "PIEH";
CV_EXPORTS_W Mat readOpticalFlow( const String& path )
{
    Mat_<Point2f> flow;
    std::ifstream file(path.c_str(), std::ios_base::binary);
    if ( !file.good() )
        return Mat(); // no file - return empty matrix

    float tag;
    file.read((char*) &tag, sizeof(float));
    if ( tag != FLOW_TAG_FLOAT )
        return Mat();

    int width, height;

    file.read((char*) &width, 4);
    file.read((char*) &height, 4);

    flow.create(height, width);

    for ( int i = 0; i < flow.rows; ++i )
    {
        for ( int j = 0; j < flow.cols; ++j )
        {
            Point2f u;
            file.read((char*) &u.x, sizeof(float));
            file.read((char*) &u.y, sizeof(float));
            if ( !file.good() )
            {
                flow.release();
                return Mat();
            }

            flow(i, j) = u;
        }
    }
    file.close();
    return Mat(flow);
}

CV_EXPORTS_W bool writeOpticalFlow( const String& path, InputArray flow )
{
    const int nChannels = 2;

    Mat input = flow.getMat();
    if ( input.channels() != nChannels || input.depth() != CV_32F || path.length() == 0 )
        return false;

    std::ofstream file(path.c_str(), std::ofstream::binary);
    if ( !file.good() )
        return false;

    int nRows, nCols;

    nRows = (int) input.size().height;
    nCols = (int) input.size().width;

    const int headerSize = 12;
    char header[headerSize];
    memcpy(header, FLOW_TAG_STRING, 4);
    // size of ints is known - has been asserted in the current function
    memcpy(header + 4, reinterpret_cast<const char*>(&nCols), sizeof(nCols));
    memcpy(header + 8, reinterpret_cast<const char*>(&nRows), sizeof(nRows));
    file.write(header, headerSize);
    if ( !file.good() )
        return false;

    int row;
    char* p;
    for ( row = 0; row < nRows; row++ )
    {
        p = input.ptr<char>(row);
        file.write(p, nCols * nChannels * sizeof(float));
        if ( !file.good() )
            return false;
    }
    file.close();
    return true;
}

}
