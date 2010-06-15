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
// Copyright (C) 2008, Willow Garage Inc., all rights reserved.
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

#include "precomp.hpp"

namespace cv
{
	

void write(FileStorage& fs, const string& objname, const vector<KeyPoint>& keypoints)
{
    WriteStructContext ws(fs, objname, CV_NODE_SEQ + CV_NODE_FLOW);
    
    int i, npoints = (int)keypoints.size();
    for( i = 0; i < npoints; i++ )
    {
        const KeyPoint& kpt = keypoints[i];
        write(fs, kpt.pt.x);
        write(fs, kpt.pt.y);
        write(fs, kpt.size);
        write(fs, kpt.angle);
        write(fs, kpt.response);
        write(fs, kpt.octave);
    }
}


void read(const FileNode& node, vector<KeyPoint>& keypoints)
{
    keypoints.resize(0);
    FileNodeIterator it = node.begin(), it_end = node.end();
    for( ; it != it_end; )
    {
        KeyPoint kpt;
        it >> kpt.pt.x >> kpt.pt.y >> kpt.size >> kpt.angle >> kpt.response >> kpt.octave;
        keypoints.push_back(kpt);
    }
}
    

void KeyPoint::convert(const std::vector<KeyPoint>& keypoints, std::vector<Point2f>& points2f,
                       const vector<int>& keypointIndexes)
{
    if( keypointIndexes.empty() )
    {
        points2f.resize( keypoints.size() );
        for( size_t i = 0; i < keypoints.size(); i++ )
            points2f[i] = keypoints[i].pt;
    }
    else
    {
        points2f.resize( keypointIndexes.size() );
        for( size_t i = 0; i < keypointIndexes.size(); i++ )
        {
            int idx = keypointIndexes[i];
            if( idx >= 0 )
                points2f[i] = keypoints[idx].pt;
            else
            {
                CV_Error( CV_StsBadArg, "keypointIndexes has element < 0. TODO: process this case" );
                //points2f[i] = Point2f(-1, -1);
            }
        }
    }
}
    
void KeyPoint::convert( const std::vector<Point2f>& points2f, std::vector<KeyPoint>& keypoints,
                        float size, float response, int octave, int class_id )
{
    for( size_t i = 0; i < points2f.size(); i++ )
        keypoints[i] = KeyPoint(points2f[i], size, -1, response, octave, class_id);
}

}
