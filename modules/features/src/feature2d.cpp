/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
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

namespace cv
{

using std::vector;

Feature2D::~Feature2D() {}

/*
 * Detect keypoints in an image.
 * image        The image.
 * keypoints    The detected keypoints.
 * mask         Mask specifying where to look for keypoints (optional). Must be a char
 *              matrix with non-zero values in the region of interest.
 */
void Feature2D::detect( InputArray image,
                        std::vector<KeyPoint>& keypoints,
                        InputArray mask )
{
    CV_INSTRUMENT_REGION();

    if( image.empty() )
    {
        keypoints.clear();
        return;
    }
    detectAndCompute(image, mask, keypoints, noArray(), false);
}


void Feature2D::detect( InputArrayOfArrays images,
                        std::vector<std::vector<KeyPoint> >& keypoints,
                        InputArrayOfArrays masks )
{
    CV_INSTRUMENT_REGION();

    int nimages = (int)images.total();

    if (!masks.empty())
    {
        CV_Assert(masks.total() == (size_t)nimages);
    }

    keypoints.resize(nimages);

    if (images.isMatVector())
    {
       for (int i = 0; i < nimages; i++)
       {
           detect(images.getMat(i), keypoints[i], masks.empty() ? noArray() : masks.getMat(i));
       }
    }
    else
    {
        // assume UMats
        for (int i = 0; i < nimages; i++)
        {
            detect(images.getUMat(i), keypoints[i], masks.empty() ? noArray() : masks.getUMat(i));
        }
    }


}

/*
 * Compute the descriptors for a set of keypoints in an image.
 * image        The image.
 * keypoints    The input keypoints. Keypoints for which a descriptor cannot be computed are removed.
 * descriptors  Copmputed descriptors. Row i is the descriptor for keypoint i.
 */
void Feature2D::compute( InputArray image,
                         std::vector<KeyPoint>& keypoints,
                         OutputArray descriptors )
{
    CV_INSTRUMENT_REGION();

    if( image.empty() )
    {
        descriptors.release();
        return;
    }
    detectAndCompute(image, noArray(), keypoints, descriptors, true);
}

void Feature2D::compute( InputArrayOfArrays images,
                         std::vector<std::vector<KeyPoint> >& keypoints,
                         OutputArrayOfArrays descriptors )
{
    CV_INSTRUMENT_REGION();

    if( !descriptors.needed() )
        return;

    int nimages = (int)images.total();

    CV_Assert( keypoints.size() == (size_t)nimages );
    // resize descriptors to appropriate size and compute
    if (descriptors.isMatVector())
    {
        vector<Mat>& vec = *descriptors.getObj<vector<Mat>>();
        vec.resize(nimages);
        for (int i = 0; i < nimages; i++)
        {
            compute(images.getMat(i), keypoints[i], vec[i]);
        }
    }
    else if (descriptors.isUMatVector())
    {
        vector<UMat>& vec = *descriptors.getObj<vector<UMat>>();
        vec.resize(nimages);
        for (int i = 0; i < nimages; i++)
        {
            compute(images.getUMat(i), keypoints[i], vec[i]);
        }
    }
    else
    {
        CV_Error(Error::StsBadArg, "descriptors must be vector<Mat> or vector<UMat>");
    }
}


/* Detects keypoints and computes the descriptors */
void Feature2D::detectAndCompute( InputArray, InputArray,
                                  std::vector<KeyPoint>&,
                                  OutputArray,
                                  bool )
{
    CV_INSTRUMENT_REGION();

    CV_Error(Error::StsNotImplemented, "");
}

void Feature2D::write( const String& fileName ) const
{
    FileStorage fs(fileName, FileStorage::WRITE);
    write(fs);
}

void Feature2D::read( const String& fileName )
{
    FileStorage fs(fileName, FileStorage::READ);
    read(fs.root());
}

void Feature2D::write( FileStorage&) const
{
}

void Feature2D::read( const FileNode&)
{
}

int Feature2D::descriptorSize() const
{
    return 0;
}

int Feature2D::descriptorType() const
{
    return CV_32F;
}

int Feature2D::defaultNorm() const
{
    int tp = descriptorType();
    return tp == CV_8U ? NORM_HAMMING : NORM_L2;
}

// Return true if detector object is empty
bool Feature2D::empty() const
{
    return true;
}

String Feature2D::getDefaultName() const
{
    return "Feature2D";
}

}
