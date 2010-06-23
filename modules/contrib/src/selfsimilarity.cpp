// This is based on Rainer Lienhart contribution. Below is the original copyright:
//
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                University of Augsburg License Agreement
//                For Open Source MultiMedia Computing (MMC) Library
//
// Copyright (C) 2007, University of Augsburg, Germany, all rights reserved.
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
//   * The name of University of Augsburg, Germany may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the University of Augsburg, Germany or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//   Author:    Rainer Lienhart
//              email: Rainer.Lienhart@informatik.uni-augsburg.de
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

// Please cite the following two papers:
// 1. Shechtman, E., Irani, M.:
//		Matching local self-similarities across images and videos.
// 		CVPR, (2007)
// 2. Eva Horster, Thomas Greif, Rainer Lienhart, Malcolm Slaney.
//		Comparing Local Feature Descriptors in pLSA-Based Image Models.
//		30th Annual Symposium of the German Association for
//      Pattern Recognition (DAGM) 2008, Munich, Germany, June 2008.

#include "precomp.hpp"

namespace cv
{

SelfSimDescriptor::SelfSimDescriptor()
{
    smallSize = DEFAULT_SMALL_SIZE;
    largeSize = DEFAULT_LARGE_SIZE;
    numberOfAngles = DEFAULT_NUM_ANGLES;
    startDistanceBucket = DEFAULT_START_DISTANCE_BUCKET;
    numberOfDistanceBuckets = DEFAULT_NUM_DISTANCE_BUCKETS;
}

SelfSimDescriptor::SelfSimDescriptor(int _ssize, int _lsize,
                                     int _startDistanceBucket,
                                     int _numberOfDistanceBuckets, int _numberOfAngles)
{
    smallSize = _ssize;
    largeSize = _lsize;
    startDistanceBucket = _startDistanceBucket;
    numberOfDistanceBuckets = _numberOfDistanceBuckets;
    numberOfAngles = _numberOfAngles;
}

SelfSimDescriptor::SelfSimDescriptor(const SelfSimDescriptor& ss)
{
    smallSize = ss.smallSize;
    largeSize = ss.largeSize;
    startDistanceBucket = ss.startDistanceBucket;
    numberOfDistanceBuckets = ss.numberOfDistanceBuckets;
    numberOfAngles = ss.numberOfAngles;
}

SelfSimDescriptor::~SelfSimDescriptor()
{
}

SelfSimDescriptor& SelfSimDescriptor::operator = (const SelfSimDescriptor& ss)
{
    if( this != &ss )
    {
        smallSize = ss.smallSize;
        largeSize = ss.largeSize;
        startDistanceBucket = ss.startDistanceBucket;
        numberOfDistanceBuckets = ss.numberOfDistanceBuckets;
        numberOfAngles = ss.numberOfAngles;
    }
    return *this;
}

size_t SelfSimDescriptor::getDescriptorSize() const
{
    return numberOfAngles*(numberOfDistanceBuckets - startDistanceBucket);
}

Size SelfSimDescriptor::getGridSize( Size imgSize, Size winStride ) const
{
    winStride.width = std::max(winStride.width, 1);
    winStride.height = std::max(winStride.height, 1);
    int border = largeSize/2 + smallSize/2;
    return Size(std::max(imgSize.width - border*2 + winStride.width - 1, 0)/winStride.width,
                std::max(imgSize.height - border*2 + winStride.height - 1, 0)/winStride.height);
}

// TODO: optimized with SSE2
void SelfSimDescriptor::SSD(const Mat& img, Point pt, Mat& ssd) const
{
	int x, y, dx, dy, r0 = largeSize/2, r1 = smallSize/2;
    int step = img.step;
    for( y = -r0; y <= r0; y++ )
    {
        float* sptr = ssd.ptr<float>(y+r0) + r0;
        for( x = -r0; x <= r0; x++ )
        {
            int sum = 0;
            const uchar* src0 = img.ptr<uchar>(y + pt.y - r1) + x + pt.x;
            const uchar* src1 = img.ptr<uchar>(pt.y - r1) + pt.x;
            for( dy = -r1; dy <= r1; dy++, src0 += step, src1 += step )
                for( dx = -r1; dx <= r1; dx++ )
                {
                    int t = src0[dx] - src1[dx];
                    sum += t*t;
                }
            sptr[x] = (float)sum;
        }
    }
}


void SelfSimDescriptor::compute(const Mat& img, vector<float>& descriptors, Size winStride,
                                const vector<Point>& locations) const
{
    CV_Assert( img.depth() == CV_8U );  

    winStride.width = std::max(winStride.width, 1);
    winStride.height = std::max(winStride.height, 1);
    Size gridSize = getGridSize(img.size(), winStride);
    int i, nwindows = locations.empty() ? gridSize.width*gridSize.height : (int)locations.size();
    int border = largeSize/2 + smallSize/2;
    int fsize = (int)getDescriptorSize();
    vector<float> tempFeature(fsize+1);
    descriptors.resize(fsize*nwindows + 1);
    Mat ssd(largeSize, largeSize, CV_32F), mappingMask;
    computeLogPolarMapping(mappingMask);

#if 0 //def _OPENMP
    int nthreads = cvGetNumThreads();
    #pragma omp parallel for num_threads(nthreads)
#endif
    for( i = 0; i < nwindows; i++ )
    {
        Point pt;
        float* feature0 = &descriptors[fsize*i];
        float* feature = &tempFeature[0];
        int x, y, j;

        if( !locations.empty() )
        {
            pt = locations[i];
            if( pt.x < border || pt.x >= img.cols - border ||
                pt.y < border || pt.y >= img.rows - border )
            {
                for( j = 0; j < fsize; j++ )
                    feature0[j] = 0.f;
                continue;
            }
        }
        else
            pt = Point((i % gridSize.width)*winStride.width + border,
                       (i / gridSize.width)*winStride.height + border);

        SSD(img, pt, ssd);

	    // Determine in the local neighborhood the largest difference and use for normalization
	    float var_noise = 1000.f;
        for( y = -1; y <= 1 ; y++ )
		    for( x = -1 ; x <= 1 ; x++ )
                var_noise = std::max(var_noise, ssd.at<float>(largeSize/2+y, largeSize/2+x));

        for( j = 0; j <= fsize; j++ )
            feature[j] = FLT_MAX;

	    // Derive feature vector before exp(-x) computation
	    // Idea: for all  x,a >= 0, a=const.   we have:
	    //       max [ exp( -x / a) ] = exp ( -min(x) / a )
	    // Thus, determine min(ssd) and store in feature[...]
    	for( y = 0; y < ssd.rows; y++ )
        {
		    const schar *mappingMaskPtr = mappingMask.ptr<schar>(y);
		    const float *ssdPtr = ssd.ptr<float>(y);
		    for( x = 0 ; x < ssd.cols; x++ )
            {
                int index = mappingMaskPtr[x];
                feature[index] = std::min(feature[index], ssdPtr[x]);
		    }
	    }

        var_noise = -1.f/var_noise;
    	for( j = 0; j < fsize; j++ )
            feature0[j] = feature[j]*var_noise;
        Mat _f(1, fsize, CV_32F, feature0);
        cv::exp(_f, _f);
    }
}

void SelfSimDescriptor::computeLogPolarMapping(Mat& mappingMask) const
{
    mappingMask.create(largeSize, largeSize, CV_8S);

    // What we want is
    //		 log_m (radius) = numberOfDistanceBuckets
    //	<==> log_10 (radius) / log_10 (m) = numberOfDistanceBuckets
    //	<==> log_10 (radius) / numberOfDistanceBuckets = log_10 (m)
    //	<==> m = 10 ^ log_10(m) = 10 ^ [log_10 (radius) / numberOfDistanceBuckets]
    //
    int radius = largeSize/2, angleBucketSize = 360 / numberOfAngles;
    int fsize = (int)getDescriptorSize();
    double inv_log10m = (double)numberOfDistanceBuckets/log10((double)radius);

    for (int y=-radius ; y<=radius ; y++)
    {
        schar* mrow = mappingMask.ptr<schar>(y+radius);
        for (int x=-radius ; x<=radius ; x++)
        {
            int index = fsize;
            float dist = (float)std::sqrt((float)x*x + (float)y*y);
            int distNo = dist > 0 ? cvRound(log10(dist)*inv_log10m) : 0;
            if( startDistanceBucket <= distNo && distNo < numberOfDistanceBuckets )
            {
                float angle = std::atan2( (float)y, (float)x ) / (float)CV_PI * 180.0f;
                if (angle < 0) angle += 360.0f;
                int angleInt = (cvRound(angle) + angleBucketSize/2) % 360;
                int angleIndex = angleInt / angleBucketSize;
                index = (distNo-startDistanceBucket)*numberOfAngles + angleIndex;
            }
            mrow[x + radius] = saturate_cast<schar>(index);
        }
    }
}

}
