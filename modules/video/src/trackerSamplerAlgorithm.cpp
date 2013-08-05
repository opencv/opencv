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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

/*
 *  TrackerSamplerAlgorithm
 */

TrackerSamplerAlgorithm::~TrackerSamplerAlgorithm()
{

}

bool TrackerSamplerAlgorithm::sampling( const Mat& image, Rect boundingBox, std::vector<Mat>& sample )
{
	if( image.empty() )
		return false;

	return samplingImpl( image, boundingBox, sample );
}


Ptr<TrackerSamplerAlgorithm> TrackerSamplerAlgorithm::create( const String& trackerSamplerType )
{
	if( trackerSamplerType.find("CSC") == 0 )
	{
		return new TrackerSamplerCSC();
	}

	if( trackerSamplerType.find("CS") == 0 )
	{
		return new TrackerSamplerCS();
	}

	CV_Error(-1, "Tracker sampler algorithm type not supported");
	return 0;
}

String TrackerSamplerAlgorithm::getClassName() const
{
	return className;
}

/**
 * TrackerSamplerCSC
 */

/**
 * Parameters
 */

TrackerSamplerCSC::Params::Params()
{
	initInRad = 3;
	initMaxNegNum = 65;
	searchWinSize = 25;
	trackInPosRad = 4;
	trackMaxNegNum = 65;
	trackMaxPosNum = 100000;

}

TrackerSamplerCSC::TrackerSamplerCSC( const TrackerSamplerCSC::Params &parameters ) :
params(parameters)
{
	className = "CSC";
	mode = MODE_INIT_POS;
}

TrackerSamplerCSC::~TrackerSamplerCSC()
{

}

bool TrackerSamplerCSC::samplingImpl( const Mat& image, Rect boundingBox, std::vector<Mat>& sample  )
{
	float inrad = 0;
	float outrad = 0;
	int maxnum = 0;

	switch ( mode ) {
		case MODE_INIT_POS:
			inrad = params.initInRad;
			sample = sampleImage( image, boundingBox.x, boundingBox.y, boundingBox.width,
					boundingBox.height, inrad );
			break;
		case MODE_INIT_NEG:
			inrad = 2.0f * params.searchWinSize;
			outrad = 1.5f * params.initInRad;
			maxnum = params.initMaxNegNum;
			sample = sampleImage( image, boundingBox.x, boundingBox.y, boundingBox.width,
					boundingBox.height, inrad, outrad, maxnum );
			break;
		case MODE_TRACK_POS:
			inrad = params.trackInPosRad;
			outrad = 0;
			maxnum = params.trackMaxPosNum;
			sample = sampleImage( image, boundingBox.x, boundingBox.y, boundingBox.width,
					boundingBox.height, inrad, outrad, maxnum );
			break;
		case MODE_TRACK_NEG:
			inrad = 1.5f * params.searchWinSize;
			outrad = params.trackInPosRad + 5;
			maxnum = params.trackMaxNegNum;
			sample = sampleImage( image, boundingBox.x, boundingBox.y, boundingBox.width,
					boundingBox.height, inrad, outrad, maxnum );
			break;
		case MODE_DETECT:
			inrad = params.searchWinSize;
			sample = sampleImage( image, boundingBox.x, boundingBox.y, boundingBox.width,
					boundingBox.height, inrad );
			break;
		default:
			inrad = params.initInRad;
			sample = sampleImage( image, boundingBox.x, boundingBox.y, boundingBox.width,
					boundingBox.height, inrad );
			break;
	}
	return false;
}

void TrackerSamplerCSC::setMode( int samplingMode )
{
	mode = samplingMode;
}

std::vector<Mat> TrackerSamplerCSC::sampleImage( const Mat& img, int x, int y, int w,
            int h, float inrad, float outrad, int maxnum )
{
	int rowsz = img.rows - h - 1;
	int colsz = img.cols - w - 1;
	float inradsq = inrad * inrad;
	float outradsq = outrad * outrad;
	int dist;

	uint minrow = max( 0, (int) y - (int) inrad );
	uint maxrow = min( (int) rowsz - 1, (int) y + (int) inrad );
	uint mincol = max( 0, (int) x - (int) inrad );
	uint maxcol = min( (int) colsz - 1, (int) x + (int) inrad );

	//fprintf(stderr,"inrad=%f minrow=%d maxrow=%d mincol=%d maxcol=%d\n",inrad,minrow,maxrow,mincol,maxcol);

	std::vector<Mat> samples;
	samples.resize( ( maxrow - minrow + 1 ) * ( maxcol - mincol + 1 ) );
	int i = 0;

	float prob = ( (float) (maxnum) ) / samples.size();

	for ( int r = minrow; r <= int( maxrow ); r++ )
		for ( int c = mincol; c <= int( maxcol ); c++ )
		{
			dist = ( y - r ) * ( y - r ) + ( x - c ) * ( x - c );
			if ( TrackerMIL::getRandFloat() < prob && dist < inradsq && dist >= outradsq )
			{
				samples[i] = img( Rect( c, r, w, h ) );
				i++;
			}
		}

	samples.resize( min( i, maxnum ) );
	return samples;
};

/**
 * TrackerSamplerCS
 */
TrackerSamplerCS::TrackerSamplerCS()
{
	className = "CS";
}

TrackerSamplerCS::~TrackerSamplerCS()
{

}

bool TrackerSamplerCS::samplingImpl( const Mat& image, Rect boundingBox, std::vector<Mat>& sample )
{
	return false;
}





} /* namespace cv */
