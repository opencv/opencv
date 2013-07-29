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
// Copyright (C) 2008-2012, Willow Garage Inc., all rights reserved.
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

#ifndef __OPENCV_PHOTO_HPP__
#define __OPENCV_PHOTO_HPP__

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

/*! \namespace cv
 Namespace where all the C++ OpenCV functionality resides
 */
namespace cv
{

//! the inpainting algorithm
enum
{
    INPAINT_NS    = 0, // Navier-Stokes algorithm
    INPAINT_TELEA = 1 // A. Telea algorithm
};

//! restores the damaged image areas using one of the available intpainting algorithms
CV_EXPORTS_W void inpaint( InputArray src, InputArray inpaintMask,
                           OutputArray dst, double inpaintRadius, int flags );


CV_EXPORTS_W void fastNlMeansDenoising( InputArray src, OutputArray dst, float h = 3,
                                        int templateWindowSize = 7, int searchWindowSize = 21);

CV_EXPORTS_W void fastNlMeansDenoisingColored( InputArray src, OutputArray dst,
                                               float h = 3, float hColor = 3,
                                               int templateWindowSize = 7, int searchWindowSize = 21);

CV_EXPORTS_W void fastNlMeansDenoisingMulti( InputArrayOfArrays srcImgs, OutputArray dst,
                                             int imgToDenoiseIndex, int temporalWindowSize,
                                             float h = 3, int templateWindowSize = 7, int searchWindowSize = 21);

CV_EXPORTS_W void fastNlMeansDenoisingColoredMulti( InputArrayOfArrays srcImgs, OutputArray dst,
                                                    int imgToDenoiseIndex, int temporalWindowSize,
                                                    float h = 3, float hColor = 3,
                                                    int templateWindowSize = 7, int searchWindowSize = 21);

CV_EXPORTS_W void makeHDR(InputArrayOfArrays srcImgs, const std::vector<float>& exp_times, OutputArray dst, Mat response = Mat());

CV_EXPORTS_W void exposureFusion(InputArrayOfArrays srcImgs, OutputArray dst, float wc = 1.0f, float ws = 1.0f, float we = 0.0f);

CV_EXPORTS_W void shiftMat(InputArray src, Point shift, OutputArray dst);

CV_EXPORTS_W Point getExpShift(InputArray img0, InputArray img1, int max_bits = 6, int exclude_range = 4);

CV_EXPORTS_W void estimateResponse(InputArrayOfArrays srcImgs, const std::vector<float>& exp_times, OutputArray dst, int samples = 50, float lambda = 10);

CV_EXPORTS_W void alignImages(InputArrayOfArrays src, std::vector<Mat>& dst);

class CV_EXPORTS_W Tonemap : public Algorithm
{
public:
	Tonemap(float gamma);
	virtual ~Tonemap();
	void process(InputArray src, OutputArray dst);
	static Ptr<Tonemap> create(const String& name);
protected:
	float gamma;
	Mat img;
	void linearMap();
	void gammaCorrection();

	virtual void tonemap() = 0;
};

class CV_EXPORTS_W TonemapLinear : public Tonemap
{
public:
	TonemapLinear(float gamma = 2.2f);
	AlgorithmInfo* info() const;
protected:
	void tonemap();
};

class CV_EXPORTS_W TonemapDrago : public Tonemap
{
public:
	TonemapDrago(float gamma = 2.2f, float bias = 0.85f);
	AlgorithmInfo* info() const;
protected:
	float bias;
	void tonemap();
};

class CV_EXPORTS_W TonemapDurand : public Tonemap
{
public:
	TonemapDurand(float gamma = 2.2f, float contrast = 4.0f, float sigma_color = 2.0f, float sigma_space = 2.0f);
	AlgorithmInfo* info() const;
protected:
	float contrast;
    float sigma_color;
    float sigma_space;
	void tonemap();
};

class CV_EXPORTS_W TonemapReinhardDevlin : public Tonemap
{
public:
	TonemapReinhardDevlin(float gamma = 2.2f, float intensity = 0.0f, float color_adapt = 0.0f, float light_adapt = 1.0f);
	AlgorithmInfo* info() const;
protected:
	float intensity;
    float color_adapt;
    float light_adapt;
	void tonemap();
};

} // cv

#endif
