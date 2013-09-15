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

#ifndef __OPENCV_GPUIMGPROC_HPP__
#define __OPENCV_GPUIMGPROC_HPP__

#ifndef __cplusplus
#  error gpuimgproc.hpp header must be compiled as C++
#endif

#include "opencv2/core/gpu.hpp"
#include "opencv2/imgproc.hpp"

namespace cv { namespace gpu {

/////////////////////////// Color Processing ///////////////////////////

//! converts image from one color space to another
CV_EXPORTS void cvtColor(InputArray src, OutputArray dst, int code, int dcn = 0, Stream& stream = Stream::Null());

enum
{
    // Bayer Demosaicing (Malvar, He, and Cutler)
    COLOR_BayerBG2BGR_MHT = 256,
    COLOR_BayerGB2BGR_MHT = 257,
    COLOR_BayerRG2BGR_MHT = 258,
    COLOR_BayerGR2BGR_MHT = 259,

    COLOR_BayerBG2RGB_MHT = COLOR_BayerRG2BGR_MHT,
    COLOR_BayerGB2RGB_MHT = COLOR_BayerGR2BGR_MHT,
    COLOR_BayerRG2RGB_MHT = COLOR_BayerBG2BGR_MHT,
    COLOR_BayerGR2RGB_MHT = COLOR_BayerGB2BGR_MHT,

    COLOR_BayerBG2GRAY_MHT = 260,
    COLOR_BayerGB2GRAY_MHT = 261,
    COLOR_BayerRG2GRAY_MHT = 262,
    COLOR_BayerGR2GRAY_MHT = 263
};
CV_EXPORTS void demosaicing(InputArray src, OutputArray dst, int code, int dcn = -1, Stream& stream = Stream::Null());

//! swap channels
//! dstOrder - Integer array describing how channel values are permutated. The n-th entry
//!            of the array contains the number of the channel that is stored in the n-th channel of
//!            the output image. E.g. Given an RGBA image, aDstOrder = [3,2,1,0] converts this to ABGR
//!            channel order.
CV_EXPORTS void swapChannels(InputOutputArray image, const int dstOrder[4], Stream& stream = Stream::Null());

//! Routines for correcting image color gamma
CV_EXPORTS void gammaCorrection(InputArray src, OutputArray dst, bool forward = true, Stream& stream = Stream::Null());

enum { ALPHA_OVER, ALPHA_IN, ALPHA_OUT, ALPHA_ATOP, ALPHA_XOR, ALPHA_PLUS, ALPHA_OVER_PREMUL, ALPHA_IN_PREMUL, ALPHA_OUT_PREMUL,
       ALPHA_ATOP_PREMUL, ALPHA_XOR_PREMUL, ALPHA_PLUS_PREMUL, ALPHA_PREMUL};

//! Composite two images using alpha opacity values contained in each image
//! Supports CV_8UC4, CV_16UC4, CV_32SC4 and CV_32FC4 types
CV_EXPORTS void alphaComp(InputArray img1, InputArray img2, OutputArray dst, int alpha_op, Stream& stream = Stream::Null());

////////////////////////////// Histogram ///////////////////////////////

//! Calculates histogram for 8u one channel image
//! Output hist will have one row, 256 cols and CV32SC1 type.
CV_EXPORTS void calcHist(InputArray src, OutputArray hist, Stream& stream = Stream::Null());

//! normalizes the grayscale image brightness and contrast by normalizing its histogram
CV_EXPORTS void equalizeHist(InputArray src, OutputArray dst, InputOutputArray buf, Stream& stream = Stream::Null());

static inline void equalizeHist(InputArray src, OutputArray dst, Stream& stream = Stream::Null())
{
    GpuMat buf;
    gpu::equalizeHist(src, dst, buf, stream);
}

class CV_EXPORTS CLAHE : public cv::CLAHE
{
public:
    using cv::CLAHE::apply;
    virtual void apply(InputArray src, OutputArray dst, Stream& stream) = 0;
};
CV_EXPORTS Ptr<gpu::CLAHE> createCLAHE(double clipLimit = 40.0, Size tileGridSize = Size(8, 8));

//! Compute levels with even distribution. levels will have 1 row and nLevels cols and CV_32SC1 type.
CV_EXPORTS void evenLevels(OutputArray levels, int nLevels, int lowerLevel, int upperLevel);

//! Calculates histogram with evenly distributed bins for signle channel source.
//! Supports CV_8UC1, CV_16UC1 and CV_16SC1 source types.
//! Output hist will have one row and histSize cols and CV_32SC1 type.
CV_EXPORTS void histEven(InputArray src, OutputArray hist, InputOutputArray buf, int histSize, int lowerLevel, int upperLevel, Stream& stream = Stream::Null());

static inline void histEven(InputArray src, OutputArray hist, int histSize, int lowerLevel, int upperLevel, Stream& stream = Stream::Null())
{
    GpuMat buf;
    gpu::histEven(src, hist, buf, histSize, lowerLevel, upperLevel, stream);
}

//! Calculates histogram with evenly distributed bins for four-channel source.
//! All channels of source are processed separately.
//! Supports CV_8UC4, CV_16UC4 and CV_16SC4 source types.
//! Output hist[i] will have one row and histSize[i] cols and CV_32SC1 type.
CV_EXPORTS void histEven(InputArray src, GpuMat hist[4], InputOutputArray buf, int histSize[4], int lowerLevel[4], int upperLevel[4], Stream& stream = Stream::Null());

static inline void histEven(InputArray src, GpuMat hist[4], int histSize[4], int lowerLevel[4], int upperLevel[4], Stream& stream = Stream::Null())
{
    GpuMat buf;
    gpu::histEven(src, hist, buf, histSize, lowerLevel, upperLevel, stream);
}

//! Calculates histogram with bins determined by levels array.
//! levels must have one row and CV_32SC1 type if source has integer type or CV_32FC1 otherwise.
//! Supports CV_8UC1, CV_16UC1, CV_16SC1 and CV_32FC1 source types.
//! Output hist will have one row and (levels.cols-1) cols and CV_32SC1 type.
CV_EXPORTS void histRange(InputArray src, OutputArray hist, InputArray levels, InputOutputArray buf, Stream& stream = Stream::Null());

static inline void histRange(InputArray src, OutputArray hist, InputArray levels, Stream& stream = Stream::Null())
{
    GpuMat buf;
    gpu::histRange(src, hist, levels, buf, stream);
}

//! Calculates histogram with bins determined by levels array.
//! All levels must have one row and CV_32SC1 type if source has integer type or CV_32FC1 otherwise.
//! All channels of source are processed separately.
//! Supports CV_8UC4, CV_16UC4, CV_16SC4 and CV_32FC4 source types.
//! Output hist[i] will have one row and (levels[i].cols-1) cols and CV_32SC1 type.
CV_EXPORTS void histRange(InputArray src, GpuMat hist[4], const GpuMat levels[4], InputOutputArray buf, Stream& stream = Stream::Null());

static inline void histRange(InputArray src, GpuMat hist[4], const GpuMat levels[4], Stream& stream = Stream::Null())
{
    GpuMat buf;
    gpu::histRange(src, hist, levels, buf, stream);
}

//////////////////////////////// Canny ////////////////////////////////

class CV_EXPORTS CannyEdgeDetector : public Algorithm
{
public:
    virtual void detect(InputArray image, OutputArray edges) = 0;
    virtual void detect(InputArray dx, InputArray dy, OutputArray edges) = 0;

    virtual void setLowThreshold(double low_thresh) = 0;
    virtual double getLowThreshold() const = 0;

    virtual void setHighThreshold(double high_thresh) = 0;
    virtual double getHighThreshold() const = 0;

    virtual void setAppertureSize(int apperture_size) = 0;
    virtual int getAppertureSize() const = 0;

    virtual void setL2Gradient(bool L2gradient) = 0;
    virtual bool getL2Gradient() const = 0;
};

CV_EXPORTS Ptr<CannyEdgeDetector> createCannyEdgeDetector(double low_thresh, double high_thresh, int apperture_size = 3, bool L2gradient = false);

/////////////////////////// Hough Transform ////////////////////////////

//////////////////////////////////////
// HoughLines

class CV_EXPORTS HoughLinesDetector : public Algorithm
{
public:
    virtual void detect(InputArray src, OutputArray lines) = 0;
    virtual void downloadResults(InputArray d_lines, OutputArray h_lines, OutputArray h_votes = noArray()) = 0;

    virtual void setRho(float rho) = 0;
    virtual float getRho() const = 0;

    virtual void setTheta(float theta) = 0;
    virtual float getTheta() const = 0;

    virtual void setThreshold(int threshold) = 0;
    virtual int getThreshold() const = 0;

    virtual void setDoSort(bool doSort) = 0;
    virtual bool getDoSort() const = 0;

    virtual void setMaxLines(int maxLines) = 0;
    virtual int getMaxLines() const = 0;
};

CV_EXPORTS Ptr<HoughLinesDetector> createHoughLinesDetector(float rho, float theta, int threshold, bool doSort = false, int maxLines = 4096);


//////////////////////////////////////
// HoughLinesP

//! finds line segments in the black-n-white image using probabalistic Hough transform
class CV_EXPORTS HoughSegmentDetector : public Algorithm
{
public:
    virtual void detect(InputArray src, OutputArray lines) = 0;

    virtual void setRho(float rho) = 0;
    virtual float getRho() const = 0;

    virtual void setTheta(float theta) = 0;
    virtual float getTheta() const = 0;

    virtual void setMinLineLength(int minLineLength) = 0;
    virtual int getMinLineLength() const = 0;

    virtual void setMaxLineGap(int maxLineGap) = 0;
    virtual int getMaxLineGap() const = 0;

    virtual void setMaxLines(int maxLines) = 0;
    virtual int getMaxLines() const = 0;
};

CV_EXPORTS Ptr<HoughSegmentDetector> createHoughSegmentDetector(float rho, float theta, int minLineLength, int maxLineGap, int maxLines = 4096);

//////////////////////////////////////
// HoughCircles

class CV_EXPORTS HoughCirclesDetector : public Algorithm
{
public:
    virtual void detect(InputArray src, OutputArray circles) = 0;

    virtual void setDp(float dp) = 0;
    virtual float getDp() const = 0;

    virtual void setMinDist(float minDist) = 0;
    virtual float getMinDist() const = 0;

    virtual void setCannyThreshold(int cannyThreshold) = 0;
    virtual int getCannyThreshold() const = 0;

    virtual void setVotesThreshold(int votesThreshold) = 0;
    virtual int getVotesThreshold() const = 0;

    virtual void setMinRadius(int minRadius) = 0;
    virtual int getMinRadius() const = 0;

    virtual void setMaxRadius(int maxRadius) = 0;
    virtual int getMaxRadius() const = 0;

    virtual void setMaxCircles(int maxCircles) = 0;
    virtual int getMaxCircles() const = 0;
};

CV_EXPORTS Ptr<HoughCirclesDetector> createHoughCirclesDetector(float dp, float minDist, int cannyThreshold, int votesThreshold, int minRadius, int maxRadius, int maxCircles = 4096);

//////////////////////////////////////
// GeneralizedHough

//! Ballard, D.H. (1981). Generalizing the Hough transform to detect arbitrary shapes. Pattern Recognition 13 (2): 111-122.
//! Detects position only without traslation and rotation
CV_EXPORTS Ptr<GeneralizedHoughBallard> createGeneralizedHoughBallard();

//! Guil, N., González-Linares, J.M. and Zapata, E.L. (1999). Bidimensional shape detection using an invariant approach. Pattern Recognition 32 (6): 1025-1038.
//! Detects position, traslation and rotation
CV_EXPORTS Ptr<GeneralizedHoughGuil> createGeneralizedHoughGuil();

////////////////////////// Corners Detection ///////////////////////////

class CV_EXPORTS CornernessCriteria : public Algorithm
{
public:
    virtual void compute(InputArray src, OutputArray dst, Stream& stream = Stream::Null()) = 0;
};

//! computes Harris cornerness criteria at each image pixel
CV_EXPORTS Ptr<CornernessCriteria> createHarrisCorner(int srcType, int blockSize, int ksize, double k, int borderType = BORDER_REFLECT101);

//! computes minimum eigen value of 2x2 derivative covariation matrix at each pixel - the cornerness criteria
CV_EXPORTS Ptr<CornernessCriteria> createMinEigenValCorner(int srcType, int blockSize, int ksize, int borderType = BORDER_REFLECT101);

////////////////////////// Corners Detection ///////////////////////////

class CV_EXPORTS CornersDetector : public Algorithm
{
public:
    //! return 1 rows matrix with CV_32FC2 type
    virtual void detect(InputArray image, OutputArray corners, InputArray mask = noArray()) = 0;
};

CV_EXPORTS Ptr<CornersDetector> createGoodFeaturesToTrackDetector(int srcType, int maxCorners = 1000, double qualityLevel = 0.01, double minDistance = 0.0,
                                                                  int blockSize = 3, bool useHarrisDetector = false, double harrisK = 0.04);

///////////////////////////// Mean Shift //////////////////////////////

//! Does mean shift filtering on GPU.
CV_EXPORTS void meanShiftFiltering(InputArray src, OutputArray dst, int sp, int sr,
                                   TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5, 1),
                                   Stream& stream = Stream::Null());

//! Does mean shift procedure on GPU.
CV_EXPORTS void meanShiftProc(InputArray src, OutputArray dstr, OutputArray dstsp, int sp, int sr,
                              TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5, 1),
                              Stream& stream = Stream::Null());

//! Does mean shift segmentation with elimination of small regions.
CV_EXPORTS void meanShiftSegmentation(InputArray src, OutputArray dst, int sp, int sr, int minsize,
                                      TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5, 1));

/////////////////////////// Match Template ////////////////////////////

//! computes the proximity map for the raster template and the image where the template is searched for
class CV_EXPORTS TemplateMatching : public Algorithm
{
public:
    virtual void match(InputArray image, InputArray templ, OutputArray result, Stream& stream = Stream::Null()) = 0;
};

CV_EXPORTS Ptr<TemplateMatching> createTemplateMatching(int srcType, int method, Size user_block_size = Size());

////////////////////////// Bilateral Filter ///////////////////////////

//! Performa bilateral filtering of passsed image
CV_EXPORTS void bilateralFilter(InputArray src, OutputArray dst, int kernel_size, float sigma_color, float sigma_spatial,
                                int borderMode = BORDER_DEFAULT, Stream& stream = Stream::Null());

///////////////////////////// Blending ////////////////////////////////

//! performs linear blending of two images
//! to avoid accuracy errors sum of weigths shouldn't be very close to zero
CV_EXPORTS void blendLinear(InputArray img1, InputArray img2, InputArray weights1, InputArray weights2,
                            OutputArray result, Stream& stream = Stream::Null());

}} // namespace cv { namespace gpu {

#endif /* __OPENCV_GPUIMGPROC_HPP__ */
