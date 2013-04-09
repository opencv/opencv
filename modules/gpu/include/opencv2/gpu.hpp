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

#ifndef __OPENCV_GPU_HPP__
#define __OPENCV_GPU_HPP__

#ifndef SKIP_INCLUDES
#include <vector>
#include <memory>
#include <iosfwd>
#endif

#include "opencv2/core/gpumat.hpp"
#include "opencv2/gpuarithm.hpp"
#include "opencv2/gpufilters.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/features2d.hpp"

namespace cv { namespace gpu {
////////////////////////////// Image processing //////////////////////////////


enum { ALPHA_OVER, ALPHA_IN, ALPHA_OUT, ALPHA_ATOP, ALPHA_XOR, ALPHA_PLUS, ALPHA_OVER_PREMUL, ALPHA_IN_PREMUL, ALPHA_OUT_PREMUL,
       ALPHA_ATOP_PREMUL, ALPHA_XOR_PREMUL, ALPHA_PLUS_PREMUL, ALPHA_PREMUL};

//! Composite two images using alpha opacity values contained in each image
//! Supports CV_8UC4, CV_16UC4, CV_32SC4 and CV_32FC4 types
CV_EXPORTS void alphaComp(const GpuMat& img1, const GpuMat& img2, GpuMat& dst, int alpha_op, Stream& stream = Stream::Null());

//! DST[x,y] = SRC[xmap[x,y],ymap[x,y]]
//! supports only CV_32FC1 map type
CV_EXPORTS void remap(const GpuMat& src, GpuMat& dst, const GpuMat& xmap, const GpuMat& ymap,
                      int interpolation, int borderMode = BORDER_CONSTANT, Scalar borderValue = Scalar(),
                      Stream& stream = Stream::Null());

//! Does mean shift filtering on GPU.
CV_EXPORTS void meanShiftFiltering(const GpuMat& src, GpuMat& dst, int sp, int sr,
                                   TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5, 1),
                                   Stream& stream = Stream::Null());

//! Does mean shift procedure on GPU.
CV_EXPORTS void meanShiftProc(const GpuMat& src, GpuMat& dstr, GpuMat& dstsp, int sp, int sr,
                              TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5, 1),
                              Stream& stream = Stream::Null());

//! Does mean shift segmentation with elimination of small regions.
CV_EXPORTS void meanShiftSegmentation(const GpuMat& src, Mat& dst, int sp, int sr, int minsize,
                                      TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5, 1));

//! Does coloring of disparity image: [0..ndisp) -> [0..240, 1, 1] in HSV.
//! Supported types of input disparity: CV_8U, CV_16S.
//! Output disparity has CV_8UC4 type in BGRA format (alpha = 255).
CV_EXPORTS void drawColorDisp(const GpuMat& src_disp, GpuMat& dst_disp, int ndisp, Stream& stream = Stream::Null());

//! Reprojects disparity image to 3D space.
//! Supports CV_8U and CV_16S types of input disparity.
//! The output is a 3- or 4-channel floating-point matrix.
//! Each element of this matrix will contain the 3D coordinates of the point (x,y,z,1), computed from the disparity map.
//! Q is the 4x4 perspective transformation matrix that can be obtained with cvStereoRectify.
CV_EXPORTS void reprojectImageTo3D(const GpuMat& disp, GpuMat& xyzw, const Mat& Q, int dst_cn = 4, Stream& stream = Stream::Null());

//! converts image from one color space to another
CV_EXPORTS void cvtColor(const GpuMat& src, GpuMat& dst, int code, int dcn = 0, Stream& stream = Stream::Null());

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
CV_EXPORTS void demosaicing(const GpuMat& src, GpuMat& dst, int code, int dcn = -1, Stream& stream = Stream::Null());

//! swap channels
//! dstOrder - Integer array describing how channel values are permutated. The n-th entry
//!            of the array contains the number of the channel that is stored in the n-th channel of
//!            the output image. E.g. Given an RGBA image, aDstOrder = [3,2,1,0] converts this to ABGR
//!            channel order.
CV_EXPORTS void swapChannels(GpuMat& image, const int dstOrder[4], Stream& stream = Stream::Null());

//! Routines for correcting image color gamma
CV_EXPORTS void gammaCorrection(const GpuMat& src, GpuMat& dst, bool forward = true, Stream& stream = Stream::Null());

//! resizes the image
//! Supports INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA
CV_EXPORTS void resize(const GpuMat& src, GpuMat& dst, Size dsize, double fx=0, double fy=0, int interpolation = INTER_LINEAR, Stream& stream = Stream::Null());

//! warps the image using affine transformation
//! Supports INTER_NEAREST, INTER_LINEAR, INTER_CUBIC
CV_EXPORTS void warpAffine(const GpuMat& src, GpuMat& dst, const Mat& M, Size dsize, int flags = INTER_LINEAR,
    int borderMode = BORDER_CONSTANT, Scalar borderValue = Scalar(), Stream& stream = Stream::Null());

CV_EXPORTS void buildWarpAffineMaps(const Mat& M, bool inverse, Size dsize, GpuMat& xmap, GpuMat& ymap, Stream& stream = Stream::Null());

//! warps the image using perspective transformation
//! Supports INTER_NEAREST, INTER_LINEAR, INTER_CUBIC
CV_EXPORTS void warpPerspective(const GpuMat& src, GpuMat& dst, const Mat& M, Size dsize, int flags = INTER_LINEAR,
    int borderMode = BORDER_CONSTANT, Scalar borderValue = Scalar(), Stream& stream = Stream::Null());

CV_EXPORTS void buildWarpPerspectiveMaps(const Mat& M, bool inverse, Size dsize, GpuMat& xmap, GpuMat& ymap, Stream& stream = Stream::Null());

//! builds plane warping maps
CV_EXPORTS void buildWarpPlaneMaps(Size src_size, Rect dst_roi, const Mat &K, const Mat& R, const Mat &T, float scale,
                                   GpuMat& map_x, GpuMat& map_y, Stream& stream = Stream::Null());

//! builds cylindrical warping maps
CV_EXPORTS void buildWarpCylindricalMaps(Size src_size, Rect dst_roi, const Mat &K, const Mat& R, float scale,
                                         GpuMat& map_x, GpuMat& map_y, Stream& stream = Stream::Null());

//! builds spherical warping maps
CV_EXPORTS void buildWarpSphericalMaps(Size src_size, Rect dst_roi, const Mat &K, const Mat& R, float scale,
                                       GpuMat& map_x, GpuMat& map_y, Stream& stream = Stream::Null());

//! rotates an image around the origin (0,0) and then shifts it
//! supports INTER_NEAREST, INTER_LINEAR, INTER_CUBIC
//! supports 1, 3 or 4 channels images with CV_8U, CV_16U or CV_32F depth
CV_EXPORTS void rotate(const GpuMat& src, GpuMat& dst, Size dsize, double angle, double xShift = 0, double yShift = 0,
                       int interpolation = INTER_LINEAR, Stream& stream = Stream::Null());

//! copies 2D array to a larger destination array and pads borders with user-specifiable constant
CV_EXPORTS void copyMakeBorder(const GpuMat& src, GpuMat& dst, int top, int bottom, int left, int right, int borderType,
                               const Scalar& value = Scalar(), Stream& stream = Stream::Null());

//! computes the integral image
//! sum will have CV_32S type, but will contain unsigned int values
//! supports only CV_8UC1 source type
CV_EXPORTS void integral(const GpuMat& src, GpuMat& sum, Stream& stream = Stream::Null());
//! buffered version
CV_EXPORTS void integralBuffered(const GpuMat& src, GpuMat& sum, GpuMat& buffer, Stream& stream = Stream::Null());

//! computes squared integral image
//! result matrix will have 64F type, but will contain 64U values
//! supports source images of 8UC1 type only
CV_EXPORTS void sqrIntegral(const GpuMat& src, GpuMat& sqsum, Stream& stream = Stream::Null());

//! computes the standard deviation of integral images
//! supports only CV_32SC1 source type and CV_32FC1 sqr type
//! output will have CV_32FC1 type
CV_EXPORTS void rectStdDev(const GpuMat& src, const GpuMat& sqr, GpuMat& dst, const Rect& rect, Stream& stream = Stream::Null());

//! computes Harris cornerness criteria at each image pixel
CV_EXPORTS void cornerHarris(const GpuMat& src, GpuMat& dst, int blockSize, int ksize, double k, int borderType = BORDER_REFLECT101);
CV_EXPORTS void cornerHarris(const GpuMat& src, GpuMat& dst, GpuMat& Dx, GpuMat& Dy, int blockSize, int ksize, double k, int borderType = BORDER_REFLECT101);
CV_EXPORTS void cornerHarris(const GpuMat& src, GpuMat& dst, GpuMat& Dx, GpuMat& Dy, GpuMat& buf, int blockSize, int ksize, double k,
                             int borderType = BORDER_REFLECT101, Stream& stream = Stream::Null());

//! computes minimum eigen value of 2x2 derivative covariation matrix at each pixel - the cornerness criteria
CV_EXPORTS void cornerMinEigenVal(const GpuMat& src, GpuMat& dst, int blockSize, int ksize, int borderType=BORDER_REFLECT101);
CV_EXPORTS void cornerMinEigenVal(const GpuMat& src, GpuMat& dst, GpuMat& Dx, GpuMat& Dy, int blockSize, int ksize, int borderType=BORDER_REFLECT101);
CV_EXPORTS void cornerMinEigenVal(const GpuMat& src, GpuMat& dst, GpuMat& Dx, GpuMat& Dy, GpuMat& buf, int blockSize, int ksize,
    int borderType=BORDER_REFLECT101, Stream& stream = Stream::Null());

//! performs per-element multiplication of two full (not packed) Fourier spectrums
//! supports 32FC2 matrixes only (interleaved format)
CV_EXPORTS void mulSpectrums(const GpuMat& a, const GpuMat& b, GpuMat& c, int flags, bool conjB=false, Stream& stream = Stream::Null());

//! performs per-element multiplication of two full (not packed) Fourier spectrums
//! supports 32FC2 matrixes only (interleaved format)
CV_EXPORTS void mulAndScaleSpectrums(const GpuMat& a, const GpuMat& b, GpuMat& c, int flags, float scale, bool conjB=false, Stream& stream = Stream::Null());

//! Performs a forward or inverse discrete Fourier transform (1D or 2D) of floating point matrix.
//! Param dft_size is the size of DFT transform.
//!
//! If the source matrix is not continous, then additional copy will be done,
//! so to avoid copying ensure the source matrix is continous one. If you want to use
//! preallocated output ensure it is continuous too, otherwise it will be reallocated.
//!
//! Being implemented via CUFFT real-to-complex transform result contains only non-redundant values
//! in CUFFT's format. Result as full complex matrix for such kind of transform cannot be retrieved.
//!
//! For complex-to-real transform it is assumed that the source matrix is packed in CUFFT's format.
CV_EXPORTS void dft(const GpuMat& src, GpuMat& dst, Size dft_size, int flags=0, Stream& stream = Stream::Null());

struct CV_EXPORTS ConvolveBuf
{
    Size result_size;
    Size block_size;
    Size user_block_size;
    Size dft_size;
    int spect_len;

    GpuMat image_spect, templ_spect, result_spect;
    GpuMat image_block, templ_block, result_data;

    void create(Size image_size, Size templ_size);
    static Size estimateBlockSize(Size result_size, Size templ_size);
};


//! computes convolution (or cross-correlation) of two images using discrete Fourier transform
//! supports source images of 32FC1 type only
//! result matrix will have 32FC1 type
CV_EXPORTS void convolve(const GpuMat& image, const GpuMat& templ, GpuMat& result, bool ccorr = false);
CV_EXPORTS void convolve(const GpuMat& image, const GpuMat& templ, GpuMat& result, bool ccorr, ConvolveBuf& buf, Stream& stream = Stream::Null());

struct CV_EXPORTS MatchTemplateBuf
{
    Size user_block_size;
    GpuMat imagef, templf;
    std::vector<GpuMat> images;
    std::vector<GpuMat> image_sums;
    std::vector<GpuMat> image_sqsums;
};

//! computes the proximity map for the raster template and the image where the template is searched for
CV_EXPORTS void matchTemplate(const GpuMat& image, const GpuMat& templ, GpuMat& result, int method, Stream &stream = Stream::Null());

//! computes the proximity map for the raster template and the image where the template is searched for
CV_EXPORTS void matchTemplate(const GpuMat& image, const GpuMat& templ, GpuMat& result, int method, MatchTemplateBuf &buf, Stream& stream = Stream::Null());

//! smoothes the source image and downsamples it
CV_EXPORTS void pyrDown(const GpuMat& src, GpuMat& dst, Stream& stream = Stream::Null());

//! upsamples the source image and then smoothes it
CV_EXPORTS void pyrUp(const GpuMat& src, GpuMat& dst, Stream& stream = Stream::Null());

//! performs linear blending of two images
//! to avoid accuracy errors sum of weigths shouldn't be very close to zero
CV_EXPORTS void blendLinear(const GpuMat& img1, const GpuMat& img2, const GpuMat& weights1, const GpuMat& weights2,
                            GpuMat& result, Stream& stream = Stream::Null());

//! Performa bilateral filtering of passsed image
CV_EXPORTS void bilateralFilter(const GpuMat& src, GpuMat& dst, int kernel_size, float sigma_color, float sigma_spatial,
                                int borderMode = BORDER_DEFAULT, Stream& stream = Stream::Null());

//! Brute force non-local means algorith (slow but universal)
CV_EXPORTS void nonLocalMeans(const GpuMat& src, GpuMat& dst, float h, int search_window = 21, int block_size = 7, int borderMode = BORDER_DEFAULT, Stream& s = Stream::Null());

//! Fast (but approximate)version of non-local means algorith similar to CPU function (running sums technique)
class CV_EXPORTS FastNonLocalMeansDenoising
{
public:
    //! Simple method, recommended for grayscale images (though it supports multichannel images)
    void simpleMethod(const GpuMat& src, GpuMat& dst, float h, int search_window = 21, int block_size = 7, Stream& s = Stream::Null());

    //! Processes luminance and color components separatelly
    void labMethod(const GpuMat& src, GpuMat& dst, float h_luminance, float h_color, int search_window = 21, int block_size = 7, Stream& s = Stream::Null());

private:

    GpuMat buffer, extended_src_buffer;
    GpuMat lab, l, ab;
};

struct CV_EXPORTS CannyBuf
{
    void create(const Size& image_size, int apperture_size = 3);
    void release();

    GpuMat dx, dy;
    GpuMat mag;
    GpuMat map;
    GpuMat st1, st2;
    Ptr<FilterEngine_GPU> filterDX, filterDY;
};

CV_EXPORTS void Canny(const GpuMat& image, GpuMat& edges, double low_thresh, double high_thresh, int apperture_size = 3, bool L2gradient = false);
CV_EXPORTS void Canny(const GpuMat& image, CannyBuf& buf, GpuMat& edges, double low_thresh, double high_thresh, int apperture_size = 3, bool L2gradient = false);
CV_EXPORTS void Canny(const GpuMat& dx, const GpuMat& dy, GpuMat& edges, double low_thresh, double high_thresh, bool L2gradient = false);
CV_EXPORTS void Canny(const GpuMat& dx, const GpuMat& dy, CannyBuf& buf, GpuMat& edges, double low_thresh, double high_thresh, bool L2gradient = false);

class CV_EXPORTS ImagePyramid
{
public:
    inline ImagePyramid() : nLayers_(0) {}
    inline ImagePyramid(const GpuMat& img, int nLayers, Stream& stream = Stream::Null())
    {
        build(img, nLayers, stream);
    }

    void build(const GpuMat& img, int nLayers, Stream& stream = Stream::Null());

    void getLayer(GpuMat& outImg, Size outRoi, Stream& stream = Stream::Null()) const;

    inline void release()
    {
        layer0_.release();
        pyramid_.clear();
        nLayers_ = 0;
    }

private:
    GpuMat layer0_;
    std::vector<GpuMat> pyramid_;
    int nLayers_;
};

//! HoughLines

struct HoughLinesBuf
{
    GpuMat accum;
    GpuMat list;
};

CV_EXPORTS void HoughLines(const GpuMat& src, GpuMat& lines, float rho, float theta, int threshold, bool doSort = false, int maxLines = 4096);
CV_EXPORTS void HoughLines(const GpuMat& src, GpuMat& lines, HoughLinesBuf& buf, float rho, float theta, int threshold, bool doSort = false, int maxLines = 4096);
CV_EXPORTS void HoughLinesDownload(const GpuMat& d_lines, OutputArray h_lines, OutputArray h_votes = noArray());

//! HoughLinesP

//! finds line segments in the black-n-white image using probabalistic Hough transform
CV_EXPORTS void HoughLinesP(const GpuMat& image, GpuMat& lines, HoughLinesBuf& buf, float rho, float theta, int minLineLength, int maxLineGap, int maxLines = 4096);

//! HoughCircles

struct HoughCirclesBuf
{
    GpuMat edges;
    GpuMat accum;
    GpuMat list;
    CannyBuf cannyBuf;
};

CV_EXPORTS void HoughCircles(const GpuMat& src, GpuMat& circles, int method, float dp, float minDist, int cannyThreshold, int votesThreshold, int minRadius, int maxRadius, int maxCircles = 4096);
CV_EXPORTS void HoughCircles(const GpuMat& src, GpuMat& circles, HoughCirclesBuf& buf, int method, float dp, float minDist, int cannyThreshold, int votesThreshold, int minRadius, int maxRadius, int maxCircles = 4096);
CV_EXPORTS void HoughCirclesDownload(const GpuMat& d_circles, OutputArray h_circles);

//! finds arbitrary template in the grayscale image using Generalized Hough Transform
//! Ballard, D.H. (1981). Generalizing the Hough transform to detect arbitrary shapes. Pattern Recognition 13 (2): 111-122.
//! Guil, N., González-Linares, J.M. and Zapata, E.L. (1999). Bidimensional shape detection using an invariant approach. Pattern Recognition 32 (6): 1025-1038.
class CV_EXPORTS GeneralizedHough_GPU : public cv::Algorithm
{
public:
    static Ptr<GeneralizedHough_GPU> create(int method);

    virtual ~GeneralizedHough_GPU();

    //! set template to search
    void setTemplate(const GpuMat& templ, int cannyThreshold = 100, Point templCenter = Point(-1, -1));
    void setTemplate(const GpuMat& edges, const GpuMat& dx, const GpuMat& dy, Point templCenter = Point(-1, -1));

    //! find template on image
    void detect(const GpuMat& image, GpuMat& positions, int cannyThreshold = 100);
    void detect(const GpuMat& edges, const GpuMat& dx, const GpuMat& dy, GpuMat& positions);

    void download(const GpuMat& d_positions, OutputArray h_positions, OutputArray h_votes = noArray());

    void release();

protected:
    virtual void setTemplateImpl(const GpuMat& edges, const GpuMat& dx, const GpuMat& dy, Point templCenter) = 0;
    virtual void detectImpl(const GpuMat& edges, const GpuMat& dx, const GpuMat& dy, GpuMat& positions) = 0;
    virtual void releaseImpl() = 0;

private:
    GpuMat edges_;
    CannyBuf cannyBuf_;
};

///////////////////////////// Calibration 3D //////////////////////////////////

CV_EXPORTS void transformPoints(const GpuMat& src, const Mat& rvec, const Mat& tvec,
                                GpuMat& dst, Stream& stream = Stream::Null());

CV_EXPORTS void projectPoints(const GpuMat& src, const Mat& rvec, const Mat& tvec,
                              const Mat& camera_mat, const Mat& dist_coef, GpuMat& dst,
                              Stream& stream = Stream::Null());

CV_EXPORTS void solvePnPRansac(const Mat& object, const Mat& image, const Mat& camera_mat,
                               const Mat& dist_coef, Mat& rvec, Mat& tvec, bool use_extrinsic_guess=false,
                               int num_iters=100, float max_dist=8.0, int min_inlier_count=100,
                               std::vector<int>* inliers=NULL);

//////////////////////////////// Image Labeling ////////////////////////////////

//!performs labeling via graph cuts of a 2D regular 4-connected graph.
CV_EXPORTS void graphcut(GpuMat& terminals, GpuMat& leftTransp, GpuMat& rightTransp, GpuMat& top, GpuMat& bottom, GpuMat& labels,
                         GpuMat& buf, Stream& stream = Stream::Null());

//!performs labeling via graph cuts of a 2D regular 8-connected graph.
CV_EXPORTS void graphcut(GpuMat& terminals, GpuMat& leftTransp, GpuMat& rightTransp, GpuMat& top, GpuMat& topLeft, GpuMat& topRight,
                         GpuMat& bottom, GpuMat& bottomLeft, GpuMat& bottomRight,
                         GpuMat& labels,
                         GpuMat& buf, Stream& stream = Stream::Null());

//! compute mask for Generalized Flood fill componetns labeling.
CV_EXPORTS void connectivityMask(const GpuMat& image, GpuMat& mask, const cv::Scalar& lo, const cv::Scalar& hi, Stream& stream = Stream::Null());

//! performs connected componnents labeling.
CV_EXPORTS void labelComponents(const GpuMat& mask, GpuMat& components, int flags = 0, Stream& stream = Stream::Null());

////////////////////////////////// Histograms //////////////////////////////////

//! Compute levels with even distribution. levels will have 1 row and nLevels cols and CV_32SC1 type.
CV_EXPORTS void evenLevels(GpuMat& levels, int nLevels, int lowerLevel, int upperLevel);
//! Calculates histogram with evenly distributed bins for signle channel source.
//! Supports CV_8UC1, CV_16UC1 and CV_16SC1 source types.
//! Output hist will have one row and histSize cols and CV_32SC1 type.
CV_EXPORTS void histEven(const GpuMat& src, GpuMat& hist, int histSize, int lowerLevel, int upperLevel, Stream& stream = Stream::Null());
CV_EXPORTS void histEven(const GpuMat& src, GpuMat& hist, GpuMat& buf, int histSize, int lowerLevel, int upperLevel, Stream& stream = Stream::Null());
//! Calculates histogram with evenly distributed bins for four-channel source.
//! All channels of source are processed separately.
//! Supports CV_8UC4, CV_16UC4 and CV_16SC4 source types.
//! Output hist[i] will have one row and histSize[i] cols and CV_32SC1 type.
CV_EXPORTS void histEven(const GpuMat& src, GpuMat hist[4], int histSize[4], int lowerLevel[4], int upperLevel[4], Stream& stream = Stream::Null());
CV_EXPORTS void histEven(const GpuMat& src, GpuMat hist[4], GpuMat& buf, int histSize[4], int lowerLevel[4], int upperLevel[4], Stream& stream = Stream::Null());
//! Calculates histogram with bins determined by levels array.
//! levels must have one row and CV_32SC1 type if source has integer type or CV_32FC1 otherwise.
//! Supports CV_8UC1, CV_16UC1, CV_16SC1 and CV_32FC1 source types.
//! Output hist will have one row and (levels.cols-1) cols and CV_32SC1 type.
CV_EXPORTS void histRange(const GpuMat& src, GpuMat& hist, const GpuMat& levels, Stream& stream = Stream::Null());
CV_EXPORTS void histRange(const GpuMat& src, GpuMat& hist, const GpuMat& levels, GpuMat& buf, Stream& stream = Stream::Null());
//! Calculates histogram with bins determined by levels array.
//! All levels must have one row and CV_32SC1 type if source has integer type or CV_32FC1 otherwise.
//! All channels of source are processed separately.
//! Supports CV_8UC4, CV_16UC4, CV_16SC4 and CV_32FC4 source types.
//! Output hist[i] will have one row and (levels[i].cols-1) cols and CV_32SC1 type.
CV_EXPORTS void histRange(const GpuMat& src, GpuMat hist[4], const GpuMat levels[4], Stream& stream = Stream::Null());
CV_EXPORTS void histRange(const GpuMat& src, GpuMat hist[4], const GpuMat levels[4], GpuMat& buf, Stream& stream = Stream::Null());

//! Calculates histogram for 8u one channel image
//! Output hist will have one row, 256 cols and CV32SC1 type.
CV_EXPORTS void calcHist(const GpuMat& src, GpuMat& hist, Stream& stream = Stream::Null());
CV_EXPORTS void calcHist(const GpuMat& src, GpuMat& hist, GpuMat& buf, Stream& stream = Stream::Null());

//! normalizes the grayscale image brightness and contrast by normalizing its histogram
CV_EXPORTS void equalizeHist(const GpuMat& src, GpuMat& dst, Stream& stream = Stream::Null());
CV_EXPORTS void equalizeHist(const GpuMat& src, GpuMat& dst, GpuMat& hist, Stream& stream = Stream::Null());
CV_EXPORTS void equalizeHist(const GpuMat& src, GpuMat& dst, GpuMat& hist, GpuMat& buf, Stream& stream = Stream::Null());

class CV_EXPORTS CLAHE : public cv::CLAHE
{
public:
    using cv::CLAHE::apply;
    virtual void apply(InputArray src, OutputArray dst, Stream& stream) = 0;
};
CV_EXPORTS Ptr<cv::gpu::CLAHE> createCLAHE(double clipLimit = 40.0, Size tileGridSize = Size(8, 8));

//////////////////////////////// StereoBM_GPU ////////////////////////////////

class CV_EXPORTS StereoBM_GPU
{
public:
    enum { BASIC_PRESET = 0, PREFILTER_XSOBEL = 1 };

    enum { DEFAULT_NDISP = 64, DEFAULT_WINSZ = 19 };

    //! the default constructor
    StereoBM_GPU();
    //! the full constructor taking the camera-specific preset, number of disparities and the SAD window size. ndisparities must be multiple of 8.
    StereoBM_GPU(int preset, int ndisparities = DEFAULT_NDISP, int winSize = DEFAULT_WINSZ);

    //! the stereo correspondence operator. Finds the disparity for the specified rectified stereo pair
    //! Output disparity has CV_8U type.
    void operator()(const GpuMat& left, const GpuMat& right, GpuMat& disparity, Stream& stream = Stream::Null());

    //! Some heuristics that tries to estmate
    // if current GPU will be faster than CPU in this algorithm.
    // It queries current active device.
    static bool checkIfGpuCallReasonable();

    int preset;
    int ndisp;
    int winSize;

    // If avergeTexThreshold  == 0 => post procesing is disabled
    // If avergeTexThreshold != 0 then disparity is set 0 in each point (x,y) where for left image
    // SumOfHorizontalGradiensInWindow(x, y, winSize) < (winSize * winSize) * avergeTexThreshold
    // i.e. input left image is low textured.
    float avergeTexThreshold;

private:
    GpuMat minSSD, leBuf, riBuf;
};

////////////////////////// StereoBeliefPropagation ///////////////////////////
// "Efficient Belief Propagation for Early Vision"
// P.Felzenszwalb

class CV_EXPORTS StereoBeliefPropagation
{
public:
    enum { DEFAULT_NDISP  = 64 };
    enum { DEFAULT_ITERS  = 5  };
    enum { DEFAULT_LEVELS = 5  };

    static void estimateRecommendedParams(int width, int height, int& ndisp, int& iters, int& levels);

    //! the default constructor
    explicit StereoBeliefPropagation(int ndisp  = DEFAULT_NDISP,
                                     int iters  = DEFAULT_ITERS,
                                     int levels = DEFAULT_LEVELS,
                                     int msg_type = CV_32F);

    //! the full constructor taking the number of disparities, number of BP iterations on each level,
    //! number of levels, truncation of data cost, data weight,
    //! truncation of discontinuity cost and discontinuity single jump
    //! DataTerm = data_weight * min(fabs(I2-I1), max_data_term)
    //! DiscTerm = min(disc_single_jump * fabs(f1-f2), max_disc_term)
    //! please see paper for more details
    StereoBeliefPropagation(int ndisp, int iters, int levels,
        float max_data_term, float data_weight,
        float max_disc_term, float disc_single_jump,
        int msg_type = CV_32F);

    //! the stereo correspondence operator. Finds the disparity for the specified rectified stereo pair,
    //! if disparity is empty output type will be CV_16S else output type will be disparity.type().
    void operator()(const GpuMat& left, const GpuMat& right, GpuMat& disparity, Stream& stream = Stream::Null());


    //! version for user specified data term
    void operator()(const GpuMat& data, GpuMat& disparity, Stream& stream = Stream::Null());

    int ndisp;

    int iters;
    int levels;

    float max_data_term;
    float data_weight;
    float max_disc_term;
    float disc_single_jump;

    int msg_type;
private:
    GpuMat u, d, l, r, u2, d2, l2, r2;
    std::vector<GpuMat> datas;
    GpuMat out;
};

/////////////////////////// StereoConstantSpaceBP ///////////////////////////
// "A Constant-Space Belief Propagation Algorithm for Stereo Matching"
// Qingxiong Yang, Liang Wang, Narendra Ahuja
// http://vision.ai.uiuc.edu/~qyang6/

class CV_EXPORTS StereoConstantSpaceBP
{
public:
    enum { DEFAULT_NDISP    = 128 };
    enum { DEFAULT_ITERS    = 8   };
    enum { DEFAULT_LEVELS   = 4   };
    enum { DEFAULT_NR_PLANE = 4   };

    static void estimateRecommendedParams(int width, int height, int& ndisp, int& iters, int& levels, int& nr_plane);

    //! the default constructor
    explicit StereoConstantSpaceBP(int ndisp    = DEFAULT_NDISP,
                                   int iters    = DEFAULT_ITERS,
                                   int levels   = DEFAULT_LEVELS,
                                   int nr_plane = DEFAULT_NR_PLANE,
                                   int msg_type = CV_32F);

    //! the full constructor taking the number of disparities, number of BP iterations on each level,
    //! number of levels, number of active disparity on the first level, truncation of data cost, data weight,
    //! truncation of discontinuity cost, discontinuity single jump and minimum disparity threshold
    StereoConstantSpaceBP(int ndisp, int iters, int levels, int nr_plane,
        float max_data_term, float data_weight, float max_disc_term, float disc_single_jump,
        int min_disp_th = 0,
        int msg_type = CV_32F);

    //! the stereo correspondence operator. Finds the disparity for the specified rectified stereo pair,
    //! if disparity is empty output type will be CV_16S else output type will be disparity.type().
    void operator()(const GpuMat& left, const GpuMat& right, GpuMat& disparity, Stream& stream = Stream::Null());

    int ndisp;

    int iters;
    int levels;

    int nr_plane;

    float max_data_term;
    float data_weight;
    float max_disc_term;
    float disc_single_jump;

    int min_disp_th;

    int msg_type;

    bool use_local_init_data_cost;
private:
    GpuMat messages_buffers;

    GpuMat temp;
    GpuMat out;
};

/////////////////////////// DisparityBilateralFilter ///////////////////////////
// Disparity map refinement using joint bilateral filtering given a single color image.
// Qingxiong Yang, Liang Wang, Narendra Ahuja
// http://vision.ai.uiuc.edu/~qyang6/

class CV_EXPORTS DisparityBilateralFilter
{
public:
    enum { DEFAULT_NDISP  = 64 };
    enum { DEFAULT_RADIUS = 3 };
    enum { DEFAULT_ITERS  = 1 };

    //! the default constructor
    explicit DisparityBilateralFilter(int ndisp = DEFAULT_NDISP, int radius = DEFAULT_RADIUS, int iters = DEFAULT_ITERS);

    //! the full constructor taking the number of disparities, filter radius,
    //! number of iterations, truncation of data continuity, truncation of disparity continuity
    //! and filter range sigma
    DisparityBilateralFilter(int ndisp, int radius, int iters, float edge_threshold, float max_disc_threshold, float sigma_range);

    //! the disparity map refinement operator. Refine disparity map using joint bilateral filtering given a single color image.
    //! disparity must have CV_8U or CV_16S type, image must have CV_8UC1 or CV_8UC3 type.
    void operator()(const GpuMat& disparity, const GpuMat& image, GpuMat& dst, Stream& stream = Stream::Null());

private:
    int ndisp;
    int radius;
    int iters;

    float edge_threshold;
    float max_disc_threshold;
    float sigma_range;

    GpuMat table_color;
    GpuMat table_space;
};


//////////////// HOG (Histogram-of-Oriented-Gradients) Descriptor and Object Detector //////////////
struct CV_EXPORTS HOGConfidence
{
   double scale;
   std::vector<Point> locations;
   std::vector<double> confidences;
   std::vector<double> part_scores[4];
};

struct CV_EXPORTS HOGDescriptor
{
    enum { DEFAULT_WIN_SIGMA = -1 };
    enum { DEFAULT_NLEVELS = 64 };
    enum { DESCR_FORMAT_ROW_BY_ROW, DESCR_FORMAT_COL_BY_COL };

    HOGDescriptor(Size win_size=Size(64, 128), Size block_size=Size(16, 16),
                  Size block_stride=Size(8, 8), Size cell_size=Size(8, 8),
                  int nbins=9, double win_sigma=DEFAULT_WIN_SIGMA,
                  double threshold_L2hys=0.2, bool gamma_correction=true,
                  int nlevels=DEFAULT_NLEVELS);

    size_t getDescriptorSize() const;
    size_t getBlockHistogramSize() const;

    void setSVMDetector(const std::vector<float>& detector);

    static std::vector<float> getDefaultPeopleDetector();
    static std::vector<float> getPeopleDetector48x96();
    static std::vector<float> getPeopleDetector64x128();

    void detect(const GpuMat& img, std::vector<Point>& found_locations,
                double hit_threshold=0, Size win_stride=Size(),
                Size padding=Size());

    void detectMultiScale(const GpuMat& img, std::vector<Rect>& found_locations,
                          double hit_threshold=0, Size win_stride=Size(),
                          Size padding=Size(), double scale0=1.05,
                          int group_threshold=2);

    void computeConfidence(const GpuMat& img, std::vector<Point>& hits, double hit_threshold,
                                                Size win_stride, Size padding, std::vector<Point>& locations, std::vector<double>& confidences);

    void computeConfidenceMultiScale(const GpuMat& img, std::vector<Rect>& found_locations,
                                                                    double hit_threshold, Size win_stride, Size padding,
                                                                    std::vector<HOGConfidence> &conf_out, int group_threshold);

    void getDescriptors(const GpuMat& img, Size win_stride,
                        GpuMat& descriptors,
                        int descr_format=DESCR_FORMAT_COL_BY_COL);

    Size win_size;
    Size block_size;
    Size block_stride;
    Size cell_size;
    int nbins;
    double win_sigma;
    double threshold_L2hys;
    bool gamma_correction;
    int nlevels;

protected:
    void computeBlockHistograms(const GpuMat& img);
    void computeGradient(const GpuMat& img, GpuMat& grad, GpuMat& qangle);

    double getWinSigma() const;
    bool checkDetectorSize() const;

    static int numPartsWithin(int size, int part_size, int stride);
    static Size numPartsWithin(Size size, Size part_size, Size stride);

    // Coefficients of the separating plane
    float free_coef;
    GpuMat detector;

    // Results of the last classification step
    GpuMat labels, labels_buf;
    Mat labels_host;

    // Results of the last histogram evaluation step
    GpuMat block_hists, block_hists_buf;

    // Gradients conputation results
    GpuMat grad, qangle, grad_buf, qangle_buf;

    // returns subbuffer with required size, reallocates buffer if nessesary.
    static GpuMat getBuffer(const Size& sz, int type, GpuMat& buf);
    static GpuMat getBuffer(int rows, int cols, int type, GpuMat& buf);

    std::vector<GpuMat> image_scales;
};


////////////////////////////////// BruteForceMatcher //////////////////////////////////

class CV_EXPORTS BFMatcher_GPU
{
public:
    explicit BFMatcher_GPU(int norm = cv::NORM_L2);

    // Add descriptors to train descriptor collection
    void add(const std::vector<GpuMat>& descCollection);

    // Get train descriptors collection
    const std::vector<GpuMat>& getTrainDescriptors() const;

    // Clear train descriptors collection
    void clear();

    // Return true if there are not train descriptors in collection
    bool empty() const;

    // Return true if the matcher supports mask in match methods
    bool isMaskSupported() const;

    // Find one best match for each query descriptor
    void matchSingle(const GpuMat& query, const GpuMat& train,
        GpuMat& trainIdx, GpuMat& distance,
        const GpuMat& mask = GpuMat(), Stream& stream = Stream::Null());

    // Download trainIdx and distance and convert it to CPU vector with DMatch
    static void matchDownload(const GpuMat& trainIdx, const GpuMat& distance, std::vector<DMatch>& matches);
    // Convert trainIdx and distance to vector with DMatch
    static void matchConvert(const Mat& trainIdx, const Mat& distance, std::vector<DMatch>& matches);

    // Find one best match for each query descriptor
    void match(const GpuMat& query, const GpuMat& train, std::vector<DMatch>& matches, const GpuMat& mask = GpuMat());

    // Make gpu collection of trains and masks in suitable format for matchCollection function
    void makeGpuCollection(GpuMat& trainCollection, GpuMat& maskCollection, const std::vector<GpuMat>& masks = std::vector<GpuMat>());

    // Find one best match from train collection for each query descriptor
    void matchCollection(const GpuMat& query, const GpuMat& trainCollection,
        GpuMat& trainIdx, GpuMat& imgIdx, GpuMat& distance,
        const GpuMat& masks = GpuMat(), Stream& stream = Stream::Null());

    // Download trainIdx, imgIdx and distance and convert it to vector with DMatch
    static void matchDownload(const GpuMat& trainIdx, const GpuMat& imgIdx, const GpuMat& distance, std::vector<DMatch>& matches);
    // Convert trainIdx, imgIdx and distance to vector with DMatch
    static void matchConvert(const Mat& trainIdx, const Mat& imgIdx, const Mat& distance, std::vector<DMatch>& matches);

    // Find one best match from train collection for each query descriptor.
    void match(const GpuMat& query, std::vector<DMatch>& matches, const std::vector<GpuMat>& masks = std::vector<GpuMat>());

    // Find k best matches for each query descriptor (in increasing order of distances)
    void knnMatchSingle(const GpuMat& query, const GpuMat& train,
        GpuMat& trainIdx, GpuMat& distance, GpuMat& allDist, int k,
        const GpuMat& mask = GpuMat(), Stream& stream = Stream::Null());

    // Download trainIdx and distance and convert it to vector with DMatch
    // compactResult is used when mask is not empty. If compactResult is false matches
    // vector will have the same size as queryDescriptors rows. If compactResult is true
    // matches vector will not contain matches for fully masked out query descriptors.
    static void knnMatchDownload(const GpuMat& trainIdx, const GpuMat& distance,
        std::vector< std::vector<DMatch> >& matches, bool compactResult = false);
    // Convert trainIdx and distance to vector with DMatch
    static void knnMatchConvert(const Mat& trainIdx, const Mat& distance,
        std::vector< std::vector<DMatch> >& matches, bool compactResult = false);

    // Find k best matches for each query descriptor (in increasing order of distances).
    // compactResult is used when mask is not empty. If compactResult is false matches
    // vector will have the same size as queryDescriptors rows. If compactResult is true
    // matches vector will not contain matches for fully masked out query descriptors.
    void knnMatch(const GpuMat& query, const GpuMat& train,
        std::vector< std::vector<DMatch> >& matches, int k, const GpuMat& mask = GpuMat(),
        bool compactResult = false);

    // Find k best matches from train collection for each query descriptor (in increasing order of distances)
    void knnMatch2Collection(const GpuMat& query, const GpuMat& trainCollection,
        GpuMat& trainIdx, GpuMat& imgIdx, GpuMat& distance,
        const GpuMat& maskCollection = GpuMat(), Stream& stream = Stream::Null());

    // Download trainIdx and distance and convert it to vector with DMatch
    // compactResult is used when mask is not empty. If compactResult is false matches
    // vector will have the same size as queryDescriptors rows. If compactResult is true
    // matches vector will not contain matches for fully masked out query descriptors.
    static void knnMatch2Download(const GpuMat& trainIdx, const GpuMat& imgIdx, const GpuMat& distance,
        std::vector< std::vector<DMatch> >& matches, bool compactResult = false);
    // Convert trainIdx and distance to vector with DMatch
    static void knnMatch2Convert(const Mat& trainIdx, const Mat& imgIdx, const Mat& distance,
        std::vector< std::vector<DMatch> >& matches, bool compactResult = false);

    // Find k best matches  for each query descriptor (in increasing order of distances).
    // compactResult is used when mask is not empty. If compactResult is false matches
    // vector will have the same size as queryDescriptors rows. If compactResult is true
    // matches vector will not contain matches for fully masked out query descriptors.
    void knnMatch(const GpuMat& query, std::vector< std::vector<DMatch> >& matches, int k,
        const std::vector<GpuMat>& masks = std::vector<GpuMat>(), bool compactResult = false);

    // Find best matches for each query descriptor which have distance less than maxDistance.
    // nMatches.at<int>(0, queryIdx) will contain matches count for queryIdx.
    // carefully nMatches can be greater than trainIdx.cols - it means that matcher didn't find all matches,
    // because it didn't have enough memory.
    // If trainIdx is empty, then trainIdx and distance will be created with size nQuery x max((nTrain / 100), 10),
    // otherwize user can pass own allocated trainIdx and distance with size nQuery x nMaxMatches
    // Matches doesn't sorted.
    void radiusMatchSingle(const GpuMat& query, const GpuMat& train,
        GpuMat& trainIdx, GpuMat& distance, GpuMat& nMatches, float maxDistance,
        const GpuMat& mask = GpuMat(), Stream& stream = Stream::Null());

    // Download trainIdx, nMatches and distance and convert it to vector with DMatch.
    // matches will be sorted in increasing order of distances.
    // compactResult is used when mask is not empty. If compactResult is false matches
    // vector will have the same size as queryDescriptors rows. If compactResult is true
    // matches vector will not contain matches for fully masked out query descriptors.
    static void radiusMatchDownload(const GpuMat& trainIdx, const GpuMat& distance, const GpuMat& nMatches,
        std::vector< std::vector<DMatch> >& matches, bool compactResult = false);
    // Convert trainIdx, nMatches and distance to vector with DMatch.
    static void radiusMatchConvert(const Mat& trainIdx, const Mat& distance, const Mat& nMatches,
        std::vector< std::vector<DMatch> >& matches, bool compactResult = false);

    // Find best matches for each query descriptor which have distance less than maxDistance
    // in increasing order of distances).
    void radiusMatch(const GpuMat& query, const GpuMat& train,
        std::vector< std::vector<DMatch> >& matches, float maxDistance,
        const GpuMat& mask = GpuMat(), bool compactResult = false);

    // Find best matches for each query descriptor which have distance less than maxDistance.
    // If trainIdx is empty, then trainIdx and distance will be created with size nQuery x max((nQuery / 100), 10),
    // otherwize user can pass own allocated trainIdx and distance with size nQuery x nMaxMatches
    // Matches doesn't sorted.
    void radiusMatchCollection(const GpuMat& query, GpuMat& trainIdx, GpuMat& imgIdx, GpuMat& distance, GpuMat& nMatches, float maxDistance,
        const std::vector<GpuMat>& masks = std::vector<GpuMat>(), Stream& stream = Stream::Null());

    // Download trainIdx, imgIdx, nMatches and distance and convert it to vector with DMatch.
    // matches will be sorted in increasing order of distances.
    // compactResult is used when mask is not empty. If compactResult is false matches
    // vector will have the same size as queryDescriptors rows. If compactResult is true
    // matches vector will not contain matches for fully masked out query descriptors.
    static void radiusMatchDownload(const GpuMat& trainIdx, const GpuMat& imgIdx, const GpuMat& distance, const GpuMat& nMatches,
        std::vector< std::vector<DMatch> >& matches, bool compactResult = false);
    // Convert trainIdx, nMatches and distance to vector with DMatch.
    static void radiusMatchConvert(const Mat& trainIdx, const Mat& imgIdx, const Mat& distance, const Mat& nMatches,
        std::vector< std::vector<DMatch> >& matches, bool compactResult = false);

    // Find best matches from train collection for each query descriptor which have distance less than
    // maxDistance (in increasing order of distances).
    void radiusMatch(const GpuMat& query, std::vector< std::vector<DMatch> >& matches, float maxDistance,
        const std::vector<GpuMat>& masks = std::vector<GpuMat>(), bool compactResult = false);

    int norm;

private:
    std::vector<GpuMat> trainDescCollection;
};

template <class Distance>
class CV_EXPORTS BruteForceMatcher_GPU;

template <typename T>
class CV_EXPORTS BruteForceMatcher_GPU< L1<T> > : public BFMatcher_GPU
{
public:
    explicit BruteForceMatcher_GPU() : BFMatcher_GPU(NORM_L1) {}
    explicit BruteForceMatcher_GPU(L1<T> /*d*/) : BFMatcher_GPU(NORM_L1) {}
};
template <typename T>
class CV_EXPORTS BruteForceMatcher_GPU< L2<T> > : public BFMatcher_GPU
{
public:
    explicit BruteForceMatcher_GPU() : BFMatcher_GPU(NORM_L2) {}
    explicit BruteForceMatcher_GPU(L2<T> /*d*/) : BFMatcher_GPU(NORM_L2) {}
};
template <> class CV_EXPORTS BruteForceMatcher_GPU< Hamming > : public BFMatcher_GPU
{
public:
    explicit BruteForceMatcher_GPU() : BFMatcher_GPU(NORM_HAMMING) {}
    explicit BruteForceMatcher_GPU(Hamming /*d*/) : BFMatcher_GPU(NORM_HAMMING) {}
};

////////////////////////////////// CascadeClassifier_GPU //////////////////////////////////////////
// The cascade classifier class for object detection: supports old haar and new lbp xlm formats and nvbin for haar cascades olny.
class CV_EXPORTS CascadeClassifier_GPU
{
public:
    CascadeClassifier_GPU();
    CascadeClassifier_GPU(const String& filename);
    ~CascadeClassifier_GPU();

    bool empty() const;
    bool load(const String& filename);
    void release();

    /* returns number of detected objects */
    int detectMultiScale(const GpuMat& image, GpuMat& objectsBuf, double scaleFactor = 1.2, int minNeighbors = 4, Size minSize = Size());
    int detectMultiScale(const GpuMat& image, GpuMat& objectsBuf, Size maxObjectSize, Size minSize = Size(), double scaleFactor = 1.1, int minNeighbors = 4);

    bool findLargestObject;
    bool visualizeInPlace;

    Size getClassifierSize() const;

private:
    struct CascadeClassifierImpl;
    CascadeClassifierImpl* impl;
    struct HaarCascade;
    struct LbpCascade;
    friend class CascadeClassifier_GPU_LBP;
};

////////////////////////////////// FAST //////////////////////////////////////////

class CV_EXPORTS FAST_GPU
{
public:
    enum
    {
        LOCATION_ROW = 0,
        RESPONSE_ROW,
        ROWS_COUNT
    };

    // all features have same size
    static const int FEATURE_SIZE = 7;

    explicit FAST_GPU(int threshold, bool nonmaxSupression = true, double keypointsRatio = 0.05);

    //! finds the keypoints using FAST detector
    //! supports only CV_8UC1 images
    void operator ()(const GpuMat& image, const GpuMat& mask, GpuMat& keypoints);
    void operator ()(const GpuMat& image, const GpuMat& mask, std::vector<KeyPoint>& keypoints);

    //! download keypoints from device to host memory
    static void downloadKeypoints(const GpuMat& d_keypoints, std::vector<KeyPoint>& keypoints);

    //! convert keypoints to KeyPoint vector
    static void convertKeypoints(const Mat& h_keypoints, std::vector<KeyPoint>& keypoints);

    //! release temporary buffer's memory
    void release();

    bool nonmaxSupression;

    int threshold;

    //! max keypoints = keypointsRatio * img.size().area()
    double keypointsRatio;

    //! find keypoints and compute it's response if nonmaxSupression is true
    //! return count of detected keypoints
    int calcKeyPointsLocation(const GpuMat& image, const GpuMat& mask);

    //! get final array of keypoints
    //! performs nonmax supression if needed
    //! return final count of keypoints
    int getKeyPoints(GpuMat& keypoints);

private:
    GpuMat kpLoc_;
    int count_;

    GpuMat score_;

    GpuMat d_keypoints_;
};

////////////////////////////////// ORB //////////////////////////////////////////

class CV_EXPORTS ORB_GPU
{
public:
    enum
    {
        X_ROW = 0,
        Y_ROW,
        RESPONSE_ROW,
        ANGLE_ROW,
        OCTAVE_ROW,
        SIZE_ROW,
        ROWS_COUNT
    };

    enum
    {
        DEFAULT_FAST_THRESHOLD = 20
    };

    //! Constructor
    explicit ORB_GPU(int nFeatures = 500, float scaleFactor = 1.2f, int nLevels = 8, int edgeThreshold = 31,
                     int firstLevel = 0, int WTA_K = 2, int scoreType = 0, int patchSize = 31);

    //! Compute the ORB features on an image
    //! image - the image to compute the features (supports only CV_8UC1 images)
    //! mask - the mask to apply
    //! keypoints - the resulting keypoints
    void operator()(const GpuMat& image, const GpuMat& mask, std::vector<KeyPoint>& keypoints);
    void operator()(const GpuMat& image, const GpuMat& mask, GpuMat& keypoints);

    //! Compute the ORB features and descriptors on an image
    //! image - the image to compute the features (supports only CV_8UC1 images)
    //! mask - the mask to apply
    //! keypoints - the resulting keypoints
    //! descriptors - descriptors array
    void operator()(const GpuMat& image, const GpuMat& mask, std::vector<KeyPoint>& keypoints, GpuMat& descriptors);
    void operator()(const GpuMat& image, const GpuMat& mask, GpuMat& keypoints, GpuMat& descriptors);

    //! download keypoints from device to host memory
    static void downloadKeyPoints(const GpuMat& d_keypoints, std::vector<KeyPoint>& keypoints);
    //! convert keypoints to KeyPoint vector
    static void convertKeyPoints(const Mat& d_keypoints, std::vector<KeyPoint>& keypoints);

    //! returns the descriptor size in bytes
    inline int descriptorSize() const { return kBytes; }

    inline void setFastParams(int threshold, bool nonmaxSupression = true)
    {
        fastDetector_.threshold = threshold;
        fastDetector_.nonmaxSupression = nonmaxSupression;
    }

    //! release temporary buffer's memory
    void release();

    //! if true, image will be blurred before descriptors calculation
    bool blurForDescriptor;

private:
    enum { kBytes = 32 };

    void buildScalePyramids(const GpuMat& image, const GpuMat& mask);

    void computeKeyPointsPyramid();

    void computeDescriptors(GpuMat& descriptors);

    void mergeKeyPoints(GpuMat& keypoints);

    int nFeatures_;
    float scaleFactor_;
    int nLevels_;
    int edgeThreshold_;
    int firstLevel_;
    int WTA_K_;
    int scoreType_;
    int patchSize_;

    // The number of desired features per scale
    std::vector<size_t> n_features_per_level_;

    // Points to compute BRIEF descriptors from
    GpuMat pattern_;

    std::vector<GpuMat> imagePyr_;
    std::vector<GpuMat> maskPyr_;

    GpuMat buf_;

    std::vector<GpuMat> keyPointsPyr_;
    std::vector<int> keyPointsCount_;

    FAST_GPU fastDetector_;

    Ptr<FilterEngine_GPU> blurFilter;

    GpuMat d_keypoints_;
};

////////////////////////////////// Optical Flow //////////////////////////////////////////

class CV_EXPORTS BroxOpticalFlow
{
public:
    BroxOpticalFlow(float alpha_, float gamma_, float scale_factor_, int inner_iterations_, int outer_iterations_, int solver_iterations_) :
        alpha(alpha_), gamma(gamma_), scale_factor(scale_factor_),
        inner_iterations(inner_iterations_), outer_iterations(outer_iterations_), solver_iterations(solver_iterations_)
    {
    }

    //! Compute optical flow
    //! frame0 - source frame (supports only CV_32FC1 type)
    //! frame1 - frame to track (with the same size and type as frame0)
    //! u      - flow horizontal component (along x axis)
    //! v      - flow vertical component (along y axis)
    void operator ()(const GpuMat& frame0, const GpuMat& frame1, GpuMat& u, GpuMat& v, Stream& stream = Stream::Null());

    //! flow smoothness
    float alpha;

    //! gradient constancy importance
    float gamma;

    //! pyramid scale factor
    float scale_factor;

    //! number of lagged non-linearity iterations (inner loop)
    int inner_iterations;

    //! number of warping iterations (number of pyramid levels)
    int outer_iterations;

    //! number of linear system solver iterations
    int solver_iterations;

    GpuMat buf;
};

class CV_EXPORTS GoodFeaturesToTrackDetector_GPU
{
public:
    explicit GoodFeaturesToTrackDetector_GPU(int maxCorners = 1000, double qualityLevel = 0.01, double minDistance = 0.0,
        int blockSize = 3, bool useHarrisDetector = false, double harrisK = 0.04);

    //! return 1 rows matrix with CV_32FC2 type
    void operator ()(const GpuMat& image, GpuMat& corners, const GpuMat& mask = GpuMat());

    int maxCorners;
    double qualityLevel;
    double minDistance;

    int blockSize;
    bool useHarrisDetector;
    double harrisK;

    void releaseMemory()
    {
        Dx_.release();
        Dy_.release();
        buf_.release();
        eig_.release();
        minMaxbuf_.release();
        tmpCorners_.release();
    }

private:
    GpuMat Dx_;
    GpuMat Dy_;
    GpuMat buf_;
    GpuMat eig_;
    GpuMat minMaxbuf_;
    GpuMat tmpCorners_;
};

inline GoodFeaturesToTrackDetector_GPU::GoodFeaturesToTrackDetector_GPU(int maxCorners_, double qualityLevel_, double minDistance_,
        int blockSize_, bool useHarrisDetector_, double harrisK_)
{
    maxCorners = maxCorners_;
    qualityLevel = qualityLevel_;
    minDistance = minDistance_;
    blockSize = blockSize_;
    useHarrisDetector = useHarrisDetector_;
    harrisK = harrisK_;
}


class CV_EXPORTS PyrLKOpticalFlow
{
public:
    PyrLKOpticalFlow();

    void sparse(const GpuMat& prevImg, const GpuMat& nextImg, const GpuMat& prevPts, GpuMat& nextPts,
        GpuMat& status, GpuMat* err = 0);

    void dense(const GpuMat& prevImg, const GpuMat& nextImg, GpuMat& u, GpuMat& v, GpuMat* err = 0);

    void releaseMemory();

    Size winSize;
    int maxLevel;
    int iters;
    bool useInitialFlow;

private:
    std::vector<GpuMat> prevPyr_;
    std::vector<GpuMat> nextPyr_;

    GpuMat buf_;

    GpuMat uPyr_[2];
    GpuMat vPyr_[2];
};


class CV_EXPORTS FarnebackOpticalFlow
{
public:
    FarnebackOpticalFlow()
    {
        numLevels = 5;
        pyrScale = 0.5;
        fastPyramids = false;
        winSize = 13;
        numIters = 10;
        polyN = 5;
        polySigma = 1.1;
        flags = 0;
    }

    int numLevels;
    double pyrScale;
    bool fastPyramids;
    int winSize;
    int numIters;
    int polyN;
    double polySigma;
    int flags;

    void operator ()(const GpuMat &frame0, const GpuMat &frame1, GpuMat &flowx, GpuMat &flowy, Stream &s = Stream::Null());

    void releaseMemory()
    {
        frames_[0].release();
        frames_[1].release();
        pyrLevel_[0].release();
        pyrLevel_[1].release();
        M_.release();
        bufM_.release();
        R_[0].release();
        R_[1].release();
        blurredFrame_[0].release();
        blurredFrame_[1].release();
        pyramid0_.clear();
        pyramid1_.clear();
    }

private:
    void prepareGaussian(
            int n, double sigma, float *g, float *xg, float *xxg,
            double &ig11, double &ig03, double &ig33, double &ig55);

    void setPolynomialExpansionConsts(int n, double sigma);

    void updateFlow_boxFilter(
            const GpuMat& R0, const GpuMat& R1, GpuMat& flowx, GpuMat &flowy,
            GpuMat& M, GpuMat &bufM, int blockSize, bool updateMatrices, Stream streams[]);

    void updateFlow_gaussianBlur(
            const GpuMat& R0, const GpuMat& R1, GpuMat& flowx, GpuMat& flowy,
            GpuMat& M, GpuMat &bufM, int blockSize, bool updateMatrices, Stream streams[]);

    GpuMat frames_[2];
    GpuMat pyrLevel_[2], M_, bufM_, R_[2], blurredFrame_[2];
    std::vector<GpuMat> pyramid0_, pyramid1_;
};


// Implementation of the Zach, Pock and Bischof Dual TV-L1 Optical Flow method
//
// see reference:
//   [1] C. Zach, T. Pock and H. Bischof, "A Duality Based Approach for Realtime TV-L1 Optical Flow".
//   [2] Javier Sanchez, Enric Meinhardt-Llopis and Gabriele Facciolo. "TV-L1 Optical Flow Estimation".
class CV_EXPORTS OpticalFlowDual_TVL1_GPU
{
public:
    OpticalFlowDual_TVL1_GPU();

    void operator ()(const GpuMat& I0, const GpuMat& I1, GpuMat& flowx, GpuMat& flowy);

    void collectGarbage();

    /**
     * Time step of the numerical scheme.
     */
    double tau;

    /**
     * Weight parameter for the data term, attachment parameter.
     * This is the most relevant parameter, which determines the smoothness of the output.
     * The smaller this parameter is, the smoother the solutions we obtain.
     * It depends on the range of motions of the images, so its value should be adapted to each image sequence.
     */
    double lambda;

    /**
     * Weight parameter for (u - v)^2, tightness parameter.
     * It serves as a link between the attachment and the regularization terms.
     * In theory, it should have a small value in order to maintain both parts in correspondence.
     * The method is stable for a large range of values of this parameter.
     */
    double theta;

    /**
     * Number of scales used to create the pyramid of images.
     */
    int nscales;

    /**
     * Number of warpings per scale.
     * Represents the number of times that I1(x+u0) and grad( I1(x+u0) ) are computed per scale.
     * This is a parameter that assures the stability of the method.
     * It also affects the running time, so it is a compromise between speed and accuracy.
     */
    int warps;

    /**
     * Stopping criterion threshold used in the numerical scheme, which is a trade-off between precision and running time.
     * A small value will yield more accurate solutions at the expense of a slower convergence.
     */
    double epsilon;

    /**
     * Stopping criterion iterations number used in the numerical scheme.
     */
    int iterations;

    double scaleStep;

    bool useInitialFlow;

private:
    void procOneScale(const GpuMat& I0, const GpuMat& I1, GpuMat& u1, GpuMat& u2);

    std::vector<GpuMat> I0s;
    std::vector<GpuMat> I1s;
    std::vector<GpuMat> u1s;
    std::vector<GpuMat> u2s;

    GpuMat I1x_buf;
    GpuMat I1y_buf;

    GpuMat I1w_buf;
    GpuMat I1wx_buf;
    GpuMat I1wy_buf;

    GpuMat grad_buf;
    GpuMat rho_c_buf;

    GpuMat p11_buf;
    GpuMat p12_buf;
    GpuMat p21_buf;
    GpuMat p22_buf;

    GpuMat diff_buf;
    GpuMat norm_buf;
};


//! Calculates optical flow for 2 images using block matching algorithm */
CV_EXPORTS void calcOpticalFlowBM(const GpuMat& prev, const GpuMat& curr,
                                  Size block_size, Size shift_size, Size max_range, bool use_previous,
                                  GpuMat& velx, GpuMat& vely, GpuMat& buf,
                                  Stream& stream = Stream::Null());

class CV_EXPORTS FastOpticalFlowBM
{
public:
    void operator ()(const GpuMat& I0, const GpuMat& I1, GpuMat& flowx, GpuMat& flowy, int search_window = 21, int block_window = 7, Stream& s = Stream::Null());

private:
    GpuMat buffer;
    GpuMat extended_I0;
    GpuMat extended_I1;
};


//! Interpolate frames (images) using provided optical flow (displacement field).
//! frame0   - frame 0 (32-bit floating point images, single channel)
//! frame1   - frame 1 (the same type and size)
//! fu       - forward horizontal displacement
//! fv       - forward vertical displacement
//! bu       - backward horizontal displacement
//! bv       - backward vertical displacement
//! pos      - new frame position
//! newFrame - new frame
//! buf      - temporary buffer, will have width x 6*height size, CV_32FC1 type and contain 6 GpuMat;
//!            occlusion masks            0, occlusion masks            1,
//!            interpolated forward flow  0, interpolated forward flow  1,
//!            interpolated backward flow 0, interpolated backward flow 1
//!
CV_EXPORTS void interpolateFrames(const GpuMat& frame0, const GpuMat& frame1,
                                  const GpuMat& fu, const GpuMat& fv,
                                  const GpuMat& bu, const GpuMat& bv,
                                  float pos, GpuMat& newFrame, GpuMat& buf,
                                  Stream& stream = Stream::Null());

CV_EXPORTS void createOpticalFlowNeedleMap(const GpuMat& u, const GpuMat& v, GpuMat& vertex, GpuMat& colors);


//////////////////////// Background/foreground segmentation ////////////////////////

// Foreground Object Detection from Videos Containing Complex Background.
// Liyuan Li, Weimin Huang, Irene Y.H. Gu, and Qi Tian.
// ACM MM2003 9p
class CV_EXPORTS FGDStatModel
{
public:
    struct CV_EXPORTS Params
    {
        int Lc;  // Quantized levels per 'color' component. Power of two, typically 32, 64 or 128.
        int N1c; // Number of color vectors used to model normal background color variation at a given pixel.
        int N2c; // Number of color vectors retained at given pixel.  Must be > N1c, typically ~ 5/3 of N1c.
        // Used to allow the first N1c vectors to adapt over time to changing background.

        int Lcc;  // Quantized levels per 'color co-occurrence' component.  Power of two, typically 16, 32 or 64.
        int N1cc; // Number of color co-occurrence vectors used to model normal background color variation at a given pixel.
        int N2cc; // Number of color co-occurrence vectors retained at given pixel.  Must be > N1cc, typically ~ 5/3 of N1cc.
        // Used to allow the first N1cc vectors to adapt over time to changing background.

        bool is_obj_without_holes; // If TRUE we ignore holes within foreground blobs. Defaults to TRUE.
        int perform_morphing;     // Number of erode-dilate-erode foreground-blob cleanup iterations.
        // These erase one-pixel junk blobs and merge almost-touching blobs. Default value is 1.

        float alpha1; // How quickly we forget old background pixel values seen. Typically set to 0.1.
        float alpha2; // "Controls speed of feature learning". Depends on T. Typical value circa 0.005.
        float alpha3; // Alternate to alpha2, used (e.g.) for quicker initial convergence. Typical value 0.1.

        float delta;   // Affects color and color co-occurrence quantization, typically set to 2.
        float T;       // A percentage value which determines when new features can be recognized as new background. (Typically 0.9).
        float minArea; // Discard foreground blobs whose bounding box is smaller than this threshold.

        // default Params
        Params();
    };

    // out_cn - channels count in output result (can be 3 or 4)
    // 4-channels require more memory, but a bit faster
    explicit FGDStatModel(int out_cn = 3);
    explicit FGDStatModel(const cv::gpu::GpuMat& firstFrame, const Params& params = Params(), int out_cn = 3);

    ~FGDStatModel();

    void create(const cv::gpu::GpuMat& firstFrame, const Params& params = Params());
    void release();

    int update(const cv::gpu::GpuMat& curFrame);

    //8UC3 or 8UC4 reference background image
    cv::gpu::GpuMat background;

    //8UC1 foreground image
    cv::gpu::GpuMat foreground;

    std::vector< std::vector<cv::Point> > foreground_regions;

private:
    FGDStatModel(const FGDStatModel&);
    FGDStatModel& operator=(const FGDStatModel&);

    class Impl;
    std::auto_ptr<Impl> impl_;
};

/*!
 Gaussian Mixture-based Backbround/Foreground Segmentation Algorithm

 The class implements the following algorithm:
 "An improved adaptive background mixture model for real-time tracking with shadow detection"
 P. KadewTraKuPong and R. Bowden,
 Proc. 2nd European Workshp on Advanced Video-Based Surveillance Systems, 2001."
 http://personal.ee.surrey.ac.uk/Personal/R.Bowden/publications/avbs01/avbs01.pdf
*/
class CV_EXPORTS MOG_GPU
{
public:
    //! the default constructor
    MOG_GPU(int nmixtures = -1);

    //! re-initiaization method
    void initialize(Size frameSize, int frameType);

    //! the update operator
    void operator()(const GpuMat& frame, GpuMat& fgmask, float learningRate = 0.0f, Stream& stream = Stream::Null());

    //! computes a background image which are the mean of all background gaussians
    void getBackgroundImage(GpuMat& backgroundImage, Stream& stream = Stream::Null()) const;

    //! releases all inner buffers
    void release();

    int history;
    float varThreshold;
    float backgroundRatio;
    float noiseSigma;

private:
    int nmixtures_;

    Size frameSize_;
    int frameType_;
    int nframes_;

    GpuMat weight_;
    GpuMat sortKey_;
    GpuMat mean_;
    GpuMat var_;
};

/*!
 The class implements the following algorithm:
 "Improved adaptive Gausian mixture model for background subtraction"
 Z.Zivkovic
 International Conference Pattern Recognition, UK, August, 2004.
 http://www.zoranz.net/Publications/zivkovic2004ICPR.pdf
*/
class CV_EXPORTS MOG2_GPU
{
public:
    //! the default constructor
    MOG2_GPU(int nmixtures = -1);

    //! re-initiaization method
    void initialize(Size frameSize, int frameType);

    //! the update operator
    void operator()(const GpuMat& frame, GpuMat& fgmask, float learningRate = -1.0f, Stream& stream = Stream::Null());

    //! computes a background image which are the mean of all background gaussians
    void getBackgroundImage(GpuMat& backgroundImage, Stream& stream = Stream::Null()) const;

    //! releases all inner buffers
    void release();

    // parameters
    // you should call initialize after parameters changes

    int history;

    //! here it is the maximum allowed number of mixture components.
    //! Actual number is determined dynamically per pixel
    float varThreshold;
    // threshold on the squared Mahalanobis distance to decide if it is well described
    // by the background model or not. Related to Cthr from the paper.
    // This does not influence the update of the background. A typical value could be 4 sigma
    // and that is varThreshold=4*4=16; Corresponds to Tb in the paper.

    /////////////////////////
    // less important parameters - things you might change but be carefull
    ////////////////////////

    float backgroundRatio;
    // corresponds to fTB=1-cf from the paper
    // TB - threshold when the component becomes significant enough to be included into
    // the background model. It is the TB=1-cf from the paper. So I use cf=0.1 => TB=0.
    // For alpha=0.001 it means that the mode should exist for approximately 105 frames before
    // it is considered foreground
    // float noiseSigma;
    float varThresholdGen;

    //correspondts to Tg - threshold on the squared Mahalan. dist. to decide
    //when a sample is close to the existing components. If it is not close
    //to any a new component will be generated. I use 3 sigma => Tg=3*3=9.
    //Smaller Tg leads to more generated components and higher Tg might make
    //lead to small number of components but they can grow too large
    float fVarInit;
    float fVarMin;
    float fVarMax;

    //initial variance  for the newly generated components.
    //It will will influence the speed of adaptation. A good guess should be made.
    //A simple way is to estimate the typical standard deviation from the images.
    //I used here 10 as a reasonable value
    // min and max can be used to further control the variance
    float fCT; //CT - complexity reduction prior
    //this is related to the number of samples needed to accept that a component
    //actually exists. We use CT=0.05 of all the samples. By setting CT=0 you get
    //the standard Stauffer&Grimson algorithm (maybe not exact but very similar)

    //shadow detection parameters
    bool bShadowDetection; //default 1 - do shadow detection
    unsigned char nShadowDetection; //do shadow detection - insert this value as the detection result - 127 default value
    float fTau;
    // Tau - shadow threshold. The shadow is detected if the pixel is darker
    //version of the background. Tau is a threshold on how much darker the shadow can be.
    //Tau= 0.5 means that if pixel is more than 2 times darker then it is not shadow
    //See: Prati,Mikic,Trivedi,Cucchiarra,"Detecting Moving Shadows...",IEEE PAMI,2003.

private:
    int nmixtures_;

    Size frameSize_;
    int frameType_;
    int nframes_;

    GpuMat weight_;
    GpuMat variance_;
    GpuMat mean_;

    GpuMat bgmodelUsedModes_; //keep track of number of modes per pixel
};

/**
 * Background Subtractor module. Takes a series of images and returns a sequence of mask (8UC1)
 * images of the same size, where 255 indicates Foreground and 0 represents Background.
 * This class implements an algorithm described in "Visual Tracking of Human Visitors under
 * Variable-Lighting Conditions for a Responsive Audio Art Installation," A. Godbehere,
 * A. Matsukawa, K. Goldberg, American Control Conference, Montreal, June 2012.
 */
class CV_EXPORTS GMG_GPU
{
public:
    GMG_GPU();

    /**
     * Validate parameters and set up data structures for appropriate frame size.
     * @param frameSize Input frame size
     * @param min       Minimum value taken on by pixels in image sequence. Usually 0
     * @param max       Maximum value taken on by pixels in image sequence. e.g. 1.0 or 255
     */
    void initialize(Size frameSize, float min = 0.0f, float max = 255.0f);

    /**
     * Performs single-frame background subtraction and builds up a statistical background image
     * model.
     * @param frame        Input frame
     * @param fgmask       Output mask image representing foreground and background pixels
     * @param stream       Stream for the asynchronous version
     */
    void operator ()(const GpuMat& frame, GpuMat& fgmask, float learningRate = -1.0f, Stream& stream = Stream::Null());

    //! Releases all inner buffers
    void release();

    //! Total number of distinct colors to maintain in histogram.
    int maxFeatures;

    //! Set between 0.0 and 1.0, determines how quickly features are "forgotten" from histograms.
    float learningRate;

    //! Number of frames of video to use to initialize histograms.
    int numInitializationFrames;

    //! Number of discrete levels in each channel to be used in histograms.
    int quantizationLevels;

    //! Prior probability that any given pixel is a background pixel. A sensitivity parameter.
    float backgroundPrior;

    //! Value above which pixel is determined to be FG.
    float decisionThreshold;

    //! Smoothing radius, in pixels, for cleaning up FG image.
    int smoothingRadius;

    //! Perform background model update.
    bool updateBackgroundModel;

private:
    float maxVal_, minVal_;

    Size frameSize_;

    int frameNum_;

    GpuMat nfeatures_;
    GpuMat colors_;
    GpuMat weights_;

    Ptr<FilterEngine_GPU> boxFilter_;
    GpuMat buf_;
};

//! removes points (CV_32FC2, single row matrix) with zero mask value
CV_EXPORTS void compactPoints(GpuMat &points0, GpuMat &points1, const GpuMat &mask);

CV_EXPORTS void calcWobbleSuppressionMaps(
        int left, int idx, int right, Size size, const Mat &ml, const Mat &mr,
        GpuMat &mapx, GpuMat &mapy);

} // namespace gpu

} // namespace cv

#endif /* __OPENCV_GPU_HPP__ */
