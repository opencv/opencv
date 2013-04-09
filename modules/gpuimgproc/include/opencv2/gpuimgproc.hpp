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

#include "opencv2/core/gpumat.hpp"
#include "opencv2/gpufilters.hpp"
#include "opencv2/imgproc.hpp"

namespace cv { namespace gpu {

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

}} // namespace cv { namespace gpu {

#endif /* __OPENCV_GPUIMGPROC_HPP__ */
