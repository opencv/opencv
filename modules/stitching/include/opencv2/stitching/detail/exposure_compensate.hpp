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

#ifndef OPENCV_STITCHING_EXPOSURE_COMPENSATE_HPP
#define OPENCV_STITCHING_EXPOSURE_COMPENSATE_HPP

#if defined(NO)
#undef NO
//#warning Detected Apple 'NO' macro definition, it can cause build conflicts. Please, include this header before any Apple headers.
#endif

#include "opencv2/core.hpp"

namespace cv {
namespace detail {

//! @addtogroup stitching_exposure
//! @{

/** @brief Base class for all exposure compensators.
 */
class CV_EXPORTS_W ExposureCompensator
{
public:
    ExposureCompensator(): updateGain(true) {}
    virtual ~ExposureCompensator() {}

    enum { NO, GAIN, GAIN_BLOCKS, CHANNELS, CHANNELS_BLOCKS };
    CV_WRAP static Ptr<ExposureCompensator> createDefault(int type);

    /**
    @param corners Source image top-left corners
    @param images Source images
    @param masks Image masks to update (second value in pair specifies the value which should be used
    to detect where image is)
        */
    CV_WRAP void feed(const std::vector<Point> &corners, const std::vector<UMat> &images,
        const std::vector<UMat> &masks);
    /** @overload */
    virtual void feed(const std::vector<Point> &corners, const std::vector<UMat> &images,
        const std::vector<std::pair<UMat, uchar> > &masks) = 0;
    /** @brief Compensate exposure in the specified image.

    @param index Image index
    @param corner Image top-left corner
    @param image Image to process
    @param mask Image mask
        */
    CV_WRAP virtual void apply(int index, Point corner, InputOutputArray image, InputArray mask) = 0;
    CV_WRAP virtual void getMatGains(CV_OUT std::vector<Mat>& ) {CV_Error(Error::StsInternal, "");}
    CV_WRAP virtual void setMatGains(std::vector<Mat>& ) { CV_Error(Error::StsInternal, ""); }
    CV_WRAP void setUpdateGain(bool b) { updateGain = b; }
    CV_WRAP bool getUpdateGain() { return updateGain; }
protected :
    bool updateGain;
};

/** @brief Stub exposure compensator which does nothing.
 */
class CV_EXPORTS_W NoExposureCompensator : public ExposureCompensator
{
public:
    void feed(const std::vector<Point> &/*corners*/, const std::vector<UMat> &/*images*/,
              const std::vector<std::pair<UMat,uchar> > &/*masks*/) CV_OVERRIDE { }
    CV_WRAP void apply(int /*index*/, Point /*corner*/, InputOutputArray /*image*/, InputArray /*mask*/) CV_OVERRIDE { }
    CV_WRAP void getMatGains(CV_OUT std::vector<Mat>& umv) CV_OVERRIDE { umv.clear(); return; }
    CV_WRAP void setMatGains(std::vector<Mat>& umv) CV_OVERRIDE { umv.clear(); return; }
};

/** @brief Exposure compensator which tries to remove exposure related artifacts by adjusting image
intensities, see @cite BL07 and @cite WJ10 for details.
 */
class CV_EXPORTS_W GainCompensator : public ExposureCompensator
{
public:
    // This Constructor only exists to make source level compatibility detector happy
    CV_WRAP GainCompensator()
            : GainCompensator(1) {}
    CV_WRAP GainCompensator(int nr_feeds)
            : nr_feeds_(nr_feeds), similarity_threshold_(1) {}
    void feed(const std::vector<Point> &corners, const std::vector<UMat> &images,
              const std::vector<std::pair<UMat,uchar> > &masks) CV_OVERRIDE;
    void singleFeed(const std::vector<Point> &corners, const std::vector<UMat> &images,
                    const std::vector<std::pair<UMat,uchar> > &masks);
    CV_WRAP void apply(int index, Point corner, InputOutputArray image, InputArray mask) CV_OVERRIDE;
    CV_WRAP void getMatGains(CV_OUT std::vector<Mat>& umv) CV_OVERRIDE ;
    CV_WRAP void setMatGains(std::vector<Mat>& umv) CV_OVERRIDE ;
    CV_WRAP void setNrFeeds(int nr_feeds) { nr_feeds_ = nr_feeds; }
    CV_WRAP int getNrFeeds() { return nr_feeds_; }
    CV_WRAP void setSimilarityThreshold(double similarity_threshold) { similarity_threshold_ = similarity_threshold; }
    CV_WRAP double getSimilarityThreshold() const { return similarity_threshold_; }
    void prepareSimilarityMask(const std::vector<Point> &corners, const std::vector<UMat> &images);
    std::vector<double> gains() const;

private:
    UMat buildSimilarityMask(InputArray src_array1, InputArray src_array2);

    Mat_<double> gains_;
    int nr_feeds_;
    double similarity_threshold_;
    std::vector<UMat> similarities_;
};

/** @brief Exposure compensator which tries to remove exposure related artifacts by adjusting image
intensities on each channel independently.
 */
class CV_EXPORTS_W ChannelsCompensator : public ExposureCompensator
{
public:
    CV_WRAP ChannelsCompensator(int nr_feeds=1)
        : nr_feeds_(nr_feeds), similarity_threshold_(1) {}
    void feed(const std::vector<Point> &corners, const std::vector<UMat> &images,
              const std::vector<std::pair<UMat,uchar> > &masks) CV_OVERRIDE;
    CV_WRAP void apply(int index, Point corner, InputOutputArray image, InputArray mask) CV_OVERRIDE;
    CV_WRAP void getMatGains(CV_OUT std::vector<Mat>& umv) CV_OVERRIDE;
    CV_WRAP void setMatGains(std::vector<Mat>& umv) CV_OVERRIDE;
    CV_WRAP void setNrFeeds(int nr_feeds) { nr_feeds_ = nr_feeds; }
    CV_WRAP int getNrFeeds() { return nr_feeds_; }
    CV_WRAP void setSimilarityThreshold(double similarity_threshold) { similarity_threshold_ = similarity_threshold; }
    CV_WRAP double getSimilarityThreshold() const { return similarity_threshold_; }
    std::vector<Scalar> gains() const { return gains_; }

private:
    std::vector<Scalar> gains_;
    int nr_feeds_;
    double similarity_threshold_;
};

/** @brief Exposure compensator which tries to remove exposure related artifacts by adjusting image blocks.
 */
class CV_EXPORTS_W BlocksCompensator : public ExposureCompensator
{
public:
    BlocksCompensator(int bl_width=32, int bl_height=32, int nr_feeds=1)
            : bl_width_(bl_width), bl_height_(bl_height), nr_feeds_(nr_feeds), nr_gain_filtering_iterations_(2),
              similarity_threshold_(1) {}
    CV_WRAP void apply(int index, Point corner, InputOutputArray image, InputArray mask) CV_OVERRIDE;
    CV_WRAP void getMatGains(CV_OUT std::vector<Mat>& umv) CV_OVERRIDE;
    CV_WRAP void setMatGains(std::vector<Mat>& umv) CV_OVERRIDE;
    CV_WRAP void setNrFeeds(int nr_feeds) { nr_feeds_ = nr_feeds; }
    CV_WRAP int getNrFeeds() { return nr_feeds_; }
    CV_WRAP void setSimilarityThreshold(double similarity_threshold) { similarity_threshold_ = similarity_threshold; }
    CV_WRAP double getSimilarityThreshold() const { return similarity_threshold_; }
    CV_WRAP void setBlockSize(int width, int height) { bl_width_ = width; bl_height_ = height; }
    CV_WRAP void setBlockSize(Size size) { setBlockSize(size.width, size.height); }
    CV_WRAP Size getBlockSize() const { return Size(bl_width_, bl_height_); }
    CV_WRAP void setNrGainsFilteringIterations(int nr_iterations) { nr_gain_filtering_iterations_ = nr_iterations; }
    CV_WRAP int getNrGainsFilteringIterations() const { return nr_gain_filtering_iterations_; }

protected:
    template<class Compensator>
    void feed(const std::vector<Point> &corners, const std::vector<UMat> &images,
              const std::vector<std::pair<UMat,uchar> > &masks);

private:
    UMat getGainMap(const GainCompensator& compensator, int bl_idx, Size bl_per_img);
    UMat getGainMap(const ChannelsCompensator& compensator, int bl_idx, Size bl_per_img);

    int bl_width_, bl_height_;
    std::vector<UMat> gain_maps_;
    int nr_feeds_;
    int nr_gain_filtering_iterations_;
    double similarity_threshold_;
};

/** @brief Exposure compensator which tries to remove exposure related artifacts by adjusting image block
intensities, see @cite UES01 for details.
 */
class CV_EXPORTS_W BlocksGainCompensator : public BlocksCompensator
{
public:
    // This Constructor only exists to make source level compatibility detector happy
    CV_WRAP BlocksGainCompensator(int bl_width = 32, int bl_height = 32)
            : BlocksGainCompensator(bl_width, bl_height, 1) {}
    CV_WRAP BlocksGainCompensator(int bl_width, int bl_height, int nr_feeds)
            : BlocksCompensator(bl_width, bl_height, nr_feeds) {}

    void feed(const std::vector<Point> &corners, const std::vector<UMat> &images,
              const std::vector<std::pair<UMat,uchar> > &masks) CV_OVERRIDE;

    // This function only exists to make source level compatibility detector happy
    CV_WRAP void apply(int index, Point corner, InputOutputArray image, InputArray mask) CV_OVERRIDE {
        BlocksCompensator::apply(index, corner, image, mask); }
    // This function only exists to make source level compatibility detector happy
    CV_WRAP void getMatGains(CV_OUT std::vector<Mat>& umv) CV_OVERRIDE { BlocksCompensator::getMatGains(umv); }
    // This function only exists to make source level compatibility detector happy
    CV_WRAP void setMatGains(std::vector<Mat>& umv) CV_OVERRIDE { BlocksCompensator::setMatGains(umv); }
};

/** @brief Exposure compensator which tries to remove exposure related artifacts by adjusting image block
on each channel.
 */
class CV_EXPORTS_W BlocksChannelsCompensator : public BlocksCompensator
{
public:
    CV_WRAP BlocksChannelsCompensator(int bl_width=32, int bl_height=32, int nr_feeds=1)
            : BlocksCompensator(bl_width, bl_height, nr_feeds) {}

    void feed(const std::vector<Point> &corners, const std::vector<UMat> &images,
              const std::vector<std::pair<UMat,uchar> > &masks) CV_OVERRIDE;
};
//! @}

} // namespace detail
} // namespace cv

#endif // OPENCV_STITCHING_EXPOSURE_COMPENSATE_HPP
