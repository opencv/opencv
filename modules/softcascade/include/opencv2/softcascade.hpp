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
// Copyright (C) 2008-2013, Willow Garage Inc., all rights reserved.
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
//     and / or other materials provided with the distribution.
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

#ifndef __OPENCV_SOFTCASCADE_HPP__
#define __OPENCV_SOFTCASCADE_HPP__

#include <iosfwd>

#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"

namespace cv { namespace softcascade {

// Representation of detectors result.
// We assume that image is less then 2^16x2^16.
struct CV_EXPORTS Detection
{
    // Creates Detection from an object bounding box and confidence.
    // Param b is a bounding box
    // Param c is a confidence that object belongs to class k
    // Param k is an object class
    Detection(const cv::Rect& b, const float c, int k = PEDESTRIAN);
    cv::Rect bb() const;
    enum {PEDESTRIAN = 1};

    ushort x;
    ushort y;
    ushort w;
    ushort h;
    float confidence;
    int kind;
};

class CV_EXPORTS Dataset
{
public:
    typedef enum {POSITIVE = 1, NEGATIVE = 2} SampleType;

    virtual cv::Mat get(SampleType type, int idx) const = 0;
    virtual int available(SampleType type) const = 0;
    virtual ~Dataset();
};

// ========================================================================== //
//                       Public interface feature pool.
// ========================================================================== //

class CV_EXPORTS FeaturePool
{
public:

    virtual int size() const = 0;
    virtual float apply(int fi, int si, const Mat& channels) const = 0;
    virtual void write( cv::FileStorage& fs, int index) const = 0;
    virtual ~FeaturePool();

    static cv::Ptr<FeaturePool> create(const cv::Size& model, int nfeatures, int nchannels );
};

// ========================================================================== //
//                         First order channel feature.
// ========================================================================== //

class CV_EXPORTS ChannelFeature
{
public:
    ChannelFeature(int x, int y, int w, int h, int ch);
    ~ChannelFeature();

    bool operator ==(ChannelFeature b);
    bool operator !=(ChannelFeature b);

    float operator() (const cv::Mat& integrals, const cv::Size& model) const;

    friend void write(cv::FileStorage& fs, const String&, const ChannelFeature& f);
    friend std::ostream& operator<<(std::ostream& out, const ChannelFeature& f);

private:
    cv::Rect bb;
    int channel;
};

void write(cv::FileStorage& fs, const String&, const ChannelFeature& f);
std::ostream& operator<<(std::ostream& out, const ChannelFeature& m);

// ========================================================================== //
//             Public Interface for Integral Channel Feature.
// ========================================================================== //

class CV_EXPORTS_W ChannelFeatureBuilder : public cv::Algorithm
{
public:
    virtual ~ChannelFeatureBuilder();

    // apply channels to source frame
    CV_WRAP_AS(compute) virtual void operator()(InputArray src, OutputArray channels, cv::Size channelsSize = cv::Size()) const = 0;

    CV_WRAP virtual int totalChannels() const = 0;
    virtual cv::AlgorithmInfo* info() const = 0;

    CV_WRAP static cv::Ptr<ChannelFeatureBuilder> create(const String& featureType);
};

// ========================================================================== //
//             Implementation of soft (stageless) cascaded detector.
// ========================================================================== //
class CV_EXPORTS_W Detector : public cv::Algorithm
{
public:

    enum { NO_REJECT = 1, DOLLAR = 2, /*PASCAL = 4,*/ DEFAULT = NO_REJECT};

    // An empty cascade will be created.
    // Param minScale is a minimum scale relative to the original size of the image on which cascade will be applied.
    // Param minScale is a maximum scale relative to the original size of the image on which cascade will be applied.
    // Param scales is a number of scales from minScale to maxScale.
    // Param rejCriteria is used for NMS.
    CV_WRAP Detector(double minScale = 0.4, double maxScale = 5., int scales = 55, int rejCriteria = 1);

    CV_WRAP virtual ~Detector();

    cv::AlgorithmInfo* info() const;

    // Load soft cascade from FileNode.
    // Param fileNode is a root node for cascade.
    CV_WRAP virtual bool load(const FileNode& fileNode);

    // Load soft cascade config.
    CV_WRAP virtual void read(const FileNode& fileNode);

    // Return the vector of Detection objects.
    // Param image is a frame on which detector will be applied.
    // Param rois is a vector of regions of interest. Only the objects that fall into one of the regions will be returned.
    // Param objects is an output array of Detections
    virtual void detect(InputArray image, InputArray rois, std::vector<Detection>& objects) const;

    // Param rects is an output array of bounding rectangles for detected objects.
    // Param confs is an output array of confidence for detected objects. i-th bounding rectangle corresponds i-th confidence.
    CV_WRAP virtual void detect(InputArray image, InputArray rois, OutputArray rects, OutputArray confs) const;

private:
    void detectNoRoi(const Mat& image, std::vector<Detection>& objects) const;

    struct Fields;
    Fields* fields;

    double minScale;
    double maxScale;

    int   scales;
    int   rejCriteria;
};

// ========================================================================== //
//     Public Interface for singe soft (stageless) cascade octave training.
// ========================================================================== //
class CV_EXPORTS Octave : public cv::Algorithm
{
public:
    enum
    {
        // Direct backward pruning. (Cha Zhang and Paul Viola)
        DBP = 1,
        // Multiple instance pruning. (Cha Zhang and Paul Viola)
        MIP = 2,
        // Originally proposed by L. Bourdev and J. Brandt
        HEURISTIC = 4
    };

    virtual ~Octave();
    static cv::Ptr<Octave> create(cv::Rect boundingBox, int npositives, int nnegatives,
        int logScale, int shrinkage, cv::Ptr<ChannelFeatureBuilder> builder);

    virtual bool train(const Dataset* dataset, const FeaturePool* pool, int weaks, int treeDepth) = 0;
    virtual void setRejectThresholds(OutputArray thresholds) = 0;
    virtual void write( cv::FileStorage &fs, const FeaturePool* pool, InputArray thresholds) const = 0;
    virtual void write( CvFileStorage* fs, String name) const = 0;
};

CV_EXPORTS bool initModule_softcascade(void);

// ======================== CUDA version for soft cascade ===================== //

class CV_EXPORTS ChannelsProcessor
{
public:
    enum
    {
        // GENERIC   = 1 << 4, does not supported
        SEPARABLE = 2 << 4
    };

    // Appends specified number of HOG first-order features integrals into given vector.
    // Param frame is an input 3-channel bgr image.
    // Param channels is a GPU matrix of optionally shrinked channels
    // Param stream is stream is a high-level CUDA stream abstraction used for asynchronous execution.
    virtual void apply(InputArray frame, OutputArray channels, cv::cuda::Stream& stream = cv::cuda::Stream::Null()) = 0;

    // Creates a specific preprocessor implementation.
    // Param shrinkage is a resizing factor. Resize is applied before the computing integral sum
    // Param bins is a number of HOG-like channels.
    // Param flags is a channel computing extra flags.
    static cv::Ptr<ChannelsProcessor> create(const int shrinkage, const int bins, const int flags = SEPARABLE);

    virtual ~ChannelsProcessor();

protected:
    ChannelsProcessor();
};

// Implementation of soft (stage-less) cascaded detector.
class CV_EXPORTS SCascade : public cv::Algorithm
{
public:

    enum { NO_REJECT = 1, DOLLAR = 2, /*PASCAL = 4,*/ DEFAULT = NO_REJECT, NMS_MASK = 0xF};

    // An empty cascade will be created.
    // Param minScale is a minimum scale relative to the original size of the image on which cascade will be applied.
    // Param minScale is a maximum scale relative to the original size of the image on which cascade will be applied.
    // Param scales is a number of scales from minScale to maxScale.
    // Param flags is an extra tuning flags.
    SCascade(const double minScale = 0.4, const double maxScale = 5., const int scales = 55,
        const int flags = NO_REJECT | ChannelsProcessor::SEPARABLE);

    virtual ~SCascade();

    cv::AlgorithmInfo* info() const;

    // Load cascade from FileNode.
    // Param fn is a root node for cascade. Should be <cascade>.
    virtual bool load(const FileNode& fn);

    // Load cascade config.
    virtual void read(const FileNode& fn);

    // Return the matrix of of detected objects.
    // Param image is a frame on which detector will be applied.
    // Param rois is a regions of interests mask generated by genRoi.
    //    Only the objects that fall into one of the regions will be returned.
    // Param objects is an output array of Detections represented as GpuMat of detections (SCascade::Detection)
    //    The first element of the matrix is  actually a count of detections.
    // Param stream is stream is a high-level CUDA stream abstraction used for asynchronous execution
    virtual void detect(InputArray image, InputArray rois, OutputArray objects, cv::cuda::Stream& stream = cv::cuda::Stream::Null()) const;

private:

    struct Fields;
    Fields* fields;

    double minScale;
    double maxScale;
    int scales;

    int flags;
};


}} // namespace cv { namespace softcascade {

#endif
