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

#include <precomp.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core.hpp>

#include <vector>
#include <string>
#include <iostream>
#include <cstdio>
#include <stdarg.h>

// used for noisy printfs
// #define WITH_DEBUG_OUT

#if defined WITH_DEBUG_OUT
# define dprintf(format, ...) \
            do { printf(format, ##__VA_ARGS__); } while (0)
#else
# define dprintf(format, ...)
#endif

namespace {

struct Octave
{
    int index;
    float scale;
    int stages;
    cv::Size size;
    int shrinkage;

    static const char *const SC_OCT_SCALE;
    static const char *const SC_OCT_STAGES;
    static const char *const SC_OCT_SHRINKAGE;

    Octave(const int i, cv::Size origObjSize, const cv::FileNode& fn)
    : index(i), scale((float)fn[SC_OCT_SCALE]), stages((int)fn[SC_OCT_STAGES]),
      size(cvRound(origObjSize.width * scale), cvRound(origObjSize.height * scale)),
      shrinkage((int)fn[SC_OCT_SHRINKAGE])
    {}
};

const char *const Octave::SC_OCT_SCALE     = "scale";
const char *const Octave::SC_OCT_STAGES    = "stageNum";
const char *const Octave::SC_OCT_SHRINKAGE = "shrinkingFactor";

struct Weak
{
    float threshold;

    static const char *const SC_STAGE_THRESHOLD;

    Weak(){}
    Weak(const cv::FileNode& fn) : threshold((float)fn[SC_STAGE_THRESHOLD]){}
};

const char *const Weak::SC_STAGE_THRESHOLD  = "stageThreshold";

struct Node
{
    int feature;
    float threshold;

    Node(){}
    Node(const int offset, cv::FileNodeIterator& fIt)
    : feature((int)(*(fIt +=2)++) + offset), threshold((float)(*(fIt++))){}
};

struct Feature
{
    int channel;
    cv::Rect rect;
    float rarea;

    static const char * const SC_F_CHANNEL;
    static const char * const SC_F_RECT;

    Feature() {}
    Feature(const cv::FileNode& fn) : channel((int)fn[SC_F_CHANNEL])
    {
        cv::FileNode rn = fn[SC_F_RECT];
        cv::FileNodeIterator r_it = rn.end();
        rect = cv::Rect(*(--r_it), *(--r_it), *(--r_it), *(--r_it));

        // 1 / area
        rarea = 1.f / ((rect.width - rect.x) * (rect.height - rect.y));
    }
};

const char * const Feature::SC_F_CHANNEL   = "channel";
const char * const Feature::SC_F_RECT      = "rect";

struct CascadeIntrinsics
{
    static const float lambda = 1.099f, a = 0.89f;

    static float getFor(bool isUp, float scaling)
    {
        if (fabs(scaling - 1.f) < FLT_EPSILON)
            return 1.f;

        // according to R. Benenson, M. Mathias, R. Timofte and L. Van Gool's and Dallal's papers
        static const float A[2][2] =
        {   //channel <= 6, otherwise
            {        0.89f, 1.f}, // down
            {        1.00f, 1.f}  // up
        };

        static const float B[2][2] =
        {   //channel <= 6,  otherwise
            { 1.099f / log(2), 2.f}, // down
            {             0.f, 2.f}  // up
        };

        float a = A[(int)(scaling >= 1)][(int)(isUp)];
        float b = B[(int)(scaling >= 1)][(int)(isUp)];

        dprintf("scaling: %f %f %f %f\n", scaling, a, b, a * pow(scaling, b));
        return a * pow(scaling, b);
    }
};

struct Level
{
    const Octave* octave;

    float origScale;
    float relScale;
    int scaleshift;

    cv::Size workRect;
    cv::Size objSize;

    enum { R_SHIFT = 1 << 15 };

    float scaling[2];
    typedef cv::SoftCascade::Detection detection_t;

    Level(const Octave& oct, const float scale, const int shrinkage, const int w, const int h)
    :  octave(&oct), origScale(scale), relScale(scale / oct.scale),
       workRect(cv::Size(cvRound(w / (float)shrinkage),cvRound(h / (float)shrinkage))),
       objSize(cv::Size(cvRound(oct.size.width * relScale), cvRound(oct.size.height * relScale)))
    {
        scaling[0] = CascadeIntrinsics::getFor(false, relScale) / (relScale * relScale);
        scaling[1] = CascadeIntrinsics::getFor(true, relScale) / (relScale * relScale);
        scaleshift = relScale * (1 << 16);
    }

    void addDetection(const int x, const int y, float confidence, std::vector<detection_t>& detections) const
    {
        int shrinkage = (*octave).shrinkage;
        cv::Rect rect(cvRound(x * shrinkage), cvRound(y * shrinkage), objSize.width, objSize.height);

        detections.push_back(detection_t(rect, confidence));
    }

    float rescale(cv::Rect& scaledRect, const float threshold, int idx) const
    {
        // rescale
        scaledRect.x      = (scaleshift * scaledRect.x + R_SHIFT) >> 16;
        scaledRect.y      = (scaleshift * scaledRect.y + R_SHIFT) >> 16;
        scaledRect.width  = (scaleshift * scaledRect.width + R_SHIFT) >> 16;
        scaledRect.height = (scaleshift * scaledRect.height + R_SHIFT) >> 16;

        float sarea = (scaledRect.width - scaledRect.x) * (scaledRect.height - scaledRect.y);

        // compensation areas rounding
        return (sarea == 0.0f)? threshold : (threshold * scaling[idx] * sarea);
    }
};

struct ChannelStorage
{
    std::vector<cv::Mat> hog;
    int shrinkage;
    int offset;
    int step;

    enum {HOG_BINS = 6, HOG_LUV_BINS = 10};

    ChannelStorage() {}
    ChannelStorage(const cv::Mat& colored, int shr) : shrinkage(shr)
    {
        hog.clear();
        cv::IntegralChannels ints(shr);

        // convert to grey
        cv::Mat grey;
        cv::cvtColor(colored, grey, CV_BGR2GRAY);

        ints.createHogBins(grey, hog, 6);
        ints.createLuvBins(colored, hog);

        step = hog[0].cols;
    }

    float get(const int channel, const cv::Rect& area) const
    {
        // CV_Assert(channel < HOG_LUV_BINS);
        const cv::Mat& m = hog[channel];
        int *ptr = ((int*)(m.data)) + offset;

        int a = ptr[area.y * step + area.x];
        int b = ptr[area.y * step + area.width];
        int c = ptr[area.height * step + area.width];
        int d = ptr[area.height * step + area.x];

        return (a - b + c - d);
    }
};

}

struct cv::SoftCascade::Filds
{
    float minScale;
    float maxScale;

    int origObjWidth;
    int origObjHeight;

    int shrinkage;

    std::vector<Octave>  octaves;
    std::vector<Weak>   stages;
    std::vector<Node>    nodes;
    std::vector<float>   leaves;
    std::vector<Feature> features;

    std::vector<Level> levels;

    cv::Size frameSize;

    enum { BOOST = 0 };

    typedef std::vector<Octave>::iterator  octIt_t;

    void detectAt(const int dx, const int dy, const Level& level, const ChannelStorage& storage,
        std::vector<Detection>& detections) const
    {
        dprintf("detect at: %d %d\n", dx, dy);

        float detectionScore = 0.f;

        const Octave& octave = *(level.octave);
        int stBegin = octave.index * octave.stages, stEnd = stBegin + 1024;//octave.stages;

        dprintf("  octave stages: %d to %d index %d %f level %f\n",
            stBegin, stEnd, octave.index, octave.scale, level.origScale);

        int st = stBegin;
        for(; st < stEnd; ++st)
        {

            dprintf("index: %d\n", st);

            const Weak& stage = stages[st];
            {
                int nId = st * 3;

                // work with root node
                const Node& node = nodes[nId];
                const Feature& feature = features[node.feature];
                cv::Rect scaledRect(feature.rect);

                float threshold = level.rescale(scaledRect, node.threshold,(int)(feature.channel > 6)) * feature.rarea;

                float sum = storage.get(feature.channel, scaledRect);

                dprintf("root feature %d %f\n",feature.channel, sum);

                int next = (sum >= threshold)? 2 : 1;

                dprintf("go: %d (%f >= %f)\n\n" ,next, sum, threshold);

                // leaves
                const Node& leaf = nodes[nId + next];
                const Feature& fLeaf = features[leaf.feature];

                scaledRect = fLeaf.rect;
                threshold = level.rescale(scaledRect, leaf.threshold, (int)(fLeaf.channel > 6)) * fLeaf.rarea;

                sum = storage.get(fLeaf.channel, scaledRect);

                int lShift = (next - 1) * 2 + ((sum >= threshold) ? 1 : 0);
                float impact = leaves[(st * 4) + lShift];

                dprintf("decided: %d (%f >= %f) %d %f\n\n" ,next, sum, threshold, lShift, impact);

                detectionScore += impact;
            }

            dprintf("extracted stage:\n");
            dprintf("ct %f\n", stage.threshold);
            dprintf("computed score %f\n\n", detectionScore);

#if defined WITH_DEBUG_OUT
            if (st - stBegin > 50   ) break;
#endif

            if (detectionScore <= stage.threshold) return;
        }

        dprintf("x %d y %d: %d\n", dx, dy, st - stBegin);
        dprintf("  got %d\n", st);

        level.addDetection(dx, dy, detectionScore, detections);
    }

    octIt_t fitOctave(const float& logFactor)
    {
        float minAbsLog = FLT_MAX;
        octIt_t res =  octaves.begin();
        for (octIt_t oct = octaves.begin(); oct < octaves.end(); ++oct)
        {
            const Octave& octave =*oct;
            float logOctave = log(octave.scale);
            float logAbsScale = fabs(logFactor - logOctave);

            if(logAbsScale < minAbsLog)
            {
                res = oct;
                minAbsLog = logAbsScale;
            }
        }
        return res;
    }

    // compute levels of full pyramid
    void calcLevels(const cv::Size& curr, int scales)
    {
        if (frameSize == curr) return;
        frameSize = curr;

        CV_Assert(scales > 1);
        levels.clear();
        float logFactor = (log(maxScale) - log(minScale)) / (scales -1);

        float scale = minScale;
        for (int sc = 0; sc < scales; ++sc)
        {
            int width  = std::max(0.0f, frameSize.width  - (origObjWidth  * scale));
            int height = std::max(0.0f, frameSize.height - (origObjHeight * scale));

            float logScale = log(scale);
            octIt_t fit = fitOctave(logScale);


            Level level(*fit, scale, shrinkage, width, height);

            if (!width || !height)
                break;
            else
                levels.push_back(level);

            if (fabs(scale - maxScale) < FLT_EPSILON) break;
            scale = std::min(maxScale, expf(log(scale) + logFactor));
        }
    }

    bool fill(const FileNode &root, const float mins, const float maxs)
    {
        minScale = mins;
        maxScale = maxs;

        // cascade properties
        static const char *const SC_STAGE_TYPE       = "stageType";
        static const char *const SC_BOOST            = "BOOST";

        static const char *const SC_FEATURE_TYPE     = "featureType";
        static const char *const SC_ICF              = "ICF";

        static const char *const SC_ORIG_W           = "width";
        static const char *const SC_ORIG_H           = "height";

        static const char *const SC_OCTAVES          = "octaves";
        static const char *const SC_STAGES           = "stages";
        static const char *const SC_FEATURES         = "features";

        static const char *const SC_WEEK             = "weakClassifiers";
        static const char *const SC_INTERNAL         = "internalNodes";
        static const char *const SC_LEAF             = "leafValues";


        // only Ada Boost supported
        std::string stageTypeStr = (string)root[SC_STAGE_TYPE];
        CV_Assert(stageTypeStr == SC_BOOST);

        // only HOG-like integral channel features cupported
        string featureTypeStr = (string)root[SC_FEATURE_TYPE];
        CV_Assert(featureTypeStr == SC_ICF);

        origObjWidth  = (int)root[SC_ORIG_W];
        origObjHeight = (int)root[SC_ORIG_H];

        // for each octave (~ one cascade in classic OpenCV xml)
        FileNode fn = root[SC_OCTAVES];
        if (fn.empty()) return false;

        // octaves.reserve(noctaves);
        FileNodeIterator it = fn.begin(), it_end = fn.end();
        int feature_offset = 0;
        int octIndex = 0;
        for (; it != it_end; ++it)
        {
            FileNode fns = *it;
            Octave octave(octIndex, cv::Size(origObjWidth, origObjHeight), fns);
            CV_Assert(octave.stages > 0);
            octaves.push_back(octave);

            FileNode ffs = fns[SC_FEATURES];
            if (ffs.empty()) return false;

            fns = fns[SC_STAGES];
            if (fn.empty()) return false;

            // for each stage (~ decision tree with H = 2)
            FileNodeIterator st = fns.begin(), st_end = fns.end();
            for (; st != st_end; ++st )
            {
                fns = *st;
                stages.push_back(Weak(fns));

                fns = fns[SC_WEEK];
                FileNodeIterator ftr = fns.begin(), ft_end = fns.end();
                for (; ftr != ft_end; ++ftr)
                {
                    fns = (*ftr)[SC_INTERNAL];
                    FileNodeIterator inIt = fns.begin(), inIt_end = fns.end();
                    for (; inIt != inIt_end;)
                        nodes.push_back(Node(feature_offset, inIt));

                    fns = (*ftr)[SC_LEAF];
                    inIt = fns.begin(), inIt_end = fns.end();
                    for (; inIt != inIt_end; ++inIt)
                        leaves.push_back((float)(*inIt));
                }
            }

            st = ffs.begin(), st_end = ffs.end();
            for (; st != st_end; ++st )
                features.push_back(Feature(*st));

            feature_offset += octave.stages * 3;
            ++octIndex;
        }

        shrinkage = octaves[0].shrinkage;
        return true;
    }
};

cv::SoftCascade::SoftCascade(const float mins, const float maxs, const int nsc)
: filds(0), minScale(mins), maxScale(maxs), scales(nsc) {}

cv::SoftCascade::SoftCascade(const cv::FileStorage& fs) : filds(0)
{
    read(fs);
}
cv::SoftCascade::~SoftCascade()
{
    delete filds;
}

bool cv::SoftCascade::read( const cv::FileStorage& fs)
{
    if (!fs.isOpened()) return false;

    if (filds)
        delete filds;
    filds = 0;

    filds = new Filds;
    Filds& flds = *filds;
    return flds.fill(fs.getFirstTopLevelNode(), minScale, maxScale);
}

void cv::SoftCascade::detectMultiScale(const Mat& image, const std::vector<cv::Rect>& /*rois*/,
                                       std::vector<Detection>& objects, const int /*rejectfactor*/) const
{
    // only color images are supperted
    CV_Assert(image.type() == CV_8UC3);

    Filds& fld = *filds;
    fld.calcLevels(image.size(), scales);

    objects.clear();

    // create integrals
    ChannelStorage storage(image, fld.shrinkage);

    typedef std::vector<Level>::const_iterator lIt;
    for (lIt it = fld.levels.begin(); it != fld.levels.end(); ++it)
    {
        const Level& level = *it;

        for (int dy = 0; dy < level.workRect.height; ++dy)
        {
            for (int dx = 0; dx < level.workRect.width; ++dx)
            {
                storage.offset = dy * storage.step + dx;
                fld.detectAt(dx, dy, level, storage, objects);
            }
        }
    }
}