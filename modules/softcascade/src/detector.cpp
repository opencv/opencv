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

#include "precomp.hpp"

cv::softcascade::Detection::Detection(const cv::Rect& b, const float c, int k)
: x(static_cast<ushort>(b.x)), y(static_cast<ushort>(b.y)),
  w(static_cast<ushort>(b.width)), h(static_cast<ushort>(b.height)), confidence(c), kind(k) {}

cv::Rect cv::softcascade::Detection::bb() const
{
    return cv::Rect(x, y, w, h);
}

namespace {

struct SOctave
{
    SOctave(const int i, const cv::Size& origObjSize, const cv::FileNode& fn)
    : index(i), weaks((int)fn[SC_OCT_WEAKS]), scale((float)std::pow(2,(float)fn[SC_OCT_SCALE])),
      size(cvRound(origObjSize.width * scale), cvRound(origObjSize.height * scale)) {}

    int   index;
    int   weaks;

    float scale;

    cv::Size size;

    static const char *const SC_OCT_SCALE;
    static const char *const SC_OCT_WEAKS;
    static const char *const SC_OCT_SHRINKAGE;
};


struct Weak
{
    Weak(){}
    Weak(const cv::FileNode& fn) : threshold((float)fn[SC_WEAK_THRESHOLD]) {}

    float threshold;

    static const char *const SC_WEAK_THRESHOLD;
};


struct Node
{
    Node(){}
    Node(const int offset, cv::FileNodeIterator& fIt)
    : feature((int)(*(fIt +=2)++) + offset), threshold((float)(*(fIt++))) {}

    int   feature;
    float threshold;
};

struct Feature
{
    Feature() {}
    Feature(const cv::FileNode& fn, bool useBoxes = false) : channel((int)fn[SC_F_CHANNEL])
    {
        cv::FileNode rn = fn[SC_F_RECT];
        cv::FileNodeIterator r_it = rn.begin();

        int x = *r_it++;
        int y = *r_it++;
        int w = *r_it++;
        int h = *r_it++;

        // ToDo: fix me
        if (useBoxes)
            rect = cv::Rect(x, y, w, h);
        else
            rect = cv::Rect(x, y, w + x, h + y);

        // 1 / area
        rarea = 1.f / ((rect.width - rect.x) * (rect.height - rect.y));
    }

    int channel;
    cv::Rect rect;
    float rarea;

    static const char *const SC_F_CHANNEL;
    static const char *const SC_F_RECT;
};

const char *const SOctave::SC_OCT_SCALE      = "scale";
const char *const SOctave::SC_OCT_WEAKS      = "weaks";
const char *const SOctave::SC_OCT_SHRINKAGE  = "shrinkingFactor";
const char *const Weak::SC_WEAK_THRESHOLD   = "treeThreshold";
const char *const Feature::SC_F_CHANNEL     = "channel";
const char *const Feature::SC_F_RECT        = "rect";

struct Level
{
    const SOctave* octave;

    float origScale;
    float relScale;
    int scaleshift;

    cv::Size workRect;
    cv::Size objSize;

    float scaling[2]; // 0-th for channels <= 6, 1-st otherwise

    Level(const SOctave& oct, const float scale, const int shrinkage, const int w, const int h)
    :  octave(&oct), origScale(scale), relScale(scale / oct.scale),
       workRect(cv::Size(cvRound(w / (float)shrinkage),cvRound(h / (float)shrinkage))),
       objSize(cv::Size(cvRound(oct.size.width * relScale), cvRound(oct.size.height * relScale)))
    {
        scaling[0] = ((relScale >= 1.f)? 1.f : (0.89f * std::pow(relScale, 1.099f / std::log(2.f)))) / (relScale * relScale);
        scaling[1] = 1.f;
        scaleshift = static_cast<int>(relScale * (1 << 16));
    }

    void addDetection(const int x, const int y, float confidence, std::vector<cv::softcascade::Detection>& detections) const
    {
        // fix me
        int shrinkage = 4;//(*octave).shrinkage;
        cv::Rect rect(cvRound(x * shrinkage), cvRound(y * shrinkage), objSize.width, objSize.height);

        detections.push_back(cv::softcascade::Detection(rect, confidence));
    }

    float rescale(cv::Rect& scaledRect, const float threshold, int idx) const
    {
#define SSHIFT(a) ((a) + (1 << 15)) >> 16
        // rescale
        scaledRect.x      = SSHIFT(scaleshift * scaledRect.x);
        scaledRect.y      = SSHIFT(scaleshift * scaledRect.y);
        scaledRect.width  = SSHIFT(scaleshift * scaledRect.width);
        scaledRect.height = SSHIFT(scaleshift * scaledRect.height);
#undef SSHIFT
        float sarea = static_cast<float>((scaledRect.width - scaledRect.x) * (scaledRect.height - scaledRect.y));

        // compensation areas rounding
        return (sarea == 0.0f)? threshold : (threshold * scaling[idx] * sarea);
    }
};
struct ChannelStorage
{
    cv::Mat hog;
    int shrinkage;
    int offset;
    size_t step;
    int model_height;

    cv::Ptr<cv::softcascade::ChannelFeatureBuilder> builder;

    enum {HOG_BINS = 6, HOG_LUV_BINS = 10};

    ChannelStorage(const cv::Mat& colored, int shr, cv::String featureTypeStr) : shrinkage(shr)
    {
        model_height = cvRound(colored.rows / (float)shrinkage);
        if (featureTypeStr == "ICF") featureTypeStr = "HOG6MagLuv";

        builder = cv::softcascade::ChannelFeatureBuilder::create(featureTypeStr);
        (*builder)(colored, hog, cv::Size(cvRound(colored.cols / (float)shrinkage), model_height));

        step = hog.step1();
    }

    float get(const int channel, const cv::Rect& area) const
    {
        const int *ptr = hog.ptr<const int>(0) + model_height * channel * step + offset;

        int a = ptr[area.y      * step + area.x];
        int b = ptr[area.y      * step + area.width];
        int c = ptr[area.height * step + area.width];
        int d = ptr[area.height * step + area.x];

        return static_cast<float>(a - b + c - d);
    }
};

}

struct cv::softcascade::Detector::Fields
{
    float minScale;
    float maxScale;
    int scales;

    int origObjWidth;
    int origObjHeight;

    int shrinkage;

    std::vector<SOctave> octaves;
    std::vector<Weak>    weaks;
    std::vector<Node>    nodes;
    std::vector<float>   leaves;
    std::vector<Feature> features;

    std::vector<Level> levels;

    cv::Size frameSize;

    typedef std::vector<SOctave>::iterator  octIt_t;
    typedef std::vector<Detection> dvector;

    String featureTypeStr;

    void detectAt(const int dx, const int dy, const Level& level, const ChannelStorage& storage, dvector& detections) const
    {
        float detectionScore = 0.f;

        const SOctave& octave = *(level.octave);

        int stBegin = octave.index * octave.weaks, stEnd = stBegin + octave.weaks;

        for(int st = stBegin; st < stEnd; ++st)
        {
            const Weak& weak = weaks[st];

            int nId = st * 3;

            // work with root node
            const Node& node = nodes[nId];
            const Feature& feature = features[node.feature];

            cv::Rect scaledRect(feature.rect);

            float threshold = level.rescale(scaledRect, node.threshold, (int)(feature.channel > 6)) * feature.rarea;
            float sum = storage.get(feature.channel, scaledRect);
            int next = (sum >= threshold)? 2 : 1;

            // leaves
            const Node& leaf = nodes[nId + next];
            const Feature& fLeaf = features[leaf.feature];

            scaledRect = fLeaf.rect;
            threshold = level.rescale(scaledRect, leaf.threshold, (int)(fLeaf.channel > 6)) * fLeaf.rarea;
            sum = storage.get(fLeaf.channel, scaledRect);

            int lShift = (next - 1) * 2 + ((sum >= threshold) ? 1 : 0);
            float impact = leaves[(st * 4) + lShift];

            detectionScore += impact;

            if (detectionScore <= weak.threshold) return;
        }

        if (detectionScore > 0)
            level.addDetection(dx, dy, detectionScore, detections);
    }

    octIt_t fitOctave(const float& logFactor)
    {
        float minAbsLog = FLT_MAX;
        octIt_t res =  octaves.begin();
        for (octIt_t oct = octaves.begin(); oct < octaves.end(); ++oct)
        {
            const SOctave& octave =*oct;
            float logOctave = std::log(octave.scale);
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
    void calcLevels(const cv::Size& curr, float mins, float maxs, int total)
    {
        if (frameSize == curr && maxs == maxScale && mins == minScale && total == scales) return;

        frameSize = curr;
        maxScale = maxs; minScale = mins; scales = total;
        CV_Assert(scales > 1);

        levels.clear();
        float logFactor = (std::log(maxScale) - std::log(minScale)) / (scales -1);

        float scale = minScale;
        for (int sc = 0; sc < scales; ++sc)
        {
            int width  = static_cast<int>(std::max(0.0f, frameSize.width  - (origObjWidth  * scale)));
            int height = static_cast<int>(std::max(0.0f, frameSize.height - (origObjHeight * scale)));

            float logScale = std::log(scale);
            octIt_t fit = fitOctave(logScale);


            Level level(*fit, scale, shrinkage, width, height);

            if (!width || !height)
                break;
            else
                levels.push_back(level);

            if (fabs(scale - maxScale) < FLT_EPSILON) break;
            scale = std::min(maxScale, expf(std::log(scale) + logFactor));
        }
    }

    bool fill(const FileNode &root)
    {
        // cascade properties
        static const char *const SC_STAGE_TYPE       = "stageType";
        static const char *const SC_BOOST            = "BOOST";

        static const char *const SC_FEATURE_TYPE     = "featureType";
        static const char *const SC_HOG6_MAG_LUV     = "HOG6MagLuv";
        static const char *const SC_ICF              = "ICF";

        static const char *const SC_ORIG_W           = "width";
        static const char *const SC_ORIG_H           = "height";

        static const char *const SC_OCTAVES          = "octaves";
        static const char *const SC_TREES            = "trees";
        static const char *const SC_FEATURES         = "features";

        static const char *const SC_INTERNAL         = "internalNodes";
        static const char *const SC_LEAF             = "leafValues";

        static const char *const SC_SHRINKAGE        = "shrinkage";

        static const char *const FEATURE_FORMAT      = "featureFormat";

        // only Ada Boost supported
        String stageTypeStr = (String)root[SC_STAGE_TYPE];
        CV_Assert(stageTypeStr == SC_BOOST);

        String fformat = (String)root[FEATURE_FORMAT];
        bool useBoxes = (fformat == "BOX");

        // only HOG-like integral channel features supported
        featureTypeStr = (String)root[SC_FEATURE_TYPE];
        CV_Assert(featureTypeStr == SC_ICF || featureTypeStr == SC_HOG6_MAG_LUV);

        origObjWidth  = (int)root[SC_ORIG_W];
        origObjHeight = (int)root[SC_ORIG_H];

        shrinkage = (int)root[SC_SHRINKAGE];

        FileNode fn = root[SC_OCTAVES];
        if (fn.empty()) return false;

        // for each octave
        FileNodeIterator it = fn.begin(), it_end = fn.end();
        for (int octIndex = 0; it != it_end; ++it, ++octIndex)
        {
            FileNode fns = *it;
            SOctave octave(octIndex, cv::Size(origObjWidth, origObjHeight), fns);
            CV_Assert(octave.weaks > 0);
            octaves.push_back(octave);

            FileNode ffs = fns[SC_FEATURES];
            if (ffs.empty()) return false;

            fns = fns[SC_TREES];
            if (fn.empty()) return false;

            FileNodeIterator st = fns.begin(), st_end = fns.end();
            for (; st != st_end; ++st )
            {
                weaks.push_back(Weak(*st));

                fns = (*st)[SC_INTERNAL];
                FileNodeIterator inIt = fns.begin(), inIt_end = fns.end();
                for (; inIt != inIt_end;)
                    nodes.push_back(Node((int)features.size(), inIt));

                fns = (*st)[SC_LEAF];
                inIt = fns.begin(), inIt_end = fns.end();

                for (; inIt != inIt_end; ++inIt)
                    leaves.push_back((float)(*inIt));
            }

            st = ffs.begin(), st_end = ffs.end();
            for (; st != st_end; ++st )
                features.push_back(Feature(*st, useBoxes));
        }

        return true;
    }
};

cv::softcascade::Detector::Detector(const double mins, const double maxs, const int nsc, const int rej)
: fields(0), minScale(mins), maxScale(maxs), scales(nsc), rejCriteria(rej) {}

cv::softcascade::Detector::~Detector() { delete fields;}

void cv::softcascade::Detector::read(const cv::FileNode& fn)
{
    Algorithm::read(fn);
}

bool cv::softcascade::Detector::load(const cv::FileNode& fn)
{
    if (fields) delete fields;

    fields = new Fields;
    return fields->fill(fn);
}

namespace {

using cv::softcascade::Detection;
typedef std::vector<Detection>  dvector;


struct ConfidenceGt
{
    bool operator()(const Detection& a, const Detection& b) const
    {
        return a.confidence > b.confidence;
    }
};

static float overlap(const cv::Rect &a, const cv::Rect &b)
{
    int w = std::min(a.x + a.width,  b.x + b.width)  - std::max(a.x, b.x);
    int h = std::min(a.y + a.height, b.y + b.height) - std::max(a.y, b.y);

    return (w < 0 || h < 0)? 0.f : (float)(w * h);
}

void DollarNMS(dvector& objects)
{
    static const float DollarThreshold = 0.65f;
    std::sort(objects.begin(), objects.end(), ConfidenceGt());

    for (dvector::iterator dIt = objects.begin(); dIt != objects.end(); ++dIt)
    {
        const Detection &a = *dIt;
        for (dvector::iterator next = dIt + 1; next != objects.end(); )
        {
            const Detection &b = *next;

            const float ovl =  overlap(a.bb(), b.bb()) / std::min(a.bb().area(), b.bb().area());

            if (ovl > DollarThreshold)
                next = objects.erase(next);
            else
                ++next;
        }
    }
}

static void suppress(int type, std::vector<Detection>& objects)
{
    CV_Assert(type == cv::softcascade::Detector::DOLLAR);
    DollarNMS(objects);
}

}

void cv::softcascade::Detector::detectNoRoi(const cv::Mat& image, std::vector<Detection>& objects) const
{
    Fields& fld = *fields;
    // create integrals
    ChannelStorage storage(image, fld.shrinkage, fld.featureTypeStr);

    typedef std::vector<Level>::const_iterator lIt;
    for (lIt it = fld.levels.begin(); it != fld.levels.end(); ++it)
    {
        const Level& level = *it;

        // we train only 3 scales.
        if (level.origScale > 2.5) break;

        for (int dy = 0; dy < level.workRect.height; ++dy)
        {
            for (int dx = 0; dx < level.workRect.width; ++dx)
            {
                storage.offset = (int)(dy * storage.step + dx);
                fld.detectAt(dx, dy, level, storage, objects);
            }
        }
    }

    if (rejCriteria != NO_REJECT) suppress(rejCriteria, objects);
}

void cv::softcascade::Detector::detect(cv::InputArray _image, cv::InputArray _rois, std::vector<Detection>& objects) const
{
    // only color images are suppered
    cv::Mat image = _image.getMat();
    CV_Assert(image.type() == CV_8UC3);

    Fields& fld = *fields;
    fld.calcLevels(image.size(),(float) minScale, (float)maxScale, scales);

    objects.clear();

    if (_rois.empty())
        return detectNoRoi(image, objects);

    int shr = fld.shrinkage;

    cv::Mat roi = _rois.getMat();
    cv::Mat mask(image.rows / shr, image.cols / shr, CV_8UC1);

    mask.setTo(cv::Scalar::all(0));
    cv::Rect* r = roi.ptr<cv::Rect>(0);
    for (int i = 0; i < (int)roi.cols; ++i)
        cv::Mat(mask, cv::Rect(r[i].x / shr, r[i].y / shr, r[i].width / shr , r[i].height / shr)).setTo(cv::Scalar::all(1));

    // create integrals
    ChannelStorage storage(image, shr, fld.featureTypeStr);

    typedef std::vector<Level>::const_iterator lIt;
    for (lIt it = fld.levels.begin(); it != fld.levels.end(); ++it)
    {
         const Level& level = *it;

        // we train only 3 scales.
        if (level.origScale > 2.5) break;

         for (int dy = 0; dy < level.workRect.height; ++dy)
         {
             uchar* m  = mask.ptr<uchar>(dy);
             for (int dx = 0; dx < level.workRect.width; ++dx)
             {
                 if (m[dx])
                 {
                     storage.offset = (int)(dy * storage.step + dx);
                     fld.detectAt(dx, dy, level, storage, objects);
                 }
             }
         }
    }

    if (rejCriteria != NO_REJECT) suppress(rejCriteria, objects);
}

void cv::softcascade::Detector::detect(InputArray _image, InputArray _rois,  OutputArray _rects, OutputArray _confs) const
{
    std::vector<Detection> objects;
    detect( _image, _rois, objects);

    _rects.create(1, (int)objects.size(), CV_32SC4);
    cv::Mat_<cv::Rect> rects = (cv::Mat_<cv::Rect>)_rects.getMat();
    cv::Rect* rectPtr = rects.ptr<cv::Rect>(0);

    _confs.create(1, (int)objects.size(), CV_32F);
    cv::Mat confs = _confs.getMat();
    float* confPtr = confs.ptr<float>(0);

    typedef std::vector<Detection>::const_iterator IDet;

    int i = 0;
    for (IDet it = objects.begin(); it != objects.end(); ++it, ++i)
    {
        rectPtr[i] = (*it).bb();
        confPtr[i] = (*it).confidence;
    }
}
