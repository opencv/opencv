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

namespace {

char *itoa(long i, char* s, int /*dummy_radix*/)
{
    sprintf(s, "%ld", i);
    return s;
}

// used for noisy printfs
// #define WITH_DEBUG_OUT

#if defined WITH_DEBUG_OUT
# define dprintf(format, ...) \
            do { printf(format, __VA_ARGS__); } while (0)
#else
# define dprintf(format, ...)
#endif

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

struct Stage
{
    float threshold;

    static const char *const SC_STAGE_THRESHOLD;

    Stage(){}
    Stage(const cv::FileNode& fn) : threshold((float)fn[SC_STAGE_THRESHOLD]){}
};

const char *const Stage::SC_STAGE_THRESHOLD  = "stageThreshold";

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

    static const char * const SC_F_CHANNEL;
    static const char * const SC_F_RECT;

    Feature() {}
    Feature(const cv::FileNode& fn) : channel((int)fn[SC_F_CHANNEL])
    {
        cv::FileNode rn = fn[SC_F_RECT];
        cv::FileNodeIterator r_it = rn.end();
        rect = cv::Rect(*(--r_it), *(--r_it), *(--r_it), *(--r_it));
    }
};

const char * const Feature::SC_F_CHANNEL   = "channel";
const char * const Feature::SC_F_RECT      = "rect";

struct Object
{
    enum Class{PEDESTRIAN};
    cv::Rect rect;
    float confidence;
    Class detType;

    Object(const cv::Rect& r, const float c, Class dt = PEDESTRIAN) : rect(r), confidence(c), detType(dt) {}
};

struct Level
{
    const Octave* octave;

    float origScale;
    float relScale;
    float shrScale; // used for marking detection

    cv::Size workRect;
    cv::Size objSize;

    Level(const Octave& oct, const float scale, const int shrinkage, const int w, const int h)
    :  octave(&oct), origScale(scale), relScale(scale / oct.scale), shrScale (relScale / (float)shrinkage),
       workRect(cv::Size(cvRound(w / (float)shrinkage),cvRound(h / (float)shrinkage))),
       objSize(cv::Size(cvRound(oct.size.width * relScale), cvRound(oct.size.height * relScale)))
    {}

    void markDetection(const int x, const int y, float confidence, std::vector<Object>& detections) const
    {
        int shrinkage = (*octave).shrinkage;
        cv::Rect rect(cvRound(x * shrinkage), cvRound(y * shrinkage), objSize.width, objSize.height);

        detections.push_back(Object(rect, confidence));
    }
};

struct CascadeIntrinsics
{
    static const float lambda = 1.099f, a = 0.89f;

    static float getFor(int channel, float scaling)
    {
        CV_Assert(channel < 10);

        if (fabs(scaling - 1.f) < FLT_EPSILON)
        // if (scaling == 1.f)
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

        float a = A[(int)(scaling >= 1)][(int)(channel > 6)];
        float b = B[(int)(scaling >= 1)][(int)(channel > 6)];

        dprintf("scaling: %f %f %f %f\n", scaling, a, b, a * pow(scaling, b));
        return a * pow(scaling, b);
    }
};


int qangle6(float dfdx, float dfdy)
{
    static const float vectors[6][2] =
    {
        {std::cos(0),                std::sin(0)               },
        {std::cos(M_PI / 6.f),       std::sin(M_PI / 6.f)      },
        {std::cos(M_PI / 3.f),       std::sin(M_PI / 3.f)      },

        {std::cos(M_PI / 2.f),       std::sin(M_PI / 2.f)      },
        {std::cos(2.f * M_PI / 3.f), std::sin(2.f * M_PI / 3.f)},
        {std::cos(5.f * M_PI / 6.f), std::sin(5.f * M_PI / 6.f)}
    };

    int index = 0;

    float dot = fabs(dfdx * vectors[0][0] + dfdy * vectors[0][1]);

    for(int i = 1; i < 6; ++i)
    {
        const float curr = fabs(dfdx * vectors[i][0] + dfdy * vectors[i][1]);

        if(curr > dot)
        {
            dot = curr;
            index = i;
        }
    }

    return index;
}

template< typename T>
struct Decimate {
    int shrinkage;

    Decimate(const int sr) : shrinkage(sr) {}

    void operator()(const cv::Mat& in, cv::Mat& out) const
    {
        int cols = in.cols / shrinkage;
        int rows = in.rows / shrinkage;
        out.create(rows, cols, in.type());

        CV_Assert(cols * shrinkage == in.cols);
        CV_Assert(rows * shrinkage == in.rows);

        for (int outIdx_y = 0; outIdx_y < rows; ++outIdx_y)
        {
            T* outPtr = out.ptr<T>(outIdx_y);
            for (int outIdx_x = 0; outIdx_x < cols; ++outIdx_x)
            {
                // do desimate
                int inIdx_y = outIdx_y * shrinkage;
                int inIdx_x = outIdx_x * shrinkage;
                int sum = 0;

                for (int y = inIdx_y; y < inIdx_y + shrinkage; ++y)
                    for (int x = inIdx_x; x < inIdx_x + shrinkage; ++x)
                        sum += in.at<T>(y, x);

                sum /= shrinkage * shrinkage;
                outPtr[outIdx_x] = cv::saturate_cast<T>(sum);
            }
        }
    }

};

// use previous stored integrals for regression testing
// #define USE_REFERENCE_VALUES

struct ChannelStorage
{
    std::vector<cv::Mat> hog;
    int shrinkage;

    enum {HOG_BINS = 6, HOG_LUV_BINS = 10};

    ChannelStorage() {}
    ChannelStorage(const cv::Mat& colored, int shr) : shrinkage(shr)
    {
        hog.clear();
        Decimate<uchar> decimate(shr);

#if defined USE_REFERENCE_VALUES
        char buff[33];
        cv::FileStorage imgs("/home/kellan/testInts.xml", cv::FileStorage::READ);

        for(int i = 0; i < HOG_LUV_BINS; ++i)
        {
            cv::Mat channel;
            imgs[std::string("channel") + itoa(i, buff, 10)] >> channel;
            hog.push_back(channel);
        }
#else
        // add gauss
        cv::Mat gauss;
        cv::GaussianBlur(colored, gauss, cv::Size(3,3), 0 ,0);

        // convert to luv
        cv::Mat luv;
        cv::cvtColor(colored, luv, CV_RGB2Luv);

        // split to 3 one channel matrix
        std::vector<cv::Mat> splited, luvs;
        split(luv, splited);

        // shrink and integrate
        for (int i = 0; i < (int)splited.size(); i++)
        {
            cv::Mat shrunk, sum;
            decimate(splited[i], shrunk);
            cv::integral(shrunk, sum, cv::noArray(), CV_32S);
            luvs.push_back(sum);
        }

        // convert to grey
        cv::Mat grey;
        cv::cvtColor(colored, grey, CV_RGB2GRAY);

        // get derivative
        cv::Mat df_dx, df_dy, mag, angle;
        cv::Sobel(grey, df_dx, CV_32F, 1, 0);
        cv::Sobel(grey, df_dy, CV_32F, 0, 1);

        // normalize
        df_dx /= 4;
        df_dy /= 4;

        // calculate magnitude
        cv::cartToPolar(df_dx, df_dy, mag, angle, true);

        // normalize to avoid uchar overflow
        static const float magnitudeScaling = 1.f / sqrt(2);
        mag *= magnitudeScaling;

        // convert to uchar
        cv::Mat saturatedMag(grey.rows, grey.cols, CV_8UC1), shrMag;
        for (int y = 0; y < grey.rows; ++y)
        {
            float* rm = mag.ptr<float>(y);
            uchar* mg = saturatedMag.ptr<uchar>(y);
            for (int x = 0; x < grey.cols; ++x)
            {
                mg[x] =  cv::saturate_cast<uchar>(rm[x]);
            }
        }

        // srink and integrate
        decimate(saturatedMag, shrMag);
        cv::integral(shrMag, mag, cv::noArray(), CV_32S);

        // create hog channels
        angle /= 60.f;

        std::vector<cv::Mat> hist;
        for (int bin = 0; bin < HOG_BINS; ++bin)
        {
            hist.push_back(cv::Mat::zeros(saturatedMag.rows, saturatedMag.cols, CV_8UC1));
        }

        for (int y = 0; y < saturatedMag.rows; ++y)
        {
            uchar* magnitude = saturatedMag.ptr<uchar>(y);
            float* ang = angle.ptr<float>(y);

            for (int x = 0; x < saturatedMag.cols; ++x)
            {
                hist[ (int)ang[x] ].ptr<uchar>(y)[x] = magnitude[x];
            }
        }

        for(int i = 0; i < HOG_BINS; ++i)
        {
            cv::Mat shrunk, sum;
            decimate(hist[i], shrunk);
            cv::integral(shrunk, sum, cv::noArray(), CV_32S);
            hog.push_back(sum);
        }

        hog.push_back(mag);
        hog.insert(hog.end(), luvs.begin(), luvs.end());
        CV_Assert(hog.size() == 10);
#endif
    }

    float get(const int x, const int y, const int channel, const cv::Rect& area) const
    {
        CV_Assert(channel < HOG_LUV_BINS);
        const cv::Mat m = hog[channel];

        dprintf("feature box %d %d %d %d ", area.x, area.y, area.width, area.height);
        dprintf("get for channel %d\n", channel);
        dprintf("!! %d\n", m.depth());

        dprintf("extract feature for: [%d %d] [%d %d] [%d %d] [%d %d]\n",
            x + area.x, y + area.y,  x + area.width,y + area.y,  x + area.width,y + area.height,
            x + area.x, y + area.height);

        dprintf("at point %d %d with offset %d\n", x, y, 0);

        int a = m.ptr<int>(y + area.y)[x + area.x];
        int b = m.ptr<int>(y + area.y)[x + area.width];
        int c = m.ptr<int>(y + area.height)[x + area.width];
        int d = m.ptr<int>(y + area.height)[x + area.x];

        dprintf("    retruved integral values: %d %d %d %d\n", a, b, c, d);

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
    std::vector<Stage>   stages;
    std::vector<Node>    nodes;
    std::vector<float>   leaves;
    std::vector<Feature> features;

    std::vector<Level> levels;

    typedef std::vector<Octave>::iterator  octIt_t;

    float rescale(const Feature& feature, const float relScale, cv::Rect& scaledRect, const float threshold) const
    {
        float scaling = CascadeIntrinsics::getFor(feature.channel, relScale);
        scaledRect = feature.rect;

        dprintf("feature %d box %d %d %d %d\n", feature.channel, scaledRect.x, scaledRect.y,
        scaledRect.width, scaledRect.height);

        dprintf("rescale: %d %f %f\n",feature.channel, relScale, scaling);

        float farea = (scaledRect.width - scaledRect.x) * (scaledRect.height - scaledRect.y);
        // rescale
        scaledRect.x      = cvRound(relScale * scaledRect.x);
        scaledRect.y      = cvRound(relScale * scaledRect.y);
        scaledRect.width  = cvRound(relScale * scaledRect.width);
        scaledRect.height = cvRound(relScale * scaledRect.height);

        dprintf("feature %d box %d %d %d %d\n", feature.channel, scaledRect.x, scaledRect.y,
        scaledRect.width, scaledRect.height);

        float sarea = (scaledRect.width - scaledRect.x) * (scaledRect.height - scaledRect.y);

        float approx = 1.f;
        if (fabs(farea - 0.f) > FLT_EPSILON && fabs(farea - 0.f) > FLT_EPSILON)
        {
            const float expected_new_area = farea * relScale * relScale;
            approx = expected_new_area / sarea;

            dprintf(" rel areas %f %f\n", expected_new_area, sarea);
        }

        // compensation areas rounding
        float rootThreshold = threshold / approx;
        rootThreshold *= scaling;

        dprintf("approximation %f %f -> %f %f\n", approx, threshold, rootThreshold, scaling);

        return rootThreshold;
    }

    void detectAt(const Level& level, const int dx, const int dy, const ChannelStorage& storage,
                  std::vector<Object>& detections) const
    {
        dprintf("detect at: %d %d\n", dx, dy);

        float detectionScore = 0.f;

        const Octave& octave = *(level.octave);
        int stBegin = octave.index * octave.stages, stEnd = stBegin + octave.stages;

        dprintf("  octave stages: %d to %d index %d %f level %f\n",
            stBegin, stEnd, octave.index, octave.scale, level.origScale);

        int st = stBegin;
        for(; st < stEnd; ++st)
        {

            dprintf("index: %d\n", st);

            const Stage& stage = stages[st];
            {
                int nId = st * 3;

                // work with root node
                const Node& node = nodes[nId];
                const Feature& feature = features[node.feature];
                cv::Rect scaledRect;
                float threshold = rescale(feature, level.relScale, scaledRect, node.threshold);


                float sum = storage.get(dx, dy, feature.channel, scaledRect);

                dprintf("root feature %d %f\n",feature.channel, sum);

                int next = (sum >= threshold)? 2 : 1;

                dprintf("go: %d (%f >= %f)\n\n" ,next, sum, threshold);

                // leaves
                const Node& leaf = nodes[nId + next];
                const Feature& fLeaf = features[leaf.feature];

                threshold = rescale(fLeaf, level.relScale, scaledRect, leaf.threshold);
                sum = storage.get(dx, dy, fLeaf.channel, scaledRect);


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

            if (detectionScore <= stage.threshold) break;
        }

        dprintf("x %d y %d: %d\n", dx, dy, st - stBegin);

        if (st == stEnd)
        {
            std::cout << "  got " << st << std::endl;
            level.markDetection(dx, dy, detectionScore, detections);
        }
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
    void calcLevels(int frameW, int frameH, int scales)
    {
        CV_Assert(scales > 1);
        levels.clear();
        float logFactor = (log(maxScale) - log(minScale)) / (scales -1);

        float scale = minScale;
        for (int sc = 0; sc < scales; ++sc)
        {
            int width  = std::max(0.0f, frameW - (origObjWidth  * scale));
            int height = std::max(0.0f, frameH - (origObjHeight * scale));

            float logScale = log(scale);
            octIt_t fit = fitOctave(logScale);


            Level level(*fit, scale, shrinkage, width, height);

            if (!width || !height)
                break;
            else
                levels.push_back(level);

            if (fabs(scale - maxScale) < FLT_EPSILON) break;
            scale = std::min(maxScale, expf(log(scale) + logFactor));

            std::cout << "level " << sc << " scale "
                      << levels[sc].origScale
                      << " octeve "
                      << levels[sc].octave->scale
                      << " "
                      << levels[sc].relScale
                      << " " << levels[sc].shrScale
                      << " [" << levels[sc].objSize.width
                      << " " << levels[sc].objSize.height << "] ["
            << levels[sc].workRect.width << " " << levels[sc].workRect.height << "]" << std::endl;
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

        origObjWidth = (int)root[SC_ORIG_W];
        CV_Assert(origObjWidth == SoftCascade::ORIG_OBJECT_WIDTH);

        origObjHeight = (int)root[SC_ORIG_H];
        CV_Assert(origObjHeight == SoftCascade::ORIG_OBJECT_HEIGHT);

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
            Octave octave(octIndex, cv::Size(SoftCascade::ORIG_OBJECT_WIDTH, SoftCascade::ORIG_OBJECT_HEIGHT), fns);
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
                stages.push_back(Stage(fns));

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

        //debug print
        // std::cout << "collected " << stages.size() << " stages" << std::endl;
        // for (int i = 0; i < (int)stages.size(); ++i)
        // {
        //     std::cout << "stage " << i << ": " << stages[i].threshold << std::endl;
        // }

        // std::cout << "collected " << nodes.size() << " nodes" << std::endl;
        // for (int i = 0; i < (int)nodes.size(); ++i)
        // {
        //     std::cout << "node " << i << ": " << nodes[i].threshold << " " << nodes[i].feature << std::endl;
        // }

        // std::cout << "collected " << leaves.size() << " leaves" << std::endl;
        // for (int i = 0; i < (int)leaves.size(); ++i)
        // {
        //     std::cout << "leaf " << i << ": " << leaves[i] << std::endl;
        // }
        return true;
    }
};

cv::SoftCascade::SoftCascade() : filds(0) {}

cv::SoftCascade::SoftCascade( const string& filename, const float minScale, const float maxScale) : filds(0)
{
    load(filename, minScale, maxScale);
}
cv::SoftCascade::~SoftCascade()
{
    delete filds;
}

bool cv::SoftCascade::load( const string& filename, const float minScale, const float maxScale)
{
    if (filds)
        delete filds;
    filds = 0;

    cv::FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened()) return false;

    filds = new Filds;
    Filds& flds = *filds;
    if (!flds.fill(fs.getFirstTopLevelNode(), minScale, maxScale)) return false;
    flds.calcLevels(FRAME_WIDTH, FRAME_HEIGHT, TOTAL_SCALES);

    return true;
}

// #define DEBUG_STORE_IMAGES
#define DEBUG_SHOW_RESULT

void cv::SoftCascade::detectMultiScale(const Mat& image, const std::vector<cv::Rect>& /*rois*/,
                                       std::vector<cv::Rect>& objects, const int /*rejectfactor*/)
{
    typedef std::vector<cv::Rect>::const_iterator RIter_t;
    // only color images are supperted
    CV_Assert(image.type() == CV_8UC3);

    // only this window size allowed
    CV_Assert(image.cols == 640 && image.rows == 480);

    objects.clear();

    const Filds& fld = *filds;

    cv::Mat image1;
    cv::cvtColor(image, image1, CV_BGR2RGB);

#if defined DEBUG_STORE_IMAGES
    cv::FileStorage fs("/home/kellan/opencvInputImage.xml", cv::FileStorage::WRITE);
    cv::imwrite("/home/kellan/opencvInputImage.jpg", image1);
    fs << "opencvInputImage" << image1;

    cv::Mat doppia;
    cv::FileStorage fsr("/home/kellan/befireGause.xml", cv::FileStorage::READ);
    fsr["input_gpu_mat"] >> doppia;

    cv::Mat diff;
    cv::absdiff(image1, doppia, diff);

    fs << "absdiff" << diff;
    fs.release();

#endif

    cv::imshow("!!", image1);
    cv::waitKey(0);

    // create integrals
    ChannelStorage storage(image, fld.shrinkage);

    // object candidates
    std::vector<Object> detections;

    typedef std::vector<Level>::const_iterator lIt;
    int total = 0, l = 0;
    for (lIt it = fld.levels.begin(); it != fld.levels.end(); ++it)
    {
        const Level& level = *it;

#if defined WITH_DEBUG_OUT
        std::cout << "================================ " << l++ << std::endl;
#else
    (void)l;
#endif
        // int dx = 79; int dy = 76;
        for (int dy = 0; dy < level.workRect.height; ++dy)
        {
            for (int dx = 0; dx < level.workRect.width; ++dx)
            {
                fld.detectAt(level, dx, dy, storage, detections);
                total++;
            }
        }
    cv::Mat out = image.clone();

#if defined DEBUG_SHOW_RESULT

        printf("TOTAL: %d from %d\n", (int)detections.size(),total) ;

        for(int i = 0; i < (int)detections.size(); ++i)
        {
            cv::rectangle(out, detections[i].rect, cv::Scalar(255, 0, 0, 255), 1);
        }

        cv::imshow("out", out);
        cv::waitKey(0);
#endif

        std::cout << "work rect: " << level.workRect.width << " " << level.workRect.height << std::endl;
        detections.clear();
    }

    // std::swap(detections, objects);
}