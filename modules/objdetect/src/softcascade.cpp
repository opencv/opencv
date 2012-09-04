/*M///////////////////////////////////////////////////////////////////////////////////////
//
// IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
//
//                          License Agreement
//               For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2008-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
//  * The name of the copyright holders may not be used to endorse or promote products
//    derived from this software without specific prior written permission.
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
//M*/

#include <precomp.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core.hpp>

#include <vector>
#include <string>
#include <stdio.h>

namespace {

    static const char* SC_OCT_SCALE        = "scale";
    static const char* SC_OCT_STAGES       = "stageNum";

    struct Octave
    {
        float scale;
        int stages;

        Octave(){}
        Octave(const cv::FileNode& fn) : scale((float)fn[SC_OCT_SCALE]), stages((int)fn[SC_OCT_STAGES])
        {printf("octave: %f %d\n", scale, stages);}
    };

    static const char *SC_STAGE_THRESHOLD  = "stageThreshold";
    static const char *SC_STAGE_WEIGHT     = "weight";

    struct Stage
    {
        float threshold;
        float weight;

        Stage(){}
        Stage(const cv::FileNode& fn) : threshold((float)fn[SC_STAGE_THRESHOLD]), weight((float)fn[SC_STAGE_WEIGHT])
        {printf("   stage: %f %f\n",threshold, weight);}
    };

    // according to R. Benenson, M. Mathias, R. Timofte and L. Van Gool paper
    struct CascadeIntrinsics
    {
        static const float lambda = 1.099f, a = 0.89f;
        static const float intrinsics[10][4];

        static float getFor(int channel, float scaling)
        {
            CV_Assert(channel < 10);

            if ((scaling - 1.f) < FLT_EPSILON)
                return 1.f;

            int ud = (int)(scaling < 1.f);
            return intrinsics[channel][(ud << 1)] * pow(scaling, intrinsics[channel][(ud << 1) + 1]);
        }

    };

    const float CascadeIntrinsics::intrinsics[10][4] =
        {   //da, db, ua, ub
            // hog-like orientation bins
            {a, lambda / log(2), 1, 2},
            {a, lambda / log(2), 1, 2},
            {a, lambda / log(2), 1, 2},
            {a, lambda / log(2), 1, 2},
            {a, lambda / log(2), 1, 2},
            {a, lambda / log(2), 1, 2},
            // gradient magnitude
            {a, lambda / log(2), 1, 2},
            // luv color channels
            {1, 2,      1, 2},
            {1, 2,      1, 2},
            {1, 2,      1, 2}
        };

    static const char *SC_F_THRESHOLD      = "threshold";
    static const char *SC_F_DIRECTION      = "direction";
    static const char *SC_F_CHANNEL        = "channel";
    static const char *SC_F_RECT           = "rect";

    struct Feature
    {
        float threshold;
        int direction;
        int channel;
        cv::Rect rect;

        Feature() {}
        Feature(const cv::FileNode& fn)
        : threshold((float)fn[SC_F_THRESHOLD]), direction((int)fn[SC_F_DIRECTION]),
          channel((int)fn[SC_F_CHANNEL])
        {
            cv::FileNode rn = fn[SC_F_RECT];
            cv::FileNodeIterator r_it = rn.begin();
            rect = cv::Rect(*(r_it++), *(r_it++), *(r_it++), *(r_it++));
            printf("       feature: %f %d %d [%d %d %d %d]\n",threshold, direction, channel, rect.x, rect.y, rect.width, rect.height);}

        Feature rescale(float relScale)
        {
            Feature res(*this);
            res.rect = cv::Rect (cvRound(rect.x * relScale), cvRound(rect.y * relScale),
                                 cvRound(rect.width * relScale), cvRound(rect.height * relScale));
            res.threshold = threshold * CascadeIntrinsics::getFor(channel, relScale);
            return res;
        }
    };

    struct Level
    {
        int index;
        float factor;
        float logFactor;
        int width;
        int height;
        float octave;
        cv::Size objSize;

        Level(int i,float f, float lf, int w, int h): index(i), factor(f), logFactor(lf), width(w), height(h), octave(0.f) {}

        void assign(float o, int detW, int detH)
        {
            octave = o;
            objSize = cv::Size(cv::saturate_cast<int>(detW * o), cv::saturate_cast<int>(detH * o));
        }

        float relScale() {return (factor / octave); }
    };
}

struct cv::SoftCascade::Filds
{
    float minScale;
    float maxScale;

    int origObjWidth;
    int origObjHeight;

    int noctaves;

    std::vector<Octave>  octaves;
    std::vector<Stage>   stages;
    std::vector<Feature> features;
    std::vector<Level>   levels;

    // compute levels of full pyramid
    void calcLevels(int frameW, int frameH, int scales)
    {
        CV_Assert(scales > 1);
        levels.clear();
        float logFactor = (log(maxScale) - log(minScale)) / (scales -1);

        float scale = minScale;
        for (int sc = 0; sc < scales; ++sc)
        {
            Level level(sc, scale, log(scale) + logFactor,
                std::max(0.0f, frameW - (origObjWidth * scale)), std::max(0.0f, frameH - (origObjHeight * scale)));
            if (!level.width || !level.height)
                break;
            else
                levels.push_back(level);

            if (fabs(scale - maxScale) < FLT_EPSILON) break;
            scale = std::min(maxScale, expf(log(scale) + logFactor));
        }

        for (std::vector<Level>::iterator level = levels.begin(); level < levels.end(); ++level)
        {
            float minAbsLog = FLT_MAX;
            for (std::vector<Octave>::iterator oct = octaves.begin(); oct < octaves.end(); ++oct)
            {
                const Octave& octave =*oct;
                float logOctave = log(octave.scale);
                float logAbsScale = fabs((*level).logFactor - logOctave);

                if(logAbsScale < minAbsLog)
                    (*level).assign(octave.scale, ORIG_OBJECT_WIDTH, ORIG_OBJECT_HEIGHT);
            }
        }
    }

    bool fill(const FileNode &root, const float mins, const float maxs)
    {
        minScale = mins;
        maxScale = maxs;

        // cascade properties
        const char *SC_STAGE_TYPE       = "stageType";
        const char *SC_BOOST            = "BOOST";
        const char *SC_FEATURE_TYPE     = "featureType";
        const char *SC_ICF              = "ICF";
        const char *SC_TREE_TYPE        = "stageTreeType";
        const char *SC_STAGE_TH2        = "TH2";
        const char *SC_NUM_OCTAVES      = "octavesNum";
        const char *SC_ORIG_W           = "origObjWidth";
        const char *SC_ORIG_H           = "origObjHeight";

        const char* SC_OCTAVES          = "octaves";
        const char *SC_STAGES           = "stages";
        const char *SC_FEATURES         = "features";


        // only boost supported
        std::string stageTypeStr = (string)root[SC_STAGE_TYPE];
        CV_Assert(stageTypeStr == SC_BOOST);

        // only HOG-like integral channel features cupported
        string featureTypeStr = (string)root[SC_FEATURE_TYPE];
        CV_Assert(featureTypeStr == SC_ICF);

        // only trees of height 2
        string stageTreeTypeStr = (string)root[SC_TREE_TYPE];
        CV_Assert(stageTreeTypeStr == SC_STAGE_TH2);

        // not empty
        noctaves = (int)root[SC_NUM_OCTAVES];
        CV_Assert(noctaves > 0);

        origObjWidth = (int)root[SC_ORIG_W];
        CV_Assert(origObjWidth == SoftCascade::ORIG_OBJECT_WIDTH);

        origObjHeight = (int)root[SC_ORIG_H];
        CV_Assert(origObjHeight == SoftCascade::ORIG_OBJECT_HEIGHT);

        // for each octave (~ one cascade in classic OpenCV xml)
        FileNode fn = root[SC_OCTAVES];
        if (fn.empty()) return false;

        octaves.reserve(noctaves);
        FileNodeIterator it = fn.begin(), it_end = fn.end();
        for (; it != it_end; ++it)
        {
            FileNode fns = *it;
            Octave octave = Octave(fns);
            CV_Assert(octave.stages > 0);
            octaves.push_back(octave);
            stages.reserve(stages.size() + octave.stages);

            fns = fns[SC_STAGES];
            if (fn.empty()) return false;

            // for each stage (~ decision tree with H = 2)
            FileNodeIterator st = fns.begin(), st_end = fns.end();
            for (; st != st_end; ++st )
            {
                fns = *st;
                stages.push_back(Stage(fns));

                fns = fns[SC_FEATURES];
                // for each feature for tree. features stored in order {root, left, right}
                FileNodeIterator ftr = fns.begin(), ft_end = fns.end();
                for (; ftr != ft_end; ++ftr)
                {
                    features.push_back(Feature(*ftr));
                }
            }
        }
        return true;
    }
};

cv::SoftCascade::SoftCascade() : filds(0) {}

cv::SoftCascade::SoftCascade( const string& filename, const float minScale, const float maxScale)
{
    filds = new Filds;
    load(filename, minScale, maxScale);
}
cv::SoftCascade::~SoftCascade()
{
    delete filds;
}

bool cv::SoftCascade::load( const string& filename, const float minScale, const float maxScale)
{
    delete filds;
    filds = 0;

    cv::FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened()) return false;

    filds = new Filds;
    Filds& flds = *filds;
    if (!flds.fill(fs.getFirstTopLevelNode(), minScale, maxScale)) return false;
    // flds.calcLevels(FRAME_WIDTH, FRAME_HEIGHT, TOTAL_SCALES);

    return true;
}

void cv::SoftCascade::detectMultiScale(const Mat& image, const std::vector<cv::Rect>& rois, std::vector<cv::Rect>& objects,
                                           const double factor, const int step, const int rejectfactor)
{}

void cv::SoftCascade::detectForOctave(const int octave)
{}