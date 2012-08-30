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

#include <vector>


struct cv::SoftCascade::Filds
{
    std::vector<float> octaves;
    // cv::Mat luv;
    // std::vector<cv::Mat> bins;
    // cv::Mat magnitude;
    // double scaleFactor;
    // int windowStep;
};

namespace {

struct Cascade {
    int logOctave;
    float octave;
    cv::Size objSize;
};

struct Level {
    int index;
    float factor;
    float logFactor;
    int width;
    int height;
    float octave;
    cv::Size objSize;

    Level(int i,float f, float lf, int w, int h) : index(i), factor(f), logFactor(lf), width(w), height(h), octave(0.f) {}

    void assign(float o, int detW, int detH)
    {
        octave = o;
        objSize = cv::Size(cv::saturate_cast<int>(detW * o), cv::saturate_cast<int>(detH * o));
    }

    float relScale() {return (factor / octave); }
};
    // compute levels of full pyramid
    void pyrLevels(int frameW, int frameH, int detW, int detH, int scales, float minScale, float maxScale, std::vector<Level> levels)
    {
        CV_Assert(scales > 1);
        levels.clear();
        float logFactor = (log(maxScale) - log(minScale)) / (scales -1);

        float scale = minScale;
        for (int sc = 0; sc < scales; ++sc)
        {
            Level level(sc, scale, log(scale) + logFactor, std::max(0.0f, frameW - (detW * scale)), std::max(0.0f, frameH - (detH * scale)));
            if (!level.width || !level.height)
                break;
            else
                levels.push_back(level);

            if (fabs(scale - maxScale) < FLT_EPSILON) break;
            scale = std::min(maxScale, expf(log(scale) + logFactor));
        }

    }

    // according to R. Benenson, M. Mathias, R. Timofte and L. Van Gool paper
    struct CascadeIntrinsics {
        static const float lambda = 1.099f/ 0.301029996f, a = 0.89f;
        static const float intrinsics[10][4];

        static float getFor(int chennel, int scaling, int ab)
        {
            CV_Assert(chennel < 10 && scaling < 2 && ab < 2);
            return intrinsics[chennel][(scaling << 1) + ab];
        }

    };

    const float CascadeIntrinsics::intrinsics[10][4] =
        {   //da, db, ua, ub
            // hog-like orientation bins
            {a, lambda, 1, 2},
            {a, lambda, 1, 2},
            {a, lambda, 1, 2},
            {a, lambda, 1, 2},
            {a, lambda, 1, 2},
            {a, lambda, 1, 2},
            // gradient magnitude
            {a, lambda / log(2), 1, 2},
            // luv -color chennels
            {1, 2,      1, 2},
            {1, 2,      1, 2},
            {1, 2,      1, 2}
        };
}




cv::SoftCascade::SoftCascade() : filds(0) {}

cv::SoftCascade::SoftCascade( const string& filename )
{
    filds = new Filds;
    load(filename);
}
cv::SoftCascade::~SoftCascade()
{
    delete filds;
}

bool cv::SoftCascade::load( const string& filename )
{
    // temp fixture
    Filds& flds = *filds;
    flds.octaves.push_back(0.5f);
    flds.octaves.push_back(1.0f);
    flds.octaves.push_back(2.0f);
    flds.octaves.push_back(4.0f);
    flds.octaves.push_back(8.0f);

    // scales calculations
    int origObjectW = 64;
    int origObjectH = 128;
    float maxScale = 5.f, minScale = 0.4f;
    std::vector<Level> levels;

    pyrLevels(FRAME_WIDTH, FRAME_HEIGHT, origObjectW, origObjectH, TOTAL_SCALES, minScale, maxScale,levels);

    for (std::vector<Level>::iterator level = levels.begin(); level < levels.end(); ++level)
    {
        float minAbsLog = FLT_MAX;
        for (std::vector<float>::iterator oct = flds.octaves.begin(); oct < flds.octaves.end(); ++oct)
        {
            float logOctave = log(*oct);
            float logAbsScale = fabs((*level).logFactor - logOctave);

            if(logAbsScale < minAbsLog)
                (*level).assign(*oct, origObjectW, origObjectH);

        }
    }

    return true;
}

void cv::SoftCascade::detectMultiScale(const Mat& image, const std::vector<cv::Rect>& rois, std::vector<cv::Rect>& objects,
                                           const double factor, const int step, const int rejectfactor)
{}

void cv::SoftCascade::detectForOctave(const int octave)
{}