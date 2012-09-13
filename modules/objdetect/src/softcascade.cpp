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
#include <iostream>

namespace {

    struct Octave
    {
        float scale;
        int stages;
        cv::Size size;
        int shrinkage;
        static const char *const SC_OCT_SCALE;
        static const char *const SC_OCT_STAGES;
        static const char *const SC_OCT_SHRINKAGE;

        Octave(){}
        Octave(cv::Size origObjSize, const cv::FileNode& fn)
        : scale((float)fn[SC_OCT_SCALE]), stages((int)fn[SC_OCT_STAGES]),
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
        Stage(const cv::FileNode& fn) : threshold((float)fn[SC_STAGE_THRESHOLD])
        { std::cout << "   stage: " << threshold << std::endl; }
    };

    const char *const Stage::SC_STAGE_THRESHOLD  = "stageThreshold";

    struct Node
    {
        int feature;
        float threshold;

        Node(){}
        Node(cv::FileNodeIterator& fIt) : feature((int)(*(fIt +=2)++)), threshold((float)(*(fIt++)))
        { std::cout << "   Node: " << feature << " " << threshold << std::endl; }
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
            std::cout << "feature: " << rect.x << " " << rect.y << " " << rect.width << " " << rect.height << " " << channel << std::endl;
        }

//         Feature rescale(float relScale)
//         {
//             Feature res(*this);
//             res.rect = cv::Rect (cvRound(rect.x * relScale), cvRound(rect.y * relScale),
//                                  cvRound(rect.width * relScale), cvRound(rect.height * relScale));
//             res.threshold = threshold * CascadeIntrinsics::getFor(channel, relScale);
//             return res;
//         }
    };

        const char * const Feature::SC_F_CHANNEL   = "channel";
        const char * const Feature::SC_F_RECT      = "rect";
//     // according to R. Benenson, M. Mathias, R. Timofte and L. Van Gool paper
//     struct CascadeIntrinsics
//     {
//         static const float lambda = 1.099f, a = 0.89f;
//         static const float intrinsics[10][4];

//         static float getFor(int channel, float scaling)
//         {
//             CV_Assert(channel < 10);

//             if ((scaling - 1.f) < FLT_EPSILON)
//                 return 1.f;

//             int ud = (int)(scaling < 1.f);
//             return intrinsics[channel][(ud << 1)] * pow(scaling, intrinsics[channel][(ud << 1) + 1]);
//         }

//     };

//     const float CascadeIntrinsics::intrinsics[10][4] =
//         {   //da, db, ua, ub
//             // hog-like orientation bins
//             {a, lambda / log(2), 1, 2},
//             {a, lambda / log(2), 1, 2},
//             {a, lambda / log(2), 1, 2},
//             {a, lambda / log(2), 1, 2},
//             {a, lambda / log(2), 1, 2},
//             {a, lambda / log(2), 1, 2},
//             // gradient magnitude
//             {a, lambda / log(2), 1, 2},
//             // luv color channels
//             {1, 2,      1, 2},
//             {1, 2,      1, 2},
//             {1, 2,      1, 2}
//         };

    struct Level
    {
//         int index;

//         float factor;
//         float logFactor;

//         int width;
//         int height;

//         Octave octave;

//         cv::Size objSize;
//         cv::Size dWinSize;

//         static const float shrinkage = 0.25;

//         Level(int i,float f, float lf, int w, int h): index(i), factor(f), logFactor(lf), width(w), height(h), octave(Octave())
//         {}

//         void assign(const Octave& o, int detW, int detH)
//         {
//             octave = o;
//             objSize = cv::Size(cv::saturate_cast<int>(detW * o.scale), cv::saturate_cast<int>(detH * o.scale));
//         }

//         float relScale() {return (factor / octave.scale); }
//         float srScale()  {return (factor / octave.scale * shrinkage); }
    };

//     struct Integral
//     {
//         cv::Mat magnitude;
//         std::vector<cv::Mat> hist;
//         cv::Mat luv;

//         Integral(cv::Mat m, std::vector<cv::Mat> h, cv::Mat l) : magnitude(m), hist(h), luv(l) {}
//     };
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
    std::vector<Node> nodes;
    std::vector<float> leaves;

    std::vector<Feature> features;

    std::vector<Level> levels;

    // typedef std::vector<Stage>::iterator stIter_t;

    //     // carrently roi must be save for out of ranges.
    // void detectInRoi(const cv::Rect& roi, const Integral& ints, std::vector<cv::Rect>& objects, const int step)
    // {
    //     for (int dy = roi.y; dy < roi.height; dy+=step)
    //         for (int dx = roi.x; dx < roi.width; dx += step)
    //         {
    //             applyCascade(ints, dx, dy);
    //         }
    // }

    // void applyCascade(const Integral& ints, const int x, const int y)
    // {
    //     for (stIter_t sIt = stages.begin(); sIt != stages.end(); ++sIt)
    //     {
    //         Stage& stage = *sIt;
    //     }
    // }

    // compute levels of full pyramid
    void calcLevels(int frameW, int frameH, int scales)
    {
        CV_Assert(scales > 1);
        levels.clear();
    //     float logFactor = (log(maxScale) - log(minScale)) / (scales -1);

    //     float scale = minScale;
    //     for (int sc = 0; sc < scales; ++sc)
    //     {
    //         Level level(sc, scale, log(scale), std::max(0.0f, frameW - (origObjWidth * scale)), std::max(0.0f, frameH - (origObjHeight * scale)));
    //         if (!level.width || !level.height)
    //             break;
    //         else
    //             levels.push_back(level);

    //         if (fabs(scale - maxScale) < FLT_EPSILON) break;
    //         scale = std::min(maxScale, expf(log(scale) + logFactor));
    //     }

    //     for (std::vector<Level>::iterator level = levels.begin(); level < levels.end(); ++level)
    //     {
    //         float minAbsLog = FLT_MAX;
    //         for (std::vector<Octave>::iterator oct = octaves.begin(); oct < octaves.end(); ++oct)
    //         {
    //             const Octave& octave =*oct;
    //             float logOctave = log(octave.scale);
    //             float logAbsScale = fabs((*level).logFactor - logOctave);


    //             if(logAbsScale < minAbsLog)
    //             {
    //                 printf("######### %f %f %f %f\n", octave.scale, logOctave, logAbsScale, (*level).logFactor);
    //                 minAbsLog = logAbsScale;
    //                 (*level).assign(octave, ORIG_OBJECT_WIDTH, ORIG_OBJECT_HEIGHT);
    //             }
    //         }
    //     }
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


        // only boost supported
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
        for (; it != it_end; ++it)
        {
            FileNode fns = *it;
            Octave octave(cv::Size(SoftCascade::ORIG_OBJECT_WIDTH, SoftCascade::ORIG_OBJECT_HEIGHT), fns);
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
                        nodes.push_back(Node(inIt));

                    fns = (*ftr)[SC_LEAF];
                    inIt = fns.begin(), inIt_end = fns.end();
                    for (; inIt != inIt_end; ++inIt)
                        leaves.push_back((float)(*inIt));
                }
            }

            st = ffs.begin(), st_end = ffs.end();
            for (; st != st_end; ++st )
                features.push_back(Feature(*st));
        }

        shrinkage = octaves[0].shrinkage;
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

namespace {

    void calcHistBins(const cv::Mat& grey, cv::Mat& magIntegral, std::vector<cv::Mat>& histInts, const int bins, int shrinkage)
    {
        CV_Assert( grey.type() == CV_8U);

        float scale = 1.f / shrinkage;

        const int rows = grey.rows + 1;
        const int cols = grey.cols + 1;
        cv::Size intSumSize(cols, rows);

        histInts.clear();
        std::vector<cv::Mat> hist;
        for (int bin = 0; bin < bins; ++bin)
        {
            hist.push_back(cv::Mat(rows, cols, CV_32FC1));
        }

        cv::Mat df_dx, df_dy, mag, angle;
        cv::Sobel(grey, df_dx, CV_32F, 1, 0);
        cv::Sobel(grey, df_dy, CV_32F, 0, 1);

        cv::cartToPolar(df_dx, df_dy, mag, angle, true);

        const float magnitudeScaling = 1.0 / sqrt(2);
        mag *= magnitudeScaling;
        angle /= 60;

        for (int h = 0; h < mag.rows; ++h)
        {
            float* magnitude = mag.ptr<float>(h);
            float* ang = angle.ptr<float>(h);

            for (int w = 0; w < mag.cols; ++w)
            {
                hist[(int)ang[w]].ptr<float>(h)[w] = magnitude[w];
            }
        }

        for (int bin = 0; bin < bins; ++bin)
        {
            cv::Mat shrunk, sum;
            cv::resize(hist[bin], shrunk, cv::Size(), scale, scale, cv::INTER_AREA);
            cv::integral(shrunk, sum);
            histInts.push_back(sum);
        }

        cv::Mat shrMag;
        cv::resize(mag, shrMag, cv::Size(), scale, scale, cv::INTER_AREA);

        cv::integral(shrMag, magIntegral, mag.depth());
    }

    struct ChannelStorage
    {
        std::vector<cv::Mat> hog;
        cv::Mat luv;
        cv::Mat magnitude;

        int shrinkage;

        enum {HOG_BINS = 6};

        ChannelStorage() {}
        ChannelStorage(const cv::Mat& colored, int shr) : shrinkage(shr)
        {
            cv::Mat _luv;
            cv::cvtColor(colored, _luv, CV_BGR2Luv);

            cv::integral(luv, luv);

            cv::Mat grey;
            cv::cvtColor(colored, grey, CV_RGB2GRAY);

            calcHistBins(grey, magnitude, hog, HOG_BINS, shrinkage);
            std::cout << magnitude.cols << " " << magnitude.rows << std::endl;
            cv::imshow("1", magnitude);
            cv::waitKey(0);
        }
    };
}



void cv::SoftCascade::detectMultiScale(const Mat& image, const std::vector<cv::Rect>& rois, std::vector<cv::Rect>& objects,
                                       const int step, const int rejectfactor)// add step scaling
{
    typedef std::vector<cv::Rect>::const_iterator RIter_t;
    // only color images are supperted
    CV_Assert(image.type() == CV_8UC3);

    // only this window size allowed
    CV_Assert(image.cols == 640 && image.rows == 480);

    objects.clear();

    const Filds& fld = *filds;

    // create integrals
    ChannelStorage storage(image, fld.shrinkage);

    // for (RIter_t it = rois.begin(); it != rois.end(); ++it)
    // {
    //     const cv::Rect& roi = *it;
    //     (*filds).detectInRoi(roi, integrals, objects, step);
    // }

}