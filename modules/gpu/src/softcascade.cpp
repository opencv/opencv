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

#if !defined (HAVE_CUDA)

cv::gpu::SoftCascade::SoftCascade() : filds(0) { throw_nogpu(); }

cv::gpu::SoftCascade::SoftCascade( const string&, const float, const float) : filds(0) { throw_nogpu(); }

cv::gpu::SoftCascade::~SoftCascade() { throw_nogpu(); }

bool cv::gpu::SoftCascade::load( const string&, const float, const float) { throw_nogpu(); }

void cv::gpu::SoftCascade::detectMultiScale(const GpuMat&, const GpuMat&, GpuMat&, const int, Stream) { throw_nogpu(); }

#else

#include <icf.hpp>

struct cv::gpu::SoftCascade::Filds
{
    // scales range
    float minScale;
    float maxScale;

    int origObjWidth;
    int origObjHeight;

    GpuMat octaves;
    GpuMat stages;
    GpuMat nodes;
    GpuMat leaves;
    GpuMat features;
    GpuMat levels;

    // preallocated buffer 640x480x10 + 640x480
    GpuMat dmem;
    // 160x120x10
    GpuMat shrunk;
    // 161x121x10
    GpuMat hogluv;

    std::vector<float> scales;

    icf::Cascade cascade;
    icf::ChannelStorage storage;

    enum { BOOST = 0 };
    enum
    {
        FRAME_WIDTH        = 640,
        FRAME_HEIGHT       = 480,
        TOTAL_SCALES       = 55,
        CLASSIFIERS        = 5,
        ORIG_OBJECT_WIDTH  = 64,
        ORIG_OBJECT_HEIGHT = 128,
        HOG_BINS           = 6,
        HOG_LUV_BINS       = 10
    };

    bool fill(const FileNode &root, const float mins, const float maxs);
    void detect(cudaStream_t stream) const
    {
        cascade.detect(hogluv, stream);
    }

private:
    void calcLevels(const std::vector<icf::Octave>& octs,
                                                    int frameW, int frameH, int nscales);

    typedef std::vector<icf::Octave>::const_iterator  octIt_t;
    int fitOctave(const std::vector<icf::Octave>& octs, const float& logFactor) const
    {
        float minAbsLog = FLT_MAX;
        int res =  0;
        for (int oct = 0; oct < (int)octs.size(); ++oct)
        {
            const icf::Octave& octave =octs[oct];
            float logOctave = ::log(octave.scale);
            float logAbsScale = ::fabs(logFactor - logOctave);

            if(logAbsScale < minAbsLog)
            {
                res = oct;
                minAbsLog = logAbsScale;
            }
        }
        return res;
    }
};

inline bool cv::gpu::SoftCascade::Filds::fill(const FileNode &root, const float mins, const float maxs)
{
    minScale = mins;
    maxScale = maxs;

    // cascade properties
    static const char *const SC_STAGE_TYPE          = "stageType";
    static const char *const SC_BOOST               = "BOOST";

    static const char *const SC_FEATURE_TYPE        = "featureType";
    static const char *const SC_ICF                 = "ICF";

    static const char *const SC_ORIG_W              = "width";
    static const char *const SC_ORIG_H              = "height";

    static const char *const SC_OCTAVES             = "octaves";
    static const char *const SC_STAGES              = "stages";
    static const char *const SC_FEATURES            = "features";

    static const char *const SC_WEEK                = "weakClassifiers";
    static const char *const SC_INTERNAL            = "internalNodes";
    static const char *const SC_LEAF                = "leafValues";

    static const char *const SC_OCT_SCALE           = "scale";
    static const char *const SC_OCT_STAGES          = "stageNum";
    static const char *const SC_OCT_SHRINKAGE       = "shrinkingFactor";

    static const char *const SC_STAGE_THRESHOLD     = "stageThreshold";

    static const char * const SC_F_CHANNEL          = "channel";
    static const char * const SC_F_RECT             = "rect";

    // only Ada Boost supported
    std::string stageTypeStr = (string)root[SC_STAGE_TYPE];
    CV_Assert(stageTypeStr == SC_BOOST);

    // only HOG-like integral channel features cupported
    string featureTypeStr = (string)root[SC_FEATURE_TYPE];
    CV_Assert(featureTypeStr == SC_ICF);

    origObjWidth = (int)root[SC_ORIG_W];
    CV_Assert(origObjWidth  == ORIG_OBJECT_WIDTH);

    origObjHeight = (int)root[SC_ORIG_H];
    CV_Assert(origObjHeight == ORIG_OBJECT_HEIGHT);

    FileNode fn = root[SC_OCTAVES];
        if (fn.empty()) return false;

    std::vector<icf::Octave>  voctaves;
    std::vector<float>        vstages;
    std::vector<icf::Node>    vnodes;
    std::vector<float>        vleaves;
    std::vector<icf::Feature> vfeatures;
    scales.clear();

    // std::vector<Level> levels;

    FileNodeIterator it = fn.begin(), it_end = fn.end();
    int feature_offset = 0;
    ushort octIndex = 0;
    ushort shrinkage = 1;

    for (; it != it_end; ++it)
    {
        FileNode fns = *it;
        float scale = (float)fns[SC_OCT_SCALE];
        scales.push_back(scale);
        ushort nstages = saturate_cast<ushort>((int)fns[SC_OCT_STAGES]);
        ushort2 size;
        size.x = cvRound(ORIG_OBJECT_WIDTH * scale);
        size.y = cvRound(ORIG_OBJECT_HEIGHT * scale);
        shrinkage = saturate_cast<ushort>((int)fns[SC_OCT_SHRINKAGE]);

        icf::Octave octave(octIndex, nstages, shrinkage, size, scale);
        CV_Assert(octave.stages > 0);
        voctaves.push_back(octave);

        FileNode ffs = fns[SC_FEATURES];
        if (ffs.empty()) return false;

        fns = fns[SC_STAGES];
        if (fn.empty()) return false;

        // for each stage (~ decision tree with H = 2)
        FileNodeIterator st = fns.begin(), st_end = fns.end();
        for (; st != st_end; ++st )
        {
            fns = *st;
            vstages.push_back((float)fns[SC_STAGE_THRESHOLD]);

            fns = fns[SC_WEEK];
            FileNodeIterator ftr = fns.begin(), ft_end = fns.end();
            for (; ftr != ft_end; ++ftr)
            {
                fns = (*ftr)[SC_INTERNAL];
                FileNodeIterator inIt = fns.begin(), inIt_end = fns.end();
                for (; inIt != inIt_end;)
                {
                    int feature = (int)(*(inIt +=2)++) + feature_offset;
                    float th = (float)(*(inIt++));
                    vnodes.push_back(icf::Node(feature, th));
                }

                fns = (*ftr)[SC_LEAF];
                inIt = fns.begin(), inIt_end = fns.end();
                for (; inIt != inIt_end; ++inIt)
                    vleaves.push_back((float)(*inIt));
            }
        }

        st = ffs.begin(), st_end = ffs.end();
        for (; st != st_end; ++st )
        {
            cv::FileNode rn = (*st)[SC_F_RECT];
            cv::FileNodeIterator r_it = rn.begin();
            uchar4 rect;
            rect.x = saturate_cast<uchar>((int)*(r_it++));
            rect.y = saturate_cast<uchar>((int)*(r_it++));
            rect.z = saturate_cast<uchar>((int)*(r_it++));
            rect.w = saturate_cast<uchar>((int)*(r_it++));
            vfeatures.push_back(icf::Feature((int)(*st)[SC_F_CHANNEL], rect));
        }

        feature_offset += octave.stages * 3;
        ++octIndex;
    }

    // upload in gpu memory
    octaves.upload(cv::Mat(1, voctaves.size() * sizeof(icf::Octave), CV_8UC1, (uchar*)&(voctaves[0]) ));
    CV_Assert(!octaves.empty());

    stages.upload(cv::Mat(vstages).reshape(1,1));
    CV_Assert(!stages.empty());

    nodes.upload(cv::Mat(1, vnodes.size() * sizeof(icf::Node), CV_8UC1, (uchar*)&(vnodes[0]) ));
    CV_Assert(!nodes.empty());

    leaves.upload(cv::Mat(vleaves).reshape(1,1));
    CV_Assert(!leaves.empty());

    features.upload(cv::Mat(1, vfeatures.size() * sizeof(icf::Feature), CV_8UC1, (uchar*)&(vfeatures[0]) ));
    CV_Assert(!features.empty());

    // compute levels
    calcLevels(voctaves, FRAME_WIDTH, FRAME_HEIGHT, TOTAL_SCALES);
    CV_Assert(!levels.empty());

    //init Cascade
    cascade = icf::Cascade(octaves, stages, nodes, leaves, features, levels);

    // allocate buffers
    dmem.create(FRAME_HEIGHT * (HOG_LUV_BINS + 1), FRAME_WIDTH, CV_8UC1);
    shrunk.create(FRAME_HEIGHT / shrinkage * HOG_LUV_BINS, FRAME_WIDTH / shrinkage, CV_8UC1);
    hogluv.create( (FRAME_HEIGHT / shrinkage * HOG_LUV_BINS) + 1, (FRAME_WIDTH / shrinkage) + 1, CV_16UC1);

    storage = icf::ChannelStorage(dmem, shrunk, hogluv, shrinkage);
    return true;
}

namespace {
    struct CascadeIntrinsics
    {
        static const float lambda = 1.099f, a = 0.89f;

        static float getFor(int channel, float scaling)
        {
            CV_Assert(channel < 10);

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

            float a = A[(int)(scaling >= 1)][(int)(channel > 6)];
            float b = B[(int)(scaling >= 1)][(int)(channel > 6)];

            // printf("!!! scaling: %f %f %f -> %f\n", scaling, a, b, a * pow(scaling, b));
            return a * pow(scaling, b);
        }
    };
}

inline void cv::gpu::SoftCascade::Filds::calcLevels(const std::vector<icf::Octave>& octs,
                                                    int frameW, int frameH, int nscales)
{
    CV_Assert(nscales > 1);

    std::vector<icf::Level> vlevels;
    float logFactor = (::log(maxScale) - ::log(minScale)) / (nscales -1);

    float scale = minScale;
    for (int sc = 0; sc < nscales; ++sc)
    {
        int width  = ::std::max(0.0f, frameW - (origObjWidth  * scale));
        int height = ::std::max(0.0f, frameH - (origObjHeight * scale));

        float logScale = ::log(scale);
        int fit = fitOctave(octs, logScale);

        icf::Level level(fit, octs[fit], scale, width, height);
        level.scaling[0] = CascadeIntrinsics::getFor(0, level.relScale);
        level.scaling[1] = CascadeIntrinsics::getFor(9, level.relScale);

        if (!width || !height)
            break;
        else
            vlevels.push_back(level);

        if (::fabs(scale - maxScale) < FLT_EPSILON) break;
        scale = ::std::min(maxScale, ::expf(::log(scale) + logFactor));

        // std::cout << "level " << sc
        //           << " octeve "
        //           << vlevels[sc].octave
        //           << " relScale "
        //           << vlevels[sc].relScale
        //           << " " << vlevels[sc].shrScale
        //           << " [" << (int)vlevels[sc].objSize.x
        //           << " " <<  (int)vlevels[sc].objSize.y << "] ["
        // <<  (int)vlevels[sc].workRect.x << " " <<  (int)vlevels[sc].workRect.y << "]" << std::endl;
    }
    levels.upload(cv::Mat(1, vlevels.size() * sizeof(icf::Level), CV_8UC1, (uchar*)&(vlevels[0]) ));
}

cv::gpu::SoftCascade::SoftCascade() : filds(0) {}

cv::gpu::SoftCascade::SoftCascade( const string& filename, const float minScale, const float maxScale) : filds(0)
{
    load(filename, minScale, maxScale);
}

cv::gpu::SoftCascade::~SoftCascade()
{
    delete filds;
}

bool cv::gpu::SoftCascade::load( const string& filename, const float minScale, const float maxScale)
{
    if (filds)
        delete filds;
    filds = 0;

    cv::FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened()) return false;

    filds = new Filds;
    Filds& flds = *filds;
    if (!flds.fill(fs.getFirstTopLevelNode(), minScale, maxScale)) return false;
    return true;
}

void cv::gpu::SoftCascade::detectMultiScale(const GpuMat& image, const GpuMat& /*rois*/,
                                GpuMat& /*objects*/, const int /*rejectfactor*/, Stream s)
{
    // only color images are supperted
    CV_Assert(image.type() == CV_8UC3);

    // only this window size allowed
    CV_Assert(image.cols == 640 && image.rows == 480);

    Filds& flds = *filds;

    cudaStream_t stream = StreamAccessor::getStream(s);

    flds.storage.frame(image, stream);
    flds.detect(stream);
}

#endif