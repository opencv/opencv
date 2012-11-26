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
#include <opencv2/highgui/highgui.hpp>

#if !defined (HAVE_CUDA)

cv::gpu::SCascade::SCascade(const double, const double, const int, const int) { throw_nogpu(); }

cv::gpu::SCascade::~SCascade() { throw_nogpu(); }

bool cv::gpu::SCascade::load(const FileNode&) { throw_nogpu(); return false;}

void cv::gpu::SCascade::detect(InputArray, InputArray, OutputArray, Stream&) const { throw_nogpu(); }

void cv::gpu::SCascade::genRoi(InputArray, OutputArray, Stream&) const { throw_nogpu(); }

void cv::gpu::SCascade::read(const FileNode& fn) { Algorithm::read(fn); }

#else

#include <icf.hpp>

cv::gpu::device::icf::Level::Level(int idx, const Octave& oct, const float scale, const int w, const int h)
:  octave(idx), step(oct.stages), relScale(scale / oct.scale)
{
    workRect.x = round(w / (float)oct.shrinkage);
    workRect.y = round(h / (float)oct.shrinkage);

    objSize.x  = cv::saturate_cast<uchar>(oct.size.x * relScale);
    objSize.y  = cv::saturate_cast<uchar>(oct.size.y * relScale);

    // according to R. Benenson, M. Mathias, R. Timofte and L. Van Gool's and Dallal's papers
    if (fabs(relScale - 1.f) < FLT_EPSILON)
        scaling[0] = scaling[1] = 1.f;
    else
    {
        scaling[0] = (relScale < 1.f) ? 0.89f * ::pow(relScale, 1.099f / ::log(2)) : 1.f;
        scaling[1] = relScale * relScale;
    }
}

namespace cv { namespace gpu { namespace device {

namespace icf {
    void fillBins(cv::gpu::PtrStepSzb hogluv, const cv::gpu::PtrStepSzf& nangle,
        const int fw, const int fh, const int bins, cudaStream_t stream);

    void suppress(const PtrStepSzb& objects, PtrStepSzb overlaps, PtrStepSzi ndetections,
        PtrStepSzb suppressed, cudaStream_t stream);
}

namespace imgproc {
    void shfl_integral_gpu_buffered(PtrStepSzb, PtrStepSz<uint4>, PtrStepSz<unsigned int>, int, cudaStream_t);

    template <typename T>
    void resize_gpu(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float fx, float fy,
                    PtrStepSzb dst, int interpolation, cudaStream_t stream);
}

}}}

struct cv::gpu::SCascade::Fields
{
    static Fields* parseCascade(const FileNode &root, const float mins, const float maxs, const int totals)
    {
        static const char *const SC_STAGE_TYPE          = "stageType";
        static const char *const SC_BOOST               = "BOOST";

        static const char *const SC_FEATURE_TYPE        = "featureType";
        static const char *const SC_ICF                 = "ICF";

        // only Ada Boost supported
        std::string stageTypeStr = (string)root[SC_STAGE_TYPE];
        CV_Assert(stageTypeStr == SC_BOOST);

        // only HOG-like integral channel features cupported
        string featureTypeStr = (string)root[SC_FEATURE_TYPE];
        CV_Assert(featureTypeStr == SC_ICF);

        static const char *const SC_ORIG_W              = "width";
        static const char *const SC_ORIG_H              = "height";

        int origWidth  = (int)root[SC_ORIG_W];
        int origHeight = (int)root[SC_ORIG_H];

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

        FileNode fn = root[SC_OCTAVES];
        if (fn.empty()) return false;

        using namespace device::icf;

        std::vector<Octave>  voctaves;
        std::vector<float>   vstages;
        std::vector<Node>    vnodes;
        std::vector<float>   vleaves;

        FileNodeIterator it = fn.begin(), it_end = fn.end();
        int feature_offset = 0;
        ushort octIndex = 0;
        ushort shrinkage = 1;

        for (; it != it_end; ++it)
        {
            FileNode fns = *it;
            float scale = (float)fns[SC_OCT_SCALE];

            bool isUPOctave = scale >= 1;

            ushort nstages = saturate_cast<ushort>((int)fns[SC_OCT_STAGES]);
            ushort2 size;
            size.x = cvRound(origWidth * scale);
            size.y = cvRound(origHeight * scale);
            shrinkage = saturate_cast<ushort>((int)fns[SC_OCT_SHRINKAGE]);

            Octave octave(octIndex, nstages, shrinkage, size, scale);
            CV_Assert(octave.stages > 0);
            voctaves.push_back(octave);

            FileNode ffs = fns[SC_FEATURES];
            if (ffs.empty()) return false;

            FileNodeIterator ftrs = ffs.begin();

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
                        // int feature = (int)(*(inIt +=2)) + feature_offset;
                        inIt +=3;
                        // extract feature, Todo:check it
                        uint th = saturate_cast<uint>((float)(*(inIt++)));
                        cv::FileNode ftn = (*ftrs)[SC_F_RECT];
                        cv::FileNodeIterator r_it = ftn.begin();
                        uchar4 rect;
                        rect.x = saturate_cast<uchar>((int)*(r_it++));
                        rect.y = saturate_cast<uchar>((int)*(r_it++));
                        rect.z = saturate_cast<uchar>((int)*(r_it++));
                        rect.w = saturate_cast<uchar>((int)*(r_it++));

                        if (isUPOctave)
                        {
                            rect.z -= rect.x;
                            rect.w -= rect.y;
                        }

                        uint channel = saturate_cast<uint>((int)(*ftrs)[SC_F_CHANNEL]);
                        vnodes.push_back(Node(rect, channel, th));
                        ++ftrs;
                    }

                    fns = (*ftr)[SC_LEAF];
                    inIt = fns.begin(), inIt_end = fns.end();
                    for (; inIt != inIt_end; ++inIt)
                        vleaves.push_back((float)(*inIt));
                }
            }

            feature_offset += octave.stages * 3;
            ++octIndex;
        }

        cv::Mat hoctaves(1, voctaves.size() * sizeof(Octave), CV_8UC1, (uchar*)&(voctaves[0]));
        CV_Assert(!hoctaves.empty());

        cv::Mat hstages(cv::Mat(vstages).reshape(1,1));
        CV_Assert(!hstages.empty());

        cv::Mat hnodes(1, vnodes.size() * sizeof(Node), CV_8UC1, (uchar*)&(vnodes[0]) );
        CV_Assert(!hnodes.empty());

        cv::Mat hleaves(cv::Mat(vleaves).reshape(1,1));
        CV_Assert(!hleaves.empty());

        Fields* fields = new Fields(mins, maxs, totals, origWidth, origHeight, shrinkage, 0,
            hoctaves, hstages, hnodes, hleaves);
        fields->voctaves = voctaves;
        fields->createLevels(FRAME_HEIGHT, FRAME_WIDTH);

        return fields;
    }

    bool check(float mins,float  maxs, int scales)
    {
        bool updated = (minScale == mins) || (maxScale == maxs) || (totals = scales);

        minScale = mins;
        maxScale = maxScale;
        totals   = scales;

        return updated;
    }

    int createLevels(const int fh, const int fw)
    {
        using namespace device::icf;
        std::vector<Level> vlevels;
        float logFactor = (::log(maxScale) - ::log(minScale)) / (totals -1);

        float scale = minScale;
        int dcs = 0;
        for (int sc = 0; sc < totals; ++sc)
        {
            int width  = ::std::max(0.0f, fw - (origObjWidth  * scale));
            int height = ::std::max(0.0f, fh - (origObjHeight * scale));

            float logScale = ::log(scale);
            int fit = fitOctave(voctaves, logScale);

            Level level(fit, voctaves[fit], scale, width, height);

            if (!width || !height)
                break;
            else
            {
                vlevels.push_back(level);
                if (voctaves[fit].scale < 1) ++dcs;
            }

            if (::fabs(scale - maxScale) < FLT_EPSILON) break;
            scale = ::std::min(maxScale, ::expf(::log(scale) + logFactor));
        }

        cv::Mat hlevels = cv::Mat(1, vlevels.size() * sizeof(Level), CV_8UC1, (uchar*)&(vlevels[0]) );
        CV_Assert(!hlevels.empty());
        levels.upload(hlevels);
        downscales = dcs;
        return dcs;
    }

    bool update(int fh, int fw, int shr)
    {
        if ((fh == luv.rows) && (fw == luv.cols)) return false;

        plane.create(fh * (HOG_LUV_BINS + 1), fw, CV_8UC1);
        fplane.create(fh * HOG_BINS, fw, CV_32FC1);
        luv.create(fh, fw, CV_8UC3);

        shrunk.create(fh / shr * HOG_LUV_BINS, fw / shr, CV_8UC1);
        integralBuffer.create(shrunk.rows, shrunk.cols, CV_32SC1);

        hogluv.create((fh / shr) * HOG_LUV_BINS + 1, fw / shr + 1, CV_32SC1);
        hogluv.setTo(cv::Scalar::all(0));

        overlaps.create(1, 5000, CV_8UC1);
        suppressed.create(1, sizeof(Detection) * 51, CV_8UC1);

        return true;
    }

    Fields( const float mins, const float maxs, const int tts, const int ow, const int oh, const int shr, const int ds,
        cv::Mat hoctaves, cv::Mat hstages, cv::Mat hnodes, cv::Mat hleaves)
    : minScale(mins), maxScale(maxs), totals(tts), origObjWidth(ow), origObjHeight(oh), shrinkage(shr), downscales(ds)
    {
        update(FRAME_HEIGHT, FRAME_WIDTH, shr);
        octaves.upload(hoctaves);
        stages.upload(hstages);
        nodes.upload(hnodes);
        leaves.upload(hleaves);
    }

    void detect(const cv::gpu::GpuMat& roi, cv::gpu::GpuMat& objects, Stream& s) const
    {
        if (s)
            s.enqueueMemSet(objects, 0);
        else
            cudaMemset(objects.data, 0, sizeof(Detection));

        cudaSafeCall( cudaGetLastError());

        device::icf::CascadeInvoker<device::icf::GK107PolicyX4> invoker
        = device::icf::CascadeInvoker<device::icf::GK107PolicyX4>(levels, stages, nodes, leaves);

        cudaStream_t stream = StreamAccessor::getStream(s);
        invoker(roi, hogluv, objects, downscales, stream);
    }

    void preprocess(const cv::gpu::GpuMat& colored, Stream& s)
    {
        if (s)
            s.enqueueMemSet(plane, 0);
        else
            cudaMemset(plane.data, 0, plane.step * plane.rows);

        const int fw = colored.cols;
        const int fh = colored.rows;

        GpuMat gray(plane, cv::Rect(0, fh * Fields::HOG_LUV_BINS, fw, fh));
        cv::gpu::cvtColor(colored, gray, CV_BGR2GRAY, s);
        createHogBins(gray ,s);

        createLuvBins(colored, s);

        integrate(fh, fw, s);
    }

    void suppress(GpuMat& objects, Stream& s)
    {
        GpuMat ndetections = GpuMat(objects, cv::Rect(0, 0, sizeof(Detection), 1));
        ensureSizeIsEnough(objects.rows, objects.cols, CV_8UC1, overlaps);

        if (s)
        {
            s.enqueueMemSet(overlaps, 0);
            s.enqueueMemSet(suppressed, 0);
        }
        else
        {
            overlaps.setTo(0);
            suppressed.setTo(0);
        }

        cudaStream_t stream = StreamAccessor::getStream(s);
        device::icf::suppress(objects, overlaps, ndetections, suppressed, stream);
    }

private:

    typedef std::vector<device::icf::Octave>::const_iterator  octIt_t;
    static int fitOctave(const std::vector<device::icf::Octave>& octs, const float& logFactor)
    {
        float minAbsLog = FLT_MAX;
        int res =  0;
        for (int oct = 0; oct < (int)octs.size(); ++oct)
        {
            const device::icf::Octave& octave =octs[oct];
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

    void createHogBins(const cv::gpu::GpuMat& gray, Stream& s)
    {
        static const int fw = gray.cols;
        static const int fh = gray.rows;

        GpuMat dfdx(fplane, cv::Rect(0,  0, fw, fh));
        GpuMat dfdy(fplane, cv::Rect(0, fh, fw, fh));

        cv::gpu::Sobel(gray, dfdx, CV_32F, 1, 0, sobelBuf, 3, 1, BORDER_DEFAULT, -1, s);
        cv::gpu::Sobel(gray, dfdy, CV_32F, 0, 1, sobelBuf, 3, 1, BORDER_DEFAULT, -1, s);

        GpuMat mag(fplane, cv::Rect(0, 2 * fh, fw, fh));
        GpuMat ang(fplane, cv::Rect(0, 3 * fh, fw, fh));

        cv::gpu::cartToPolar(dfdx, dfdy, mag, ang, true, s);

        // normolize magnitude to uchar interval and angles to 6 bins
        GpuMat nmag(fplane, cv::Rect(0, 4 * fh, fw, fh));
        GpuMat nang(fplane, cv::Rect(0, 5 * fh, fw, fh));

        cv::gpu::multiply(mag, cv::Scalar::all(1.f / (8 *::log(2))), nmag, 1, -1, s);
        cv::gpu::multiply(ang, cv::Scalar::all(1.f / 60.f),     nang, 1, -1, s);

        //create uchar magnitude
        GpuMat cmag(plane, cv::Rect(0, fh * Fields::HOG_BINS, fw, fh));
        if (s)
            s.enqueueConvert(nmag, cmag, CV_8UC1);
        else
            nmag.convertTo(cmag, CV_8UC1);

        cudaStream_t stream = StreamAccessor::getStream(s);
        device::icf::fillBins(plane, nang, fw, fh, Fields::HOG_BINS, stream);
    }

    void createLuvBins(const cv::gpu::GpuMat& colored, Stream& s)
    {
        static const int fw = colored.cols;
        static const int fh = colored.rows;

        cv::gpu::cvtColor(colored, luv, CV_BGR2Luv, s);

        std::vector<GpuMat> splited;
        for(int i = 0; i < Fields::LUV_BINS; ++i)
        {
            splited.push_back(GpuMat(plane, cv::Rect(0, fh * (7 + i), fw, fh)));
        }

        cv::gpu::split(luv, splited, s);
    }

    void integrate(const int fh, const int fw, Stream& s)
    {
        GpuMat channels(plane, cv::Rect(0, 0, fw, fh * Fields::HOG_LUV_BINS));
        cv::gpu::resize(channels, shrunk, cv::Size(), 1.f / shrinkage, 1.f / shrinkage, CV_INTER_AREA, s);

        if (info.majorVersion() < 3)
            cv::gpu::integralBuffered(shrunk, hogluv, integralBuffer, s);
        else
        {
            cudaStream_t stream = StreamAccessor::getStream(s);
            device::imgproc::shfl_integral_gpu_buffered(shrunk, integralBuffer, hogluv, 12, stream);
        }
    }

public:

    // scales range
    float minScale;
    float maxScale;

    int totals;

    int origObjWidth;
    int origObjHeight;

    const int shrinkage;
    int downscales;

    // preallocated buffer 640x480x10 for hogluv + 640x480 got gray
    GpuMat plane;

    // preallocated buffer for floating point operations
    GpuMat fplane;

    // temporial mat for cvtColor
    GpuMat luv;

    // 160x120x10
    GpuMat shrunk;

    // temporial mat for integrall
    GpuMat integralBuffer;

    // 161x121x10
    GpuMat hogluv;

    // used for area overlap computing during
    GpuMat overlaps;

    // used for suppression
    GpuMat suppressed;

    // Cascade from xml
    GpuMat octaves;
    GpuMat stages;
    GpuMat nodes;
    GpuMat leaves;
    GpuMat levels;

    GpuMat sobelBuf;

    GpuMat collected;

    std::vector<device::icf::Octave> voctaves;

    DeviceInfo info;

    enum { BOOST = 0 };
    enum
    {
        FRAME_WIDTH        = 640,
        FRAME_HEIGHT       = 480,
        HOG_BINS           = 6,
        LUV_BINS           = 3,
        HOG_LUV_BINS       = 10
    };
};

cv::gpu::SCascade::SCascade(const double mins, const double maxs, const int sc, const int rjf)
: fields(0),  minScale(mins), maxScale(maxs), scales(sc), rejCriteria(rjf) {}

cv::gpu::SCascade::~SCascade() { delete fields; }

bool cv::gpu::SCascade::load(const FileNode& fn)
{
    if (fields) delete fields;
    fields = Fields::parseCascade(fn, minScale, maxScale, scales);
    return fields != 0;
}

void cv::gpu::SCascade::detect(InputArray image, InputArray _rois, OutputArray _objects, Stream& s) const
{
    CV_Assert(fields);
    const GpuMat colored = image.getGpuMat();

    // only color images are supperted
    CV_Assert(colored.type() == CV_8UC3 || colored.type() == CV_32SC1);

    GpuMat rois = _rois.getGpuMat(), objects = _objects.getGpuMat();
    Fields& flds = *fields;

    if (colored.type() == CV_8UC3)
    {
        if (!flds.update(colored.rows, colored.cols, flds.shrinkage) || flds.check(minScale, maxScale, scales))
            flds.createLevels(colored.rows, colored.cols);
        flds.preprocess(colored, s);
    }
    else
    {
        if (s)
            s.enqueueCopy(colored, flds.hogluv);
        else
            colored.copyTo(flds.hogluv);
    }

    flds.detect(rois, objects, s);

    if (rejCriteria != NO_REJECT)
    {
        GpuMat spr(objects, cv::Rect(0, 0, flds.suppressed.cols, flds.suppressed.rows));
        flds.suppress(objects, s);
        flds.suppressed.copyTo(spr);
    }
}

void cv::gpu::SCascade::genRoi(InputArray _roi, OutputArray _mask, Stream& stream) const
{
    CV_Assert(fields);
    int shr = (*fields).shrinkage;

    const GpuMat roi = _roi.getGpuMat();
    _mask.create( roi.cols / shr, roi.rows / shr, roi.type() );
    GpuMat mask = _mask.getGpuMat();
    cv::gpu::GpuMat tmp;

    cv::gpu::resize(roi, tmp, cv::Size(), 1.f / shr, 1.f / shr, CV_INTER_AREA, stream);
    cv::gpu::transpose(tmp, mask, stream);
}

void cv::gpu::SCascade::read(const FileNode& fn)
{
    Algorithm::read(fn);
}

#endif