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

cv::gpu::SoftCascade::SoftCascade() : filds(0) { throw_nogpu(); }
cv::gpu::SoftCascade::SoftCascade( const string&, const float, const float) : filds(0) { throw_nogpu(); }
cv::gpu::SoftCascade::~SoftCascade() { throw_nogpu(); }
bool cv::gpu::SoftCascade::load( const string&, const float, const float) { throw_nogpu(); return false; }
void cv::gpu::SoftCascade::detectMultiScale(const GpuMat&, const GpuMat&, GpuMat&, const int, int) const
{
    throw_nogpu();
}

void cv::gpu::SoftCascade::detectMultiScale(const GpuMat&, const GpuMat&, GpuMat&, int, GpuMat&, Stream) const
{
    throw_nogpu();
}

cv::Size cv::gpu::SoftCascade::getRoiSize() const { throw_nogpu(); return cv::Size();}

#else

#include <icf.hpp>

cv::gpu::device::icf::Level::Level(int idx, const Octave& oct, const float scale, const int w, const int h)
:  octave(idx), relScale(scale / oct.scale), shrScale (relScale / (float)oct.shrinkage)
{
    workRect.x = round(w / (float)oct.shrinkage);
    workRect.y = round(h / (float)oct.shrinkage);

    objSize.x  = cv::saturate_cast<uchar>(oct.size.x * relScale);
    objSize.y  = cv::saturate_cast<uchar>(oct.size.y * relScale);
}

namespace cv { namespace gpu { namespace device {
namespace icf {
    void fillBins(cv::gpu::PtrStepSzb hogluv, const cv::gpu::PtrStepSzf& nangle,
        const int fw, const int fh, const int bins);
}
namespace imgproc
{
    void shfl_integral_gpu(PtrStepSzb img, PtrStepSz<unsigned int> integral, cudaStream_t stream);
}
}}}

struct cv::gpu::SoftCascade::Filds
{
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
                { 1.099f / ::log(2), 2.f}, // down
                {             0.f, 2.f}  // up
            };

            float a = A[(int)(scaling >= 1)][(int)(channel > 6)];
            float b = B[(int)(scaling >= 1)][(int)(channel > 6)];

            // printf("!!! scaling: %f %f %f -> %f\n", scaling, a, b, a * pow(scaling, b));
            return a * ::pow(scaling, b);
        }
    };

    static Filds* parseCascade(const FileNode &root, const float mins, const float maxs)
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

        int origWidth = (int)root[SC_ORIG_W];
        CV_Assert(origWidth  == ORIG_OBJECT_WIDTH);

        int origHeight = (int)root[SC_ORIG_H];
        CV_Assert(origHeight == ORIG_OBJECT_HEIGHT);

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
            size.x = cvRound(ORIG_OBJECT_WIDTH * scale);
            size.y = cvRound(ORIG_OBJECT_HEIGHT * scale);
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

        std::vector<Level> vlevels;
        float logFactor = (::log(maxs) - ::log(mins)) / (TOTAL_SCALES -1);

        float scale = mins;
        int downscales = 0;
        for (int sc = 0; sc < TOTAL_SCALES; ++sc)
        {
            int width  = ::std::max(0.0f, FRAME_WIDTH - (origWidth  * scale));
            int height = ::std::max(0.0f, FRAME_HEIGHT - (origHeight * scale));

            float logScale = ::log(scale);
            int fit = fitOctave(voctaves, logScale);

            Level level(fit, voctaves[fit], scale, width, height);
            level.scaling[0] = CascadeIntrinsics::getFor(0, level.relScale);
            level.scaling[1] = CascadeIntrinsics::getFor(9, level.relScale);

            if (!width || !height)
                break;
            else
            {
                vlevels.push_back(level);
                if (voctaves[fit].scale < 1) ++downscales;
            }

            if (::fabs(scale - maxs) < FLT_EPSILON) break;
            scale = ::std::min(maxs, ::expf(::log(scale) + logFactor));

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

        cv::Mat hlevels(1, vlevels.size() * sizeof(Level), CV_8UC1, (uchar*)&(vlevels[0]) );
        CV_Assert(!hlevels.empty());

        Filds* filds = new Filds(mins, maxs, origWidth, origHeight, shrinkage, downscales,
            hoctaves, hstages, hnodes, hleaves, hlevels);

        return filds;
    }

    Filds( const float mins, const float maxs, const int ow, const int oh, const int shr, const int ds,
        cv::Mat hoctaves, cv::Mat hstages, cv::Mat hnodes, cv::Mat hleaves, cv::Mat hlevels)
    : minScale(mins), maxScale(maxs), origObjWidth(ow), origObjHeight(oh), shrinkage(shr), downscales(ds)
    {
        plane.create(FRAME_HEIGHT * (HOG_LUV_BINS + 1), FRAME_WIDTH, CV_8UC1);
        fplane.create(FRAME_HEIGHT * 6, FRAME_WIDTH, CV_32FC1);
        luv.create(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3);
        shrunk.create(FRAME_HEIGHT / shr * HOG_LUV_BINS, FRAME_WIDTH / shr, CV_8UC1);
        integralBuffer.create(1 , (shrunk.rows + 1) * HOG_LUV_BINS * (shrunk.cols + 1), CV_32SC1);
        hogluv.create((FRAME_HEIGHT / shr + 1) * HOG_LUV_BINS, FRAME_WIDTH / shr + 64, CV_32SC1);
        detCounter.create(1,1, CV_32SC1);

        octaves.upload(hoctaves);
        stages.upload(hstages);
        nodes.upload(hnodes);
        leaves.upload(hleaves);
        levels.upload(hlevels);

        invoker = device::icf::CascadeInvoker<device::icf::CascadePolicy>(levels, octaves, stages, nodes, leaves);

    }

    void detect(int scale, const cv::gpu::GpuMat& roi, cv::gpu::GpuMat& objects, cudaStream_t stream) const
    {
        cudaMemset(detCounter.data, 0, detCounter.step * detCounter.rows * sizeof(int));
        invoker(roi, hogluv, objects, detCounter, downscales, scale);
    }

    void preprocess(const cv::gpu::GpuMat& colored)
    {
        cudaMemset(plane.data, 0, plane.step * plane.rows);

        static const int fw = Filds::FRAME_WIDTH;
        static const int fh = Filds::FRAME_HEIGHT;

        GpuMat gray(plane, cv::Rect(0, fh * Filds::HOG_LUV_BINS, fw, fh));
        cv::gpu::cvtColor(colored, gray, CV_BGR2GRAY);
        createHogBins(gray);

        createLuvBins(colored);

        integrate();
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

    void createHogBins(const cv::gpu::GpuMat& gray)
    {
        static const int fw = Filds::FRAME_WIDTH;
        static const int fh = Filds::FRAME_HEIGHT;

        GpuMat dfdx(fplane, cv::Rect(0,  0, fw, fh));
        GpuMat dfdy(fplane, cv::Rect(0, fh, fw, fh));

        cv::gpu::Sobel(gray, dfdx, CV_32F, 1, 0, 3, 0.125f);
        cv::gpu::Sobel(gray, dfdy, CV_32F, 0, 1, 3, 0.125f);

        GpuMat mag(fplane, cv::Rect(0, 2 * fh, fw, fh));
        GpuMat ang(fplane, cv::Rect(0, 3 * fh, fw, fh));

        cv::gpu::cartToPolar(dfdx, dfdy, mag, ang, true);

        // normolize magnitude to uchar interval and angles to 6 bins

        GpuMat nmag(fplane, cv::Rect(0, 4 * fh, fw, fh));
        GpuMat nang(fplane, cv::Rect(0, 5 * fh, fw, fh));

        cv::gpu::multiply(mag, cv::Scalar::all(1.f / ::log(2)), nmag);
        cv::gpu::multiply(ang, cv::Scalar::all(1.f / 60.f),     nang);

        //create uchar magnitude
        GpuMat cmag(plane, cv::Rect(0, fh * Filds::HOG_BINS, fw, fh));
        nmag.convertTo(cmag, CV_8UC1);

        device::icf::fillBins(plane, nang, fw, fh, Filds::HOG_BINS);
    }

    void createLuvBins(const cv::gpu::GpuMat& colored)
    {
        static const int fw = Filds::FRAME_WIDTH;
        static const int fh = Filds::FRAME_HEIGHT;

        cv::gpu::cvtColor(colored, luv, CV_BGR2Luv);

        std::vector<GpuMat> splited;
        for(int i = 0; i < Filds::LUV_BINS; ++i)
        {
            splited.push_back(GpuMat(plane, cv::Rect(0, fh * (7 + i), fw, fh)));
        }

        cv::gpu::split(luv, splited);
    }

    void integrate()
    {
        int fw = Filds::FRAME_WIDTH;
        int fh = Filds::FRAME_HEIGHT;

        GpuMat channels(plane, cv::Rect(0, 0, fw, fh * Filds::HOG_LUV_BINS));
        cv::gpu::resize(channels, shrunk, cv::Size(), 0.25, 0.25, CV_INTER_AREA);

        fw /= shrinkage;
        fh /= shrinkage;

        for(int i = 0; i < Filds::HOG_LUV_BINS; ++i)
        {
            GpuMat channel(shrunk, cv::Rect(0, fh  * i, fw, fh ));
            GpuMat sum(hogluv, cv::Rect(0, (fh + 1) * i, fw + 1, fh + 1));
            cv::gpu::integralBuffered(channel, sum, integralBuffer);
        }
    }

public:

    // scales range
    float minScale;
    float maxScale;

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

    GpuMat detCounter;

    // Cascade from xml
    GpuMat octaves;
    GpuMat stages;
    GpuMat nodes;
    GpuMat leaves;
    GpuMat levels;

    device::icf::CascadeInvoker<device::icf::CascadePolicy> invoker;

    enum { BOOST = 0 };
    enum
    {
        FRAME_WIDTH        = 640,
        FRAME_HEIGHT       = 480,
        TOTAL_SCALES       = 55,
        ORIG_OBJECT_WIDTH  = 64,
        ORIG_OBJECT_HEIGHT = 128,
        HOG_BINS           = 6,
        LUV_BINS           = 3,
        HOG_LUV_BINS       = 10
    };
};

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
    if (filds) delete filds;

    cv::FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened()) return false;

    filds = Filds::parseCascade(fs.getFirstTopLevelNode(), minScale, maxScale);
    return filds != 0;
}

void cv::gpu::SoftCascade::detectMultiScale(const GpuMat& colored, const GpuMat& rois,
                                GpuMat& objects, const int /*rejectfactor*/, int specificScale) const
{
    // only color images are supperted
    CV_Assert(colored.type() == CV_8UC3);

    // we guess user knows about shrincage
    CV_Assert((rois.size().width == getRoiSize().height) && (rois.type() == CV_8UC1));

    // only this window size allowed
    CV_Assert(colored.cols == Filds::FRAME_WIDTH && colored.rows == Filds::FRAME_HEIGHT);

    Filds& flds = *filds;

    flds.preprocess(colored);

    flds.detect(specificScale, rois, objects, 0);

    cv::Mat out(flds.detCounter);
    int ndetections = *(out.data);

    if (! ndetections)
        objects = GpuMat();
    else
        objects = GpuMat(objects, cv::Rect(0, 0, ndetections * sizeof(Detection), 1));
}

void cv::gpu::SoftCascade::detectMultiScale(const GpuMat&, const GpuMat&, GpuMat&, int, GpuMat&, Stream) const
{
    // cudaStream_t stream = StreamAccessor::getStream(s);
}

cv::Size cv::gpu::SoftCascade::getRoiSize() const
{
    return cv::Size(Filds::FRAME_WIDTH / (*filds).shrinkage, Filds::FRAME_HEIGHT / (*filds).shrinkage);
}

#endif