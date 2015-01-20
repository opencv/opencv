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
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#include "precomp.hpp"
#include "opencv2/objdetect/objdetect_c.h"

using namespace cv;
using namespace cv::cuda;

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

Ptr<cuda::CascadeClassifier> cv::cuda::CascadeClassifier::create(const String&) { throw_no_cuda(); return Ptr<cuda::CascadeClassifier>(); }
Ptr<cuda::CascadeClassifier> cv::cuda::CascadeClassifier::create(const FileStorage&) { throw_no_cuda(); return Ptr<cuda::CascadeClassifier>(); }

#else

//
// CascadeClassifierBase
//

namespace
{
    class CascadeClassifierBase : public cuda::CascadeClassifier
    {
    public:
        CascadeClassifierBase();

        virtual void setMaxObjectSize(Size maxObjectSize) { maxObjectSize_ = maxObjectSize; }
        virtual Size getMaxObjectSize() const { return maxObjectSize_; }

        virtual void setMinObjectSize(Size minSize) { minObjectSize_ = minSize; }
        virtual Size getMinObjectSize() const { return minObjectSize_; }

        virtual void setScaleFactor(double scaleFactor) { scaleFactor_ = scaleFactor; }
        virtual double getScaleFactor() const { return scaleFactor_; }

        virtual void setMinNeighbors(int minNeighbors) { minNeighbors_ = minNeighbors; }
        virtual int getMinNeighbors() const { return minNeighbors_; }

        virtual void setFindLargestObject(bool findLargestObject) { findLargestObject_ = findLargestObject; }
        virtual bool getFindLargestObject() { return findLargestObject_; }

        virtual void setMaxNumObjects(int maxNumObjects) { maxNumObjects_ = maxNumObjects; }
        virtual int getMaxNumObjects() const { return maxNumObjects_; }

    protected:
        Size maxObjectSize_;
        Size minObjectSize_;
        double scaleFactor_;
        int minNeighbors_;
        bool findLargestObject_;
        int maxNumObjects_;
    };

    CascadeClassifierBase::CascadeClassifierBase() :
        maxObjectSize_(),
        minObjectSize_(),
        scaleFactor_(1.2),
        minNeighbors_(4),
        findLargestObject_(false),
        maxNumObjects_(100)
    {
    }
}

//
// HaarCascade
//

#ifdef HAVE_OPENCV_CUDALEGACY

namespace
{
    class HaarCascade_Impl : public CascadeClassifierBase
    {
    public:
        explicit HaarCascade_Impl(const String& filename);

        virtual Size getClassifierSize() const;

        virtual void detectMultiScale(InputArray image,
                                      OutputArray objects,
                                      Stream& stream);

        virtual void convert(OutputArray gpu_objects,
                             std::vector<Rect>& objects);

    private:
        NCVStatus load(const String& classifierFile);
        NCVStatus calculateMemReqsAndAllocate(const Size& frameSize);
        NCVStatus process(const GpuMat& src, GpuMat& objects, cv::Size ncvMinSize, /*out*/ unsigned int& numDetections);

        Size lastAllocatedFrameSize;

        Ptr<NCVMemStackAllocator> gpuAllocator;
        Ptr<NCVMemStackAllocator> cpuAllocator;

        cudaDeviceProp devProp;
        NCVStatus ncvStat;

        Ptr<NCVMemNativeAllocator> gpuCascadeAllocator;
        Ptr<NCVMemNativeAllocator> cpuCascadeAllocator;

        Ptr<NCVVectorAlloc<HaarStage64> >           h_haarStages;
        Ptr<NCVVectorAlloc<HaarClassifierNode128> > h_haarNodes;
        Ptr<NCVVectorAlloc<HaarFeature64> >         h_haarFeatures;

        HaarClassifierCascadeDescriptor haar;

        Ptr<NCVVectorAlloc<HaarStage64> >           d_haarStages;
        Ptr<NCVVectorAlloc<HaarClassifierNode128> > d_haarNodes;
        Ptr<NCVVectorAlloc<HaarFeature64> >         d_haarFeatures;
    };

    static void NCVDebugOutputHandler(const String &msg)
    {
        CV_Error(Error::GpuApiCallError, msg.c_str());
    }

    HaarCascade_Impl::HaarCascade_Impl(const String& filename) :
        lastAllocatedFrameSize(-1, -1)
    {
        ncvSetDebugOutputHandler(NCVDebugOutputHandler);
        ncvSafeCall( load(filename) );
    }

    Size HaarCascade_Impl::getClassifierSize() const
    {
        return Size(haar.ClassifierSize.width, haar.ClassifierSize.height);
    }

    void HaarCascade_Impl::detectMultiScale(InputArray _image,
                                            OutputArray _objects,
                                            Stream& stream)
    {
        const GpuMat image = _image.getGpuMat();

        CV_Assert( image.depth() == CV_8U);
        CV_Assert( scaleFactor_ > 1 );
        CV_Assert( !stream );

        Size ncvMinSize = getClassifierSize();
        if (ncvMinSize.width < minObjectSize_.width && ncvMinSize.height < minObjectSize_.height)
        {
            ncvMinSize.width = minObjectSize_.width;
            ncvMinSize.height = minObjectSize_.height;
        }

        BufferPool pool(stream);
        GpuMat objectsBuf = pool.getBuffer(1, maxNumObjects_, DataType<Rect>::type);

        unsigned int numDetections;
        ncvSafeCall( process(image, objectsBuf, ncvMinSize, numDetections) );

        if (numDetections > 0)
        {
            objectsBuf.colRange(0, numDetections).copyTo(_objects);
        }
        else
        {
            _objects.release();
        }
    }

    void HaarCascade_Impl::convert(OutputArray _gpu_objects, std::vector<Rect>& objects)
    {
        if (_gpu_objects.empty())
        {
            objects.clear();
            return;
        }

        Mat gpu_objects;
        if (_gpu_objects.kind() == _InputArray::CUDA_GPU_MAT)
        {
            _gpu_objects.getGpuMat().download(gpu_objects);
        }
        else
        {
            gpu_objects = _gpu_objects.getMat();
        }

        CV_Assert( gpu_objects.rows == 1 );
        CV_Assert( gpu_objects.type() == DataType<Rect>::type );

        Rect* ptr = gpu_objects.ptr<Rect>();
        objects.assign(ptr, ptr + gpu_objects.cols);
    }

    NCVStatus HaarCascade_Impl::load(const String& classifierFile)
    {
        int devId = cv::cuda::getDevice();
        ncvAssertCUDAReturn(cudaGetDeviceProperties(&devProp, devId), NCV_CUDA_ERROR);

        // Load the classifier from file (assuming its size is about 1 mb) using a simple allocator
        gpuCascadeAllocator = makePtr<NCVMemNativeAllocator>(NCVMemoryTypeDevice, static_cast<int>(devProp.textureAlignment));
        cpuCascadeAllocator = makePtr<NCVMemNativeAllocator>(NCVMemoryTypeHostPinned, static_cast<int>(devProp.textureAlignment));

        ncvAssertPrintReturn(gpuCascadeAllocator->isInitialized(), "Error creating cascade GPU allocator", NCV_CUDA_ERROR);
        ncvAssertPrintReturn(cpuCascadeAllocator->isInitialized(), "Error creating cascade CPU allocator", NCV_CUDA_ERROR);

        Ncv32u haarNumStages, haarNumNodes, haarNumFeatures;
        ncvStat = ncvHaarGetClassifierSize(classifierFile, haarNumStages, haarNumNodes, haarNumFeatures);
        ncvAssertPrintReturn(ncvStat == NCV_SUCCESS, "Error reading classifier size (check the file)", NCV_FILE_ERROR);

        h_haarStages.reset  (new NCVVectorAlloc<HaarStage64>(*cpuCascadeAllocator, haarNumStages));
        h_haarNodes.reset   (new NCVVectorAlloc<HaarClassifierNode128>(*cpuCascadeAllocator, haarNumNodes));
        h_haarFeatures.reset(new NCVVectorAlloc<HaarFeature64>(*cpuCascadeAllocator, haarNumFeatures));

        ncvAssertPrintReturn(h_haarStages->isMemAllocated(), "Error in cascade CPU allocator", NCV_CUDA_ERROR);
        ncvAssertPrintReturn(h_haarNodes->isMemAllocated(), "Error in cascade CPU allocator", NCV_CUDA_ERROR);
        ncvAssertPrintReturn(h_haarFeatures->isMemAllocated(), "Error in cascade CPU allocator", NCV_CUDA_ERROR);

        ncvStat = ncvHaarLoadFromFile_host(classifierFile, haar, *h_haarStages, *h_haarNodes, *h_haarFeatures);
        ncvAssertPrintReturn(ncvStat == NCV_SUCCESS, "Error loading classifier", NCV_FILE_ERROR);

        d_haarStages.reset  (new NCVVectorAlloc<HaarStage64>(*gpuCascadeAllocator, haarNumStages));
        d_haarNodes.reset   (new NCVVectorAlloc<HaarClassifierNode128>(*gpuCascadeAllocator, haarNumNodes));
        d_haarFeatures.reset(new NCVVectorAlloc<HaarFeature64>(*gpuCascadeAllocator, haarNumFeatures));

        ncvAssertPrintReturn(d_haarStages->isMemAllocated(), "Error in cascade GPU allocator", NCV_CUDA_ERROR);
        ncvAssertPrintReturn(d_haarNodes->isMemAllocated(), "Error in cascade GPU allocator", NCV_CUDA_ERROR);
        ncvAssertPrintReturn(d_haarFeatures->isMemAllocated(), "Error in cascade GPU allocator", NCV_CUDA_ERROR);

        ncvStat = h_haarStages->copySolid(*d_haarStages, 0);
        ncvAssertPrintReturn(ncvStat == NCV_SUCCESS, "Error copying cascade to GPU", NCV_CUDA_ERROR);
        ncvStat = h_haarNodes->copySolid(*d_haarNodes, 0);
        ncvAssertPrintReturn(ncvStat == NCV_SUCCESS, "Error copying cascade to GPU", NCV_CUDA_ERROR);
        ncvStat = h_haarFeatures->copySolid(*d_haarFeatures, 0);
        ncvAssertPrintReturn(ncvStat == NCV_SUCCESS, "Error copying cascade to GPU", NCV_CUDA_ERROR);

        return NCV_SUCCESS;
    }

    NCVStatus HaarCascade_Impl::calculateMemReqsAndAllocate(const Size& frameSize)
    {
        if (lastAllocatedFrameSize == frameSize)
        {
            return NCV_SUCCESS;
        }

        // Calculate memory requirements and create real allocators
        NCVMemStackAllocator gpuCounter(static_cast<int>(devProp.textureAlignment));
        NCVMemStackAllocator cpuCounter(static_cast<int>(devProp.textureAlignment));

        ncvAssertPrintReturn(gpuCounter.isInitialized(), "Error creating GPU memory counter", NCV_CUDA_ERROR);
        ncvAssertPrintReturn(cpuCounter.isInitialized(), "Error creating CPU memory counter", NCV_CUDA_ERROR);

        NCVMatrixAlloc<Ncv8u> d_src(gpuCounter, frameSize.width, frameSize.height);
        NCVMatrixAlloc<Ncv8u> h_src(cpuCounter, frameSize.width, frameSize.height);

        ncvAssertReturn(d_src.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);
        ncvAssertReturn(h_src.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);

        NCVVectorAlloc<NcvRect32u> d_rects(gpuCounter, 100);
        ncvAssertReturn(d_rects.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);

        NcvSize32u roi;
        roi.width = d_src.width();
        roi.height = d_src.height();
        Ncv32u numDetections;
        ncvStat = ncvDetectObjectsMultiScale_device(d_src, roi, d_rects, numDetections, haar, *h_haarStages,
            *d_haarStages, *d_haarNodes, *d_haarFeatures, haar.ClassifierSize, 4, 1.2f, 1, 0, gpuCounter, cpuCounter, devProp, 0);

        ncvAssertReturnNcvStat(ncvStat);
        ncvAssertCUDAReturn(cudaStreamSynchronize(0), NCV_CUDA_ERROR);

        gpuAllocator = makePtr<NCVMemStackAllocator>(NCVMemoryTypeDevice, gpuCounter.maxSize(), static_cast<int>(devProp.textureAlignment));
        cpuAllocator = makePtr<NCVMemStackAllocator>(NCVMemoryTypeHostPinned, cpuCounter.maxSize(), static_cast<int>(devProp.textureAlignment));

        ncvAssertPrintReturn(gpuAllocator->isInitialized(), "Error creating GPU memory allocator", NCV_CUDA_ERROR);
        ncvAssertPrintReturn(cpuAllocator->isInitialized(), "Error creating CPU memory allocator", NCV_CUDA_ERROR);

        lastAllocatedFrameSize = frameSize;
        return NCV_SUCCESS;
    }

    NCVStatus HaarCascade_Impl::process(const GpuMat& src, GpuMat& objects, cv::Size ncvMinSize, /*out*/ unsigned int& numDetections)
    {
        calculateMemReqsAndAllocate(src.size());

        NCVMemPtr src_beg;
        src_beg.ptr = (void*)src.ptr<Ncv8u>();
        src_beg.memtype = NCVMemoryTypeDevice;

        NCVMemSegment src_seg;
        src_seg.begin = src_beg;
        src_seg.size  = src.step * src.rows;

        NCVMatrixReuse<Ncv8u> d_src(src_seg, static_cast<int>(devProp.textureAlignment), src.cols, src.rows, static_cast<int>(src.step), true);
        ncvAssertReturn(d_src.isMemReused(), NCV_ALLOCATOR_BAD_REUSE);

        CV_Assert(objects.rows == 1);

        NCVMemPtr objects_beg;
        objects_beg.ptr = (void*)objects.ptr<NcvRect32u>();
        objects_beg.memtype = NCVMemoryTypeDevice;

        NCVMemSegment objects_seg;
        objects_seg.begin = objects_beg;
        objects_seg.size = objects.step * objects.rows;
        NCVVectorReuse<NcvRect32u> d_rects(objects_seg, objects.cols);
        ncvAssertReturn(d_rects.isMemReused(), NCV_ALLOCATOR_BAD_REUSE);

        NcvSize32u roi;
        roi.width = d_src.width();
        roi.height = d_src.height();

        NcvSize32u winMinSize(ncvMinSize.width, ncvMinSize.height);

        Ncv32u flags = 0;
        flags |= findLargestObject_ ? NCVPipeObjDet_FindLargestObject : 0;

        ncvStat = ncvDetectObjectsMultiScale_device(
            d_src, roi, d_rects, numDetections, haar, *h_haarStages,
            *d_haarStages, *d_haarNodes, *d_haarFeatures,
            winMinSize,
            minNeighbors_,
            scaleFactor_, 1,
            flags,
            *gpuAllocator, *cpuAllocator, devProp, 0);
        ncvAssertReturnNcvStat(ncvStat);
        ncvAssertCUDAReturn(cudaStreamSynchronize(0), NCV_CUDA_ERROR);

        return NCV_SUCCESS;
    }
}

#endif

//
// LbpCascade
//

namespace cv { namespace cuda { namespace device
{
    namespace lbp
    {
        void classifyPyramid(int frameW,
                             int frameH,
                             int windowW,
                             int windowH,
                             float initalScale,
                             float factor,
                             int total,
                             const PtrStepSzb& mstages,
                             const int nstages,
                             const PtrStepSzi& mnodes,
                             const PtrStepSzf& mleaves,
                             const PtrStepSzi& msubsets,
                             const PtrStepSzb& mfeatures,
                             const int subsetSize,
                             PtrStepSz<int4> objects,
                             unsigned int* classified,
                             PtrStepSzi integral);

        void connectedConmonents(PtrStepSz<int4> candidates,
                                 int ncandidates,
                                 PtrStepSz<int4> objects,
                                 int groupThreshold,
                                 float grouping_eps,
                                 unsigned int* nclasses);
    }
}}}

namespace
{
    cv::Size operator -(const cv::Size& a, const cv::Size& b)
    {
        return cv::Size(a.width - b.width, a.height - b.height);
    }

    cv::Size operator +(const cv::Size& a, const int& i)
    {
        return cv::Size(a.width + i, a.height + i);
    }

    cv::Size operator *(const cv::Size& a, const float& f)
    {
        return cv::Size(cvRound(a.width * f), cvRound(a.height * f));
    }

    cv::Size operator /(const cv::Size& a, const float& f)
    {
        return cv::Size(cvRound(a.width / f), cvRound(a.height / f));
    }

    bool operator <=(const cv::Size& a, const cv::Size& b)
    {
        return a.width <= b.width && a.height <= b.width;
    }

    struct PyrLavel
    {
        PyrLavel(int _order, float _scale, cv::Size frame, cv::Size window, cv::Size minObjectSize)
        {
            do
            {
                order = _order;
                scale = pow(_scale, order);
                sFrame = frame / scale;
                workArea = sFrame - window + 1;
                sWindow = window * scale;
                _order++;
            } while (sWindow <= minObjectSize);
        }

        bool isFeasible(cv::Size maxObj)
        {
            return workArea.width > 0 && workArea.height > 0 && sWindow <= maxObj;
        }

        PyrLavel next(float factor, cv::Size frame, cv::Size window, cv::Size minObjectSize)
        {
            return PyrLavel(order + 1, factor, frame, window, minObjectSize);
        }

        int order;
        float scale;
        cv::Size sFrame;
        cv::Size workArea;
        cv::Size sWindow;
    };

    class LbpCascade_Impl : public CascadeClassifierBase
    {
    public:
        explicit LbpCascade_Impl(const FileStorage& file);

        virtual Size getClassifierSize() const { return NxM; }

        virtual void detectMultiScale(InputArray image,
                                      OutputArray objects,
                                      Stream& stream);

        virtual void convert(OutputArray gpu_objects,
                             std::vector<Rect>& objects);

    private:
        bool load(const FileNode &root);
        void allocateBuffers(cv::Size frame);

    private:
        struct Stage
        {
            int    first;
            int    ntrees;
            float  threshold;
        };

        enum stage { BOOST = 0 };
        enum feature { LBP = 1, HAAR = 2 };

        static const stage stageType = BOOST;
        static const feature featureType = LBP;

        cv::Size NxM;
        bool isStumps;
        int ncategories;
        int subsetSize;
        int nodeStep;

        // gpu representation of classifier
        GpuMat stage_mat;
        GpuMat trees_mat;
        GpuMat nodes_mat;
        GpuMat leaves_mat;
        GpuMat subsets_mat;
        GpuMat features_mat;

        GpuMat integral;
        GpuMat integralBuffer;
        GpuMat resuzeBuffer;

        GpuMat candidates;
        static const int integralFactor = 4;
    };

    LbpCascade_Impl::LbpCascade_Impl(const FileStorage& file)
    {
        load(file.getFirstTopLevelNode());
    }

    void LbpCascade_Impl::detectMultiScale(InputArray _image,
                                           OutputArray _objects,
                                           Stream& stream)
    {
        const GpuMat image = _image.getGpuMat();

        CV_Assert( image.depth() == CV_8U);
        CV_Assert( scaleFactor_ > 1 );
        CV_Assert( !stream );

        const float grouping_eps = 0.2f;

        BufferPool pool(stream);
        GpuMat objects = pool.getBuffer(1, maxNumObjects_, DataType<Rect>::type);

        // used for debug
        // candidates.setTo(cv::Scalar::all(0));
        // objects.setTo(cv::Scalar::all(0));

        if (maxObjectSize_ == cv::Size())
            maxObjectSize_ = image.size();

        allocateBuffers(image.size());

        unsigned int classified = 0;
        GpuMat dclassified(1, 1, CV_32S);
        cudaSafeCall( cudaMemcpy(dclassified.ptr(), &classified, sizeof(int), cudaMemcpyHostToDevice) );

        PyrLavel level(0, scaleFactor_, image.size(), NxM, minObjectSize_);

        while (level.isFeasible(maxObjectSize_))
        {
            int acc = level.sFrame.width + 1;
            float iniScale = level.scale;

            cv::Size area = level.workArea;
            int step = 1 + (level.scale <= 2.f);

            int total = 0, prev  = 0;

            while (acc <= integralFactor * (image.cols + 1) && level.isFeasible(maxObjectSize_))
            {
                // create sutable matrix headers
                GpuMat src  = resuzeBuffer(cv::Rect(0, 0, level.sFrame.width, level.sFrame.height));
                GpuMat sint = integral(cv::Rect(prev, 0, level.sFrame.width + 1, level.sFrame.height + 1));

                // generate integral for scale
                cuda::resize(image, src, level.sFrame, 0, 0, cv::INTER_LINEAR);
                cuda::integral(src, sint);

                // calculate job
                int totalWidth = level.workArea.width / step;
                total += totalWidth * (level.workArea.height / step);

                // go to next pyramide level
                level = level.next(scaleFactor_, image.size(), NxM, minObjectSize_);
                area = level.workArea;

                step = (1 + (level.scale <= 2.f));
                prev = acc;
                acc += level.sFrame.width + 1;
            }

            device::lbp::classifyPyramid(image.cols, image.rows, NxM.width - 1, NxM.height - 1, iniScale, scaleFactor_, total, stage_mat, stage_mat.cols / sizeof(Stage), nodes_mat,
                leaves_mat, subsets_mat, features_mat, subsetSize, candidates, dclassified.ptr<unsigned int>(), integral);
        }

        if (minNeighbors_ <= 0  || objects.empty())
            return;

        cudaSafeCall( cudaMemcpy(&classified, dclassified.ptr(), sizeof(int), cudaMemcpyDeviceToHost) );
        device::lbp::connectedConmonents(candidates, classified, objects, minNeighbors_, grouping_eps, dclassified.ptr<unsigned int>());

        cudaSafeCall( cudaMemcpy(&classified, dclassified.ptr(), sizeof(int), cudaMemcpyDeviceToHost) );
        cudaSafeCall( cudaDeviceSynchronize() );

        if (classified > 0)
        {
            objects.colRange(0, classified).copyTo(_objects);
        }
        else
        {
            _objects.release();
        }
    }

    void LbpCascade_Impl::convert(OutputArray _gpu_objects, std::vector<Rect>& objects)
    {
        if (_gpu_objects.empty())
        {
            objects.clear();
            return;
        }

        Mat gpu_objects;
        if (_gpu_objects.kind() == _InputArray::CUDA_GPU_MAT)
        {
            _gpu_objects.getGpuMat().download(gpu_objects);
        }
        else
        {
            gpu_objects = _gpu_objects.getMat();
        }

        CV_Assert( gpu_objects.rows == 1 );
        CV_Assert( gpu_objects.type() == DataType<Rect>::type );

        Rect* ptr = gpu_objects.ptr<Rect>();
        objects.assign(ptr, ptr + gpu_objects.cols);
    }

    bool LbpCascade_Impl::load(const FileNode &root)
    {
        const char *CUDA_CC_STAGE_TYPE       = "stageType";
        const char *CUDA_CC_FEATURE_TYPE     = "featureType";
        const char *CUDA_CC_BOOST            = "BOOST";
        const char *CUDA_CC_LBP              = "LBP";
        const char *CUDA_CC_MAX_CAT_COUNT    = "maxCatCount";
        const char *CUDA_CC_HEIGHT           = "height";
        const char *CUDA_CC_WIDTH            = "width";
        const char *CUDA_CC_STAGE_PARAMS     = "stageParams";
        const char *CUDA_CC_MAX_DEPTH        = "maxDepth";
        const char *CUDA_CC_FEATURE_PARAMS   = "featureParams";
        const char *CUDA_CC_STAGES           = "stages";
        const char *CUDA_CC_STAGE_THRESHOLD  = "stageThreshold";
        const float CUDA_THRESHOLD_EPS       = 1e-5f;
        const char *CUDA_CC_WEAK_CLASSIFIERS = "weakClassifiers";
        const char *CUDA_CC_INTERNAL_NODES   = "internalNodes";
        const char *CUDA_CC_LEAF_VALUES      = "leafValues";
        const char *CUDA_CC_FEATURES         = "features";
        const char *CUDA_CC_RECT             = "rect";

        String stageTypeStr = (String)root[CUDA_CC_STAGE_TYPE];
        CV_Assert(stageTypeStr == CUDA_CC_BOOST);

        String featureTypeStr = (String)root[CUDA_CC_FEATURE_TYPE];
        CV_Assert(featureTypeStr == CUDA_CC_LBP);

        NxM.width =  (int)root[CUDA_CC_WIDTH];
        NxM.height = (int)root[CUDA_CC_HEIGHT];
        CV_Assert( NxM.height > 0 && NxM.width > 0 );

        isStumps = ((int)(root[CUDA_CC_STAGE_PARAMS][CUDA_CC_MAX_DEPTH]) == 1) ? true : false;
        CV_Assert(isStumps);

        FileNode fn = root[CUDA_CC_FEATURE_PARAMS];
        if (fn.empty())
            return false;

        ncategories = fn[CUDA_CC_MAX_CAT_COUNT];

        subsetSize = (ncategories + 31) / 32;
        nodeStep = 3 + ( ncategories > 0 ? subsetSize : 1 );

        fn = root[CUDA_CC_STAGES];
        if (fn.empty())
            return false;

        std::vector<Stage> stages;
        stages.reserve(fn.size());

        std::vector<int> cl_trees;
        std::vector<int> cl_nodes;
        std::vector<float> cl_leaves;
        std::vector<int> subsets;

        FileNodeIterator it = fn.begin(), it_end = fn.end();
        for (size_t si = 0; it != it_end; si++, ++it )
        {
            FileNode fns = *it;
            Stage st;
            st.threshold = (float)fns[CUDA_CC_STAGE_THRESHOLD] - CUDA_THRESHOLD_EPS;

            fns = fns[CUDA_CC_WEAK_CLASSIFIERS];
            if (fns.empty())
                return false;

            st.ntrees = (int)fns.size();
            st.first = (int)cl_trees.size();

            stages.push_back(st);// (int, int, float)

            cl_trees.reserve(stages[si].first + stages[si].ntrees);

            // weak trees
            FileNodeIterator it1 = fns.begin(), it1_end = fns.end();
            for ( ; it1 != it1_end; ++it1 )
            {
                FileNode fnw = *it1;

                FileNode internalNodes = fnw[CUDA_CC_INTERNAL_NODES];
                FileNode leafValues = fnw[CUDA_CC_LEAF_VALUES];
                if ( internalNodes.empty() || leafValues.empty() )
                    return false;

                int nodeCount = (int)internalNodes.size()/nodeStep;
                cl_trees.push_back(nodeCount);

                cl_nodes.reserve((cl_nodes.size() + nodeCount) * 3);
                cl_leaves.reserve(cl_leaves.size() + leafValues.size());

                if( subsetSize > 0 )
                    subsets.reserve(subsets.size() + nodeCount * subsetSize);

                // nodes
                FileNodeIterator iIt = internalNodes.begin(), iEnd = internalNodes.end();

                for( ; iIt != iEnd; )
                {
                    cl_nodes.push_back((int)*(iIt++));
                    cl_nodes.push_back((int)*(iIt++));
                    cl_nodes.push_back((int)*(iIt++));

                    if( subsetSize > 0 )
                        for( int j = 0; j < subsetSize; j++, ++iIt )
                            subsets.push_back((int)*iIt);
                }

                // leaves
                iIt = leafValues.begin(), iEnd = leafValues.end();
                for( ; iIt != iEnd; ++iIt )
                    cl_leaves.push_back((float)*iIt);
            }
        }

        fn = root[CUDA_CC_FEATURES];
        if( fn.empty() )
            return false;
        std::vector<uchar> features;
        features.reserve(fn.size() * 4);
        FileNodeIterator f_it = fn.begin(), f_end = fn.end();
        for (; f_it != f_end; ++f_it)
        {
            FileNode rect = (*f_it)[CUDA_CC_RECT];
            FileNodeIterator r_it = rect.begin();
            features.push_back(saturate_cast<uchar>((int)*(r_it++)));
            features.push_back(saturate_cast<uchar>((int)*(r_it++)));
            features.push_back(saturate_cast<uchar>((int)*(r_it++)));
            features.push_back(saturate_cast<uchar>((int)*(r_it++)));
        }

        // copy data structures on gpu
        stage_mat.upload(cv::Mat(1, (int) (stages.size() * sizeof(Stage)), CV_8UC1, (uchar*)&(stages[0]) ));
        trees_mat.upload(cv::Mat(cl_trees).reshape(1,1));
        nodes_mat.upload(cv::Mat(cl_nodes).reshape(1,1));
        leaves_mat.upload(cv::Mat(cl_leaves).reshape(1,1));
        subsets_mat.upload(cv::Mat(subsets).reshape(1,1));
        features_mat.upload(cv::Mat(features).reshape(4,1));

        return true;
    }

    void LbpCascade_Impl::allocateBuffers(cv::Size frame)
    {
        if (frame == cv::Size())
            return;

        if (resuzeBuffer.empty() || frame.width > resuzeBuffer.cols || frame.height > resuzeBuffer.rows)
        {
            resuzeBuffer.create(frame, CV_8UC1);

            integral.create(frame.height + 1, integralFactor * (frame.width + 1), CV_32SC1);

        #ifdef HAVE_OPENCV_CUDALEGACY
            NcvSize32u roiSize;
            roiSize.width = frame.width;
            roiSize.height = frame.height;

            cudaDeviceProp prop;
            cudaSafeCall( cudaGetDeviceProperties(&prop, cv::cuda::getDevice()) );

            Ncv32u bufSize;
            ncvSafeCall( nppiStIntegralGetSize_8u32u(roiSize, &bufSize, prop) );
            integralBuffer.create(1, bufSize, CV_8UC1);
        #endif

            candidates.create(1 , frame.width >> 1, CV_32SC4);
        }
    }

}

//
// create
//

Ptr<cuda::CascadeClassifier> cv::cuda::CascadeClassifier::create(const String& filename)
{
    String fext = filename.substr(filename.find_last_of(".") + 1);
    fext = fext.toLowerCase();

    if (fext == "nvbin")
    {
    #ifndef HAVE_OPENCV_CUDALEGACY
        CV_Error(Error::StsUnsupportedFormat, "OpenCV CUDA objdetect was built without HaarCascade");
        return Ptr<cuda::CascadeClassifier>();
    #else
        return makePtr<HaarCascade_Impl>(filename);
    #endif
    }

    FileStorage fs(filename, FileStorage::READ);

    if (!fs.isOpened())
    {
    #ifndef HAVE_OPENCV_CUDALEGACY
        CV_Error(Error::StsUnsupportedFormat, "OpenCV CUDA objdetect was built without HaarCascade");
        return Ptr<cuda::CascadeClassifier>();
    #else
        return makePtr<HaarCascade_Impl>(filename);
    #endif
    }

    const char *CUDA_CC_LBP = "LBP";
    String featureTypeStr = (String)fs.getFirstTopLevelNode()["featureType"];
    if (featureTypeStr == CUDA_CC_LBP)
    {
        return makePtr<LbpCascade_Impl>(fs);
    }
    else
    {
    #ifndef HAVE_OPENCV_CUDALEGACY
        CV_Error(Error::StsUnsupportedFormat, "OpenCV CUDA objdetect was built without HaarCascade");
        return Ptr<cuda::CascadeClassifier>();
    #else
        return makePtr<HaarCascade_Impl>(filename);
    #endif
    }

    CV_Error(Error::StsUnsupportedFormat, "Unsupported format for CUDA CascadeClassifier");
    return Ptr<cuda::CascadeClassifier>();
}

Ptr<cuda::CascadeClassifier> cv::cuda::CascadeClassifier::create(const FileStorage& file)
{
    return makePtr<LbpCascade_Impl>(file);
}

#endif
