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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
#include "op_halide.hpp"
#include "op_inf_engine.hpp"
#include "op_vkcom.hpp"
#include "halide_scheduler.hpp"
#include <set>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iterator>
#include <numeric>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

// this option is useful to run valgrind memory errors detection
static bool DNN_DISABLE_MEMORY_OPTIMIZATIONS = utils::getConfigurationParameterBool("OPENCV_DNN_DISABLE_MEMORY_OPTIMIZATIONS", false);

#ifdef HAVE_OPENCL
static bool DNN_OPENCL_ALLOW_ALL_DEVICES = utils::getConfigurationParameterBool("OPENCV_DNN_OPENCL_ALLOW_ALL_DEVICES", false);
#endif

static int PARAM_DNN_BACKEND_DEFAULT = (int)utils::getConfigurationParameterSizeT("OPENCV_DNN_BACKEND_DEFAULT",
#ifdef HAVE_INF_ENGINE
    (size_t)DNN_BACKEND_INFERENCE_ENGINE
#else
    (size_t)DNN_BACKEND_OPENCV
#endif
);

// Additional checks (slowdowns execution!)
static bool DNN_CHECK_NAN_INF = utils::getConfigurationParameterBool("OPENCV_DNN_CHECK_NAN_INF", false);
static bool DNN_CHECK_NAN_INF_DUMP = utils::getConfigurationParameterBool("OPENCV_DNN_CHECK_NAN_INF_DUMP", false);
static bool DNN_CHECK_NAN_INF_RAISE_ERROR = utils::getConfigurationParameterBool("OPENCV_DNN_CHECK_NAN_INF_RAISE_ERROR", false);

using std::vector;
using std::map;
using std::make_pair;
using std::set;

namespace
{
    typedef std::vector<MatShape> ShapesVec;

    struct LayerShapes
    {
        ShapesVec in, out, internal;
        // No guarantees that layer which support in-place computations
        // will be computed in-place (input.data_ptr == output.data_ptr).
        // If layer said that it could work in-place and layers after it
        // no longer use input blob, we'll set output = input.
        bool supportInPlace;
        LayerShapes() {supportInPlace = false;}
    };
}

Mat blobFromImage(InputArray image, double scalefactor, const Size& size,
                  const Scalar& mean, bool swapRB, bool crop, int ddepth)
{
    CV_TRACE_FUNCTION();
    Mat blob;
    blobFromImage(image, blob, scalefactor, size, mean, swapRB, crop, ddepth);
    return blob;
}

void blobFromImage(InputArray image, OutputArray blob, double scalefactor,
                   const Size& size, const Scalar& mean, bool swapRB, bool crop, int ddepth)
{
    CV_TRACE_FUNCTION();
    std::vector<Mat> images(1, image.getMat());
    blobFromImages(images, blob, scalefactor, size, mean, swapRB, crop, ddepth);
}

Mat blobFromImages(InputArrayOfArrays images, double scalefactor, Size size,
                   const Scalar& mean, bool swapRB, bool crop, int ddepth)
{
    CV_TRACE_FUNCTION();
    Mat blob;
    blobFromImages(images, blob, scalefactor, size, mean, swapRB, crop, ddepth);
    return blob;
}

void blobFromImages(InputArrayOfArrays images_, OutputArray blob_, double scalefactor,
                    Size size, const Scalar& mean_, bool swapRB, bool crop, int ddepth)
{
    CV_TRACE_FUNCTION();
    CV_CheckType(ddepth, ddepth == CV_32F || ddepth == CV_8U, "Blob depth should be CV_32F or CV_8U");
    if (ddepth == CV_8U)
    {
        CV_CheckEQ(scalefactor, 1.0, "Scaling is not supported for CV_8U blob depth");
        CV_Assert(mean_ == Scalar() && "Mean subtraction is not supported for CV_8U blob depth");
    }

    std::vector<Mat> images;
    images_.getMatVector(images);
    CV_Assert(!images.empty());
    for (int i = 0; i < images.size(); i++)
    {
        Size imgSize = images[i].size();
        if (size == Size())
            size = imgSize;
        if (size != imgSize)
        {
            if(crop)
            {
              float resizeFactor = std::max(size.width / (float)imgSize.width,
                                            size.height / (float)imgSize.height);
              resize(images[i], images[i], Size(), resizeFactor, resizeFactor, INTER_LINEAR);
              Rect crop(Point(0.5 * (images[i].cols - size.width),
                              0.5 * (images[i].rows - size.height)),
                        size);
              images[i] = images[i](crop);
            }
            else
              resize(images[i], images[i], size, 0, 0, INTER_LINEAR);
        }
        if(images[i].depth() == CV_8U && ddepth == CV_32F)
            images[i].convertTo(images[i], CV_32F);
        Scalar mean = mean_;
        if (swapRB)
            std::swap(mean[0], mean[2]);

        images[i] -= mean;
        images[i] *= scalefactor;
    }

    size_t i, nimages = images.size();
    Mat image0 = images[0];
    int nch = image0.channels();
    CV_Assert(image0.dims == 2);
    Mat image;
    if (nch == 3 || nch == 4)
    {
        int sz[] = { (int)nimages, nch, image0.rows, image0.cols };
        blob_.create(4, sz, ddepth);
        Mat blob = blob_.getMat();
        Mat ch[4];

        for( i = 0; i < nimages; i++ )
        {
            image = images[i];
            CV_Assert(image.depth() == blob_.depth());
            nch = image.channels();
            CV_Assert(image.dims == 2 && (nch == 3 || nch == 4));
            CV_Assert(image.size() == image0.size());

            for( int j = 0; j < nch; j++ )
                ch[j] = Mat(image.rows, image.cols, ddepth, blob.ptr((int)i, j));
            if(swapRB)
                std::swap(ch[0], ch[2]);
            split(image, ch);
        }
    }
    else
    {
       CV_Assert(nch == 1);
       int sz[] = { (int)nimages, 1, image0.rows, image0.cols };
       blob_.create(4, sz, ddepth);
       Mat blob = blob_.getMat();

       for( i = 0; i < nimages; i++ )
       {
           Mat image = images[i];
           CV_Assert(image.depth() == blob_.depth());
           nch = image.channels();
           CV_Assert(image.dims == 2 && (nch == 1));
           CV_Assert(image.size() == image0.size());

           image.copyTo(Mat(image.rows, image.cols, ddepth, blob.ptr((int)i, 0)));
       }
    }
}

void imagesFromBlob(const cv::Mat& blob_, OutputArrayOfArrays images_)
{
    CV_TRACE_FUNCTION();

    //A blob is a 4 dimensional matrix in floating point precision
    //blob_[0] = batchSize = nbOfImages
    //blob_[1] = nbOfChannels
    //blob_[2] = height
    //blob_[3] = width
    CV_Assert(blob_.depth() == CV_32F);
    CV_Assert(blob_.dims == 4);

    images_.create(cv::Size(1, blob_.size[0]), blob_.depth());

    std::vector<Mat> vectorOfChannels(blob_.size[1]);
    for (int n = 0; n <  blob_.size[0]; ++n)
    {
        for (int c = 0; c < blob_.size[1]; ++c)
        {
            vectorOfChannels[c] = getPlane(blob_, n, c);
        }
        cv::merge(vectorOfChannels, images_.getMatRef(n));
    }
}

class OpenCLBackendWrapper : public BackendWrapper
{
public:
    OpenCLBackendWrapper(Mat& m) : BackendWrapper(DNN_BACKEND_OPENCV, DNN_TARGET_OPENCL)
    {
        m.copyTo(umat);
        host = &m;
        hostDirty = false;
    }

    OpenCLBackendWrapper(const Ptr<BackendWrapper>& baseBuffer, Mat& m)
        : BackendWrapper(DNN_BACKEND_OPENCV, DNN_TARGET_OPENCL)
    {
        Ptr<OpenCLBackendWrapper> base = baseBuffer.dynamicCast<OpenCLBackendWrapper>();
        CV_Assert(!base.empty());

        host = &m;

        int shape[] = {1, (int)base->umat.total()};
        umat = base->umat.reshape(1, 2, &shape[0])
                         .colRange(0, host->total())
                         .reshape(1, host->dims, &host->size[0]);
        hostDirty = false;
    }

    static Ptr<BackendWrapper> create(Mat& m)
    {
        return Ptr<BackendWrapper>(new OpenCLBackendWrapper(m));
    }

    static Ptr<BackendWrapper> create(const Ptr<BackendWrapper>& baseBuffer, Mat& m)
    {
        return Ptr<BackendWrapper>(new OpenCLBackendWrapper(baseBuffer, m));
    }

    static std::vector<UMat> getUMatVector(const std::vector<Ptr<BackendWrapper> >& wrappers)
    {
        const int numWrappers = wrappers.size();
        std::vector<UMat> mats(wrappers.size());
        for (int i = 0; i < numWrappers; ++i)
        {
            Ptr<OpenCLBackendWrapper> umatWrapper = wrappers[i].dynamicCast<OpenCLBackendWrapper>();
            CV_Assert(!umatWrapper.empty());
            umatWrapper->copyToDevice();
            mats[i] = umatWrapper->umat;
        }
        return mats;
    }

    // Replaces all umats in wrappers to specific ones.
    static void update(const std::vector<Ptr<BackendWrapper> >& wrappers,
                       const std::vector<UMat>& umats)
    {
        CV_Assert(wrappers.size() == umats.size());
        for (int i = 0, n = umats.size(); i < n; ++i)
        {
            Ptr<OpenCLBackendWrapper> umatWrapper = wrappers[i].dynamicCast<OpenCLBackendWrapper>();
            CV_Assert(!umatWrapper.empty());
            umatWrapper->umat = umats[i];
        }
    }

    ~OpenCLBackendWrapper() {}

    // Copies data from device to a host memory.
    virtual void copyToHost() CV_OVERRIDE
    {
        umat.copyTo(*host);
    }

    virtual void setHostDirty() CV_OVERRIDE
    {
        hostDirty = true;
    };

    void copyToDevice()
    {
        if (hostDirty)
        {
            host->copyTo(umat);
            hostDirty = false;
        }
    }

private:
    UMat umat;
    Mat* host;
    bool hostDirty;
};

struct LayerPin
{
    int lid;
    int oid;

    LayerPin(int layerId = -1, int outputId = -1)
        : lid(layerId), oid(outputId) {}

    bool valid() const
    {
        return (lid >= 0 && oid >= 0);
    }

    bool equal(const LayerPin &r) const
    {
        return (lid == r.lid && oid == r.oid);
    }

    bool operator<(const LayerPin &r) const
    {
        return lid < r.lid || lid == r.lid && oid < r.oid;
    }

    bool operator ==(const LayerPin &r) const
    {
        return lid == r.lid && oid == r.oid;
    }
};

struct LayerData
{
    LayerData() : id(-1), skip(false), flag(0) {}
    LayerData(int _id, const String &_name, const String &_type, LayerParams &_params)
        : id(_id), name(_name), type(_type), params(_params), skip(false), flag(0)
    {
        CV_TRACE_FUNCTION();

        //add logging info
        params.name = name;
        params.type = type;
    }

    int id;
    String name;
    String type;
    LayerParams params;

    std::vector<LayerPin> inputBlobsId;
    std::set<int> inputLayersId;
    std::set<int> requiredOutputs;
    std::vector<LayerPin> consumers;
    std::vector<Ptr<BackendWrapper> > outputBlobsWrappers;
    std::vector<Ptr<BackendWrapper> > inputBlobsWrappers;
    std::vector<Ptr<BackendWrapper> > internalBlobsWrappers;

    Ptr<Layer> layerInstance;
    std::vector<Mat> outputBlobs;
    std::vector<Mat*> inputBlobs;
    std::vector<Mat> internals;
    // Computation nodes of implemented backends (except DEFAULT).
    std::map<int, Ptr<BackendNode> > backendNodes;
    // Flag for skip layer computation for specific backend.
    bool skip;

    int flag;

    Ptr<Layer> getLayerInstance()
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(type, "type", type.c_str());

        if (layerInstance)
            return layerInstance;

        layerInstance = LayerFactory::createLayerInstance(type, params);
        if (!layerInstance)
        {
            CV_Error(Error::StsError, "Can't create layer \"" + name + "\" of type \"" + type + "\"");
        }

        return layerInstance;
    }
};

//fake layer containing network input blobs
struct DataLayer : public Layer
{
    DataLayer() : Layer()
    {
        skip = false;
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_INFERENCE_ENGINE && inputsData.size() == 1;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        if (outputs_arr.depth() == CV_16S)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> outputs, internals;
        outputs_arr.getMatVector(outputs);
        internals_arr.getMatVector(internals);

        // Supported modes:
        // | Input type | Output type |
        // |       fp32 |        fp32 |
        // |      uint8 |        fp32 |
        for (int i = 0; i < inputsData.size(); ++i)
        {
            double scale = scaleFactors[i];
            Scalar& mean = means[i];
            CV_Assert(mean == Scalar() || inputsData[i].size[1] <= 4);
            CV_CheckTypeEQ(outputs[i].type(), CV_32FC1, "");

            bool singleMean = true;
            for (int j = 1; j < std::min(4, inputsData[i].size[1]) && singleMean; ++j)
            {
                singleMean = mean[j] == mean[j - 1];
            }

            if (singleMean)
            {
                inputsData[i].convertTo(outputs[i], CV_32F, scale, -mean[0] * scale);
            }
            else
            {
                for (int n = 0; n < inputsData[i].size[0]; ++n)
                    for (int c = 0; c < inputsData[i].size[1]; ++c)
                    {
                        Mat inp = getPlane(inputsData[i], n, c);
                        Mat out = getPlane(outputs[i], n, c);
                        inp.convertTo(out, CV_32F, scale, -mean[c] * scale);
                    }
            }
        }
    }

#ifdef HAVE_OPENCL
    std::vector<Mat> tmp_expressions;
    bool forward_ocl(InputArrayOfArrays, OutputArrayOfArrays outputs_, OutputArrayOfArrays internals_)
    {
        // Supported modes:
        // | Input type | Output type |
        // |       fp32 |        fp32 |
        // |       fp32 |        fp16 |
        // |      uint8 |        fp32 |
        std::vector<UMat> outputs;
        outputs_.getUMatVector(outputs);

        tmp_expressions.clear();
        for (int i = 0; i < inputsData.size(); ++i)
        {
            Mat inputData = inputsData[i];

            double scale = scaleFactors[i];
            Scalar& mean = means[i];

            CV_Assert(mean == Scalar() || inputsData[i].size[1] <= 4);
            bool singleMean = true;
            for (int j = 1; j < std::min(4, inputsData[i].size[1]) && singleMean; ++j)
            {
                singleMean = mean[j] == mean[j - 1];
            }

            if (outputs_.depth() == CV_16S)
            {
                if (singleMean)
                {
                    tmp_expressions.push_back(Mat(scale * (inputsData[i] - mean[0])));
                    convertFp16(tmp_expressions.back(), outputs[i]);
                }
                else
                {
                    for (int n = 0; n < inputsData[i].size[0]; ++n)
                        for (int c = 0; c < inputsData[i].size[1]; ++c)
                        {
                            Mat inp = getPlane(inputsData[i], n, c);

                            std::vector<cv::Range> plane(4, Range::all());
                            plane[0] = Range(n, n + 1);
                            plane[1] = Range(c, c + 1);
                            UMat out = outputs[i](plane).reshape(1, inp.dims, inp.size);

                            tmp_expressions.push_back(scale * (inp - mean[c]));
                            convertFp16(tmp_expressions.back(), out);
                        }
                }
            }
            else
            {
                CV_Assert(outputs_.depth() == CV_32F);
                if (singleMean)
                {
                    inputsData[i].convertTo(outputs[i], CV_32F, scale, -mean[0] * scale);
                }
                else
                {
                    for (int n = 0; n < inputsData[i].size[0]; ++n)
                        for (int c = 0; c < inputsData[i].size[1]; ++c)
                        {
                            Mat inp = getPlane(inputsData[i], n, c);

                            std::vector<cv::Range> plane(4, Range::all());
                            plane[0] = Range(n, n + 1);
                            plane[1] = Range(c, c + 1);
                            UMat out = outputs[i](plane).reshape(1, inp.dims, inp.size);

                            inp.convertTo(out, CV_32F, scale, -mean[c] * scale);
                        }
                }
            }
        }
        return true;
    }
#endif

    int outputNameToIndex(const String& tgtName) CV_OVERRIDE
    {
        int idx = (int)(std::find(outNames.begin(), outNames.end(), tgtName) - outNames.begin());
        return (idx < (int)outNames.size()) ? idx : -1;
    }

    void setNames(const std::vector<String> &names)
    {
        outNames.assign(names.begin(), names.end());
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == requiredOutputs);
        outputs.assign(inputs.begin(), inputs.end());
        return false;
    }

    virtual void finalize(InputArrayOfArrays, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        std::vector<Mat> outputs;
        outputs_arr.getMatVector(outputs);

        CV_Assert_N(outputs.size() == scaleFactors.size(), outputs.size() == means.size(),
                  inputsData.size() == outputs.size());
        skip = true;
        for (int i = 0; skip && i < inputsData.size(); ++i)
        {
            if (inputsData[i].data != outputs[i].data || scaleFactors[i] != 1.0 || means[i] != Scalar())
                skip = false;
        }
    }

    virtual Ptr<BackendNode> initInfEngine(const std::vector<Ptr<BackendWrapper> >&) CV_OVERRIDE
    {
#ifdef HAVE_INF_ENGINE
        InferenceEngine::LayerParams lp;
        lp.name = name;
        lp.type = "ScaleShift";
        lp.precision = InferenceEngine::Precision::FP32;
        std::shared_ptr<InferenceEngine::ScaleShiftLayer> ieLayer(new InferenceEngine::ScaleShiftLayer(lp));

        CV_CheckEQ(inputsData.size(), (size_t)1, "");
        CV_CheckEQ(inputsData[0].dims, 4, "");
        const size_t numChannels = inputsData[0].size[1];
        CV_Assert(numChannels <= 4);

        // Scale
        auto weights = InferenceEngine::make_shared_blob<float>(InferenceEngine::Precision::FP32,
                                                                {numChannels});
        weights->allocate();
        weights->set(std::vector<float>(numChannels, scaleFactors[0]));
        ieLayer->_weights = weights;

        // Mean subtraction
        auto biases = InferenceEngine::make_shared_blob<float>(InferenceEngine::Precision::FP32,
                                                               {numChannels});
        biases->allocate();
        std::vector<float> biasesVec(numChannels);
        for (int i = 0; i < numChannels; ++i)
        {
            biasesVec[i] = -means[0][i] * scaleFactors[0];
        }
        biases->set(biasesVec);
        ieLayer->_biases = biases;

        return Ptr<BackendNode>(new InfEngineBackendNode(ieLayer));
#endif  // HAVE_INF_ENGINE
        return Ptr<BackendNode>();
    }

    std::vector<String> outNames;
    // Preprocessing parameters for each network's input.
    std::vector<double> scaleFactors;
    std::vector<Scalar> means;
    std::vector<Mat> inputsData;
    bool skip;
};

struct BlobManager
{
public:
    // Increase references counter to layer output.
    void addReference(const LayerPin& lp)
    {
        std::map<LayerPin, int>::iterator it = refCounter.find(lp);
        if (it == refCounter.end())
            refCounter[lp] = 1;
        else
            it->second += 1;
    }

    void addReferences(const std::vector<LayerPin>& pins)
    {
        for (int i = 0; i < pins.size(); i++)
        {
            addReference(pins[i]);
        }
    }

    // Returns number of references to allocated memory that used in specific
    // layer blob.
    int numReferences(const LayerPin& lp)
    {
        std::map<LayerPin, LayerPin>::iterator mapIt = reuseMap.find(lp);
        CV_Assert(mapIt != reuseMap.end());
        LayerPin memHost = mapIt->second;

        std::map<LayerPin, int>::iterator refIt = refCounter.find(memHost);
        CV_Assert(refIt != refCounter.end());
        return refIt->second;
    }

    // Reuse data allocated in <host> inside the <user> blob.
    void reuse(const LayerPin& host, const LayerPin& user)
    {
        CV_Assert(reuseMap.find(user) == reuseMap.end());
        CV_Assert(reuseMap.find(host) != reuseMap.end());
        LayerPin memHost = reuseMap[host];
        reuseMap[user] = memHost;
        if (refCounter.find(memHost) != refCounter.end())
        {
            std::map<LayerPin, int>::iterator userRefIt = refCounter.find(user);
            if (userRefIt != refCounter.end())
            {
                refCounter[memHost] += userRefIt->second;
                refCounter.erase(userRefIt);
            }
            else
                refCounter[memHost] += 1;
        }
    }

    // Decrease references counter to allocated memory inside specific blob.
    void releaseReference(const LayerPin& lp)
    {
        std::map<LayerPin, LayerPin>::iterator mapIt = reuseMap.find(lp);
        CV_Assert(mapIt != reuseMap.end());

        std::map<LayerPin, int>::iterator refIt = refCounter.find(mapIt->second);
        CV_Assert(refIt != refCounter.end());
        CV_Assert(refIt->second > 0);
        refIt->second -= 1;
    }

    void releaseReferences(const std::vector<LayerPin>& pins)
    {
        for (int i = 0; i < pins.size(); i++)
        {
            releaseReference(pins[i]);
        }
    }

    void reuseOrCreate(const MatShape& shape, const LayerPin& lp, Mat& dst, bool use_half)
    {
        if (!DNN_DISABLE_MEMORY_OPTIMIZATIONS)
        {
            Mat bestBlob;
            LayerPin bestBlobPin;

            std::map<LayerPin, Mat>::iterator hostIt;
            std::map<LayerPin, int>::iterator refIt;

            const int targetTotal = total(shape);
            int bestBlobTotal = INT_MAX;

            for (hostIt = memHosts.begin(); hostIt != memHosts.end(); ++hostIt)
            {
                refIt = refCounter.find(hostIt->first);
                // Use only blobs that had references before because if not,
                // it might be used as output.
                if (refIt != refCounter.end() && refIt->second == 0)
                {
                    Mat& unusedBlob = hostIt->second;
                    if (unusedBlob.total() >= targetTotal &&
                        unusedBlob.total() < bestBlobTotal)
                    {
                        bestBlobPin = hostIt->first;
                        bestBlob = unusedBlob;
                        bestBlobTotal = unusedBlob.total();
                    }
                }
            }
            if (!bestBlob.empty())
            {
                reuse(bestBlobPin, lp);
                dst = bestBlob.reshape(1, 1).colRange(0, targetTotal).reshape(1, shape);
                return;
            }
        }

        {
            // if dst already has been allocated with total(shape) elements,
            // it won't be recreated and pointer of dst.data remains the same.
            dst.create(shape, use_half ? CV_16S : CV_32F);
            addHost(lp, dst);
        }
    }

    void allocateBlobsForLayer(LayerData &ld, const LayerShapes& layerShapes,
                               std::vector<LayerPin>& pinsForInternalBlobs,
                               bool use_half = false)
    {
        CV_TRACE_FUNCTION();

        pinsForInternalBlobs.clear();

        std::vector<Mat>& outputBlobs = ld.outputBlobs,
                &internalBlobs = ld.internals;

        const ShapesVec& outShapes = layerShapes.out,
                internalShapes = layerShapes.internal;

        outputBlobs.resize(std::max((size_t)1, outShapes.size())); //layer produce at least one output blob
        internalBlobs.resize(internalShapes.size());

        CV_Assert(ld.requiredOutputs.size() <= outShapes.size());

        // Check that layer could work in-place.
        bool inPlace = false;
        if (layerShapes.supportInPlace)
        {
            if (ld.inputBlobs.size() == 1)
            {
                // Get number of references to the input memory.
                int numRef = numReferences(ld.inputBlobsId[0]);
                // If current layer is one and only customer of this blob.
                inPlace = numRef == 1;
            }
        }

        ShapesVec shapes(outShapes);
        shapes.insert(shapes.end(), internalShapes.begin(), internalShapes.end());
        std::vector<Mat*> blobs;
        for(int i = 0; i < outputBlobs.size(); i++)
        {
            blobs.push_back(&outputBlobs[i]);
        }

        for(int i = 0; i < internalBlobs.size(); i++)
        {
            blobs.push_back(&internalBlobs[i]);
            if (total(internalShapes[i]))
            {
                pinsForInternalBlobs.push_back(LayerPin(ld.id, ld.outputBlobs.size() + i));
            }
        }

        addReferences(pinsForInternalBlobs);

        std::map<int, std::vector<int> > idxSizes;
        for(int i = 0; i < shapes.size(); i++)
        {
            idxSizes[total(shapes[i])].push_back(i);
        }

        std::map<int, std::vector<int> >::reverse_iterator it;
        for(it = idxSizes.rbegin(); it != idxSizes.rend(); it++)
        {
            for(int j = 0; j < it->second.size(); j++)
            {
                int index = it->second[j];
                if (total(shapes[index]))
                {
                    LayerPin blobPin(ld.id, index);
                    if (index < outShapes.size() && inPlace)
                    {
                        CV_Assert(ld.inputBlobs[0]->total() == total(shapes[index]));
                        ld.outputBlobs[index] = ld.inputBlobs[0]->reshape(1, shapes[index]);
                        reuse(ld.inputBlobsId[0], blobPin);
                    }
                    else
                        reuseOrCreate(shapes[index], blobPin, *blobs[index], use_half);
                }
            }
        }
    }

    // Clear internal state. Calls before an every reallocation.
    void reset()
    {
        CV_TRACE_FUNCTION();

        refCounter.clear();
        reuseMap.clear();
        memHosts.clear();
    }

private:
    // Register allocated memory.
    void addHost(const LayerPin& lp, const Mat& mat)
    {
        CV_Assert(memHosts.find(lp) == memHosts.end());
        reuseMap[lp] = lp;
        memHosts[lp] = mat;
    }

    std::map<LayerPin, int> refCounter;
    // Maps pin to origin blob (for whom memory was allocated firstly).
    // For origin blobs key == value.
    std::map<LayerPin, LayerPin> reuseMap;
    std::map<LayerPin, Mat> memHosts;
};

static Ptr<BackendWrapper> wrapMat(int backendId, int targetId, cv::Mat& m)
{
    if (backendId == DNN_BACKEND_OPENCV)
    {
        if (targetId == DNN_TARGET_CPU)
            return Ptr<BackendWrapper>();
        else if (IS_DNN_OPENCL_TARGET(targetId))
            return OpenCLBackendWrapper::create(m);
        else
            CV_Error(Error::StsNotImplemented, "Unknown target identifier");
    }
    else if (backendId == DNN_BACKEND_HALIDE)
    {
        CV_Assert(haveHalide());
#ifdef HAVE_HALIDE
        return Ptr<BackendWrapper>(new HalideBackendWrapper(targetId, m));
#endif  // HAVE_HALIDE
    }
    else if (backendId == DNN_BACKEND_INFERENCE_ENGINE)
    {
        CV_Assert(haveInfEngine());
#ifdef HAVE_INF_ENGINE
        return Ptr<BackendWrapper>(new InfEngineBackendWrapper(targetId, m));
#endif  // HAVE_INF_ENGINE
    }
    else if (backendId == DNN_BACKEND_VKCOM)
    {
        CV_Assert(haveVulkan());
#ifdef HAVE_VULKAN
        return Ptr<BackendWrapper>(new VkComBackendWrapper(m));
#endif  // HAVE_VULKAN
    }
    else
        CV_Error(Error::StsNotImplemented, "Unknown backend identifier");
    return Ptr<BackendWrapper>();
}

struct Net::Impl
{
    typedef std::map<int, LayerShapes> LayersShapesMap;
    typedef std::map<int, LayerData> MapIdToLayerData;

    ~Impl()
    {
#ifdef HAVE_VULKAN
        // Vulkan requires explicit releasing the child objects of
        // VkDevice object prior to releasing VkDevice object itself.
        layers.clear();
        backendWrappers.clear();
        vkcom::deinitPerThread();
#endif
    }
    Impl()
    {
#ifdef HAVE_VULKAN
        vkcom::initPerThread();
#endif
        //allocate fake net input layer
        netInputLayer = Ptr<DataLayer>(new DataLayer());
        LayerData &inpl = layers.insert( make_pair(0, LayerData()) ).first->second;
        inpl.id = 0;
        netInputLayer->name = inpl.name = "_input";
        inpl.type = "__NetInputLayer__";
        inpl.layerInstance = netInputLayer;
        layerNameToId.insert(std::make_pair(inpl.name, inpl.id));

        lastLayerId = 0;
        netWasAllocated = false;
        fusion = true;
        preferableBackend = DNN_BACKEND_DEFAULT;
        preferableTarget = DNN_TARGET_CPU;
        skipInfEngineInit = false;
    }

    Ptr<DataLayer> netInputLayer;
    std::vector<LayerPin> blobsToKeep;
    MapIdToLayerData layers;
    std::map<String, int> layerNameToId;
    BlobManager blobManager;
    int preferableBackend;
    int preferableTarget;
    String halideConfigFile;
    bool skipInfEngineInit;
    // Map host data to backend specific wrapper.
    std::map<void*, Ptr<BackendWrapper> > backendWrappers;

    int lastLayerId;

    bool netWasAllocated;
    bool fusion;
    std::vector<int64> layersTimings;
    Mat output_blob;

    Ptr<BackendWrapper> wrap(Mat& host)
    {
        if (preferableBackend == DNN_BACKEND_OPENCV && preferableTarget == DNN_TARGET_CPU)
            return Ptr<BackendWrapper>();

        MatShape shape(host.dims);
        for (int i = 0; i < host.dims; ++i)
            shape[i] = host.size[i];

        void* data = host.data;
        if (backendWrappers.find(data) != backendWrappers.end())
        {
            Ptr<BackendWrapper> baseBuffer = backendWrappers[data];
            if (preferableBackend == DNN_BACKEND_OPENCV)
            {
                CV_Assert(IS_DNN_OPENCL_TARGET(preferableTarget));
                return OpenCLBackendWrapper::create(baseBuffer, host);
            }
            else if (preferableBackend == DNN_BACKEND_HALIDE)
            {
                CV_Assert(haveHalide());
  #ifdef HAVE_HALIDE
                return Ptr<BackendWrapper>(new HalideBackendWrapper(baseBuffer, shape));
  #endif  // HAVE_HALIDE
            }
            else if (preferableBackend == DNN_BACKEND_INFERENCE_ENGINE)
            {
                return wrapMat(preferableBackend, preferableTarget, host);
            }
            else if (preferableBackend == DNN_BACKEND_VKCOM)
            {
  #ifdef HAVE_VULKAN
                return Ptr<BackendWrapper>(new VkComBackendWrapper(baseBuffer, host));
  #endif
            }
            else
                CV_Error(Error::StsNotImplemented, "Unknown backend identifier");
        }

        Ptr<BackendWrapper> wrapper = wrapMat(preferableBackend, preferableTarget, host);
        backendWrappers[data] = wrapper;
        return wrapper;
    }

#ifdef HAVE_HALIDE
    void compileHalide()
    {
        CV_TRACE_FUNCTION();

        CV_Assert(preferableBackend == DNN_BACKEND_HALIDE);

        HalideScheduler scheduler(halideConfigFile);
        std::vector< std::reference_wrapper<LayerData> > compileList; compileList.reserve(64);
        for (MapIdToLayerData::iterator it = layers.begin(); it != layers.end(); ++it)
        {
            LayerData &ld = it->second;
            Ptr<Layer> layer = ld.layerInstance;
            if (layer->supportBackend(DNN_BACKEND_HALIDE) && !ld.skip)
            {
                CV_Assert(!ld.backendNodes[DNN_BACKEND_HALIDE].empty());
                bool scheduled = scheduler.process(ld.backendNodes[DNN_BACKEND_HALIDE]);
                if (!scheduled)
                {
                    // Use automatic scheduling provided by layer.
                    layer->applyHalideScheduler(ld.backendNodes[DNN_BACKEND_HALIDE],
                                                ld.inputBlobs, ld.outputBlobs,
                                                preferableTarget);
                }
                compileList.emplace_back(ld);
            }
        }
        std::atomic<int> progress(0);
        auto fn = ([&] () -> void
        {
            for (;;)
            {
                int id = progress.fetch_add(1);
                if ((size_t)id >= compileList.size())
                    return;
                const LayerData& ld = compileList[id].get();
                Ptr<BackendNode> node = ld.backendNodes.find(DNN_BACKEND_HALIDE)->second;
                dnn::compileHalide(ld.outputBlobs, node, preferableTarget);
            }
        });
        size_t num_threads = std::min(compileList.size(), (size_t)std::thread::hardware_concurrency());
        num_threads = std::max((size_t)1u, std::min((size_t)8u, num_threads));
        std::vector<std::thread> threads(num_threads - 1);
        for (auto& t: threads) t = std::thread(fn);
        fn(); // process own tasks
        for (auto& t: threads) t.join();
    }
#endif

    void clear()
    {
        CV_TRACE_FUNCTION();

        MapIdToLayerData::iterator it;
        for (it = layers.begin(); it != layers.end(); it++)
        {
            if (it->second.id != 0) {
                it->second.inputBlobs.clear();
                it->second.outputBlobs.clear();
                it->second.internals.clear();
            }
            it->second.skip = false;
            //it->second.consumers.clear();
            Ptr<Layer> currLayer = it->second.layerInstance;

            if( currLayer.empty() )
                continue;

            currLayer->unsetAttached();

            Ptr<PoolingLayer> poolingLayer = currLayer.dynamicCast<PoolingLayer>();
            if( !poolingLayer.empty() )
            {
                poolingLayer->computeMaxIdx = true;
            }
        }

        layersTimings.clear();
    }

    void setUpNet(const std::vector<LayerPin>& blobsToKeep_ = std::vector<LayerPin>())
    {
        CV_TRACE_FUNCTION();

        if (preferableBackend == DNN_BACKEND_DEFAULT)
            preferableBackend = (Backend)PARAM_DNN_BACKEND_DEFAULT;

        CV_Assert(preferableBackend != DNN_BACKEND_OPENCV ||
                  preferableTarget == DNN_TARGET_CPU ||
                  preferableTarget == DNN_TARGET_OPENCL ||
                  preferableTarget == DNN_TARGET_OPENCL_FP16);
        CV_Assert(preferableBackend != DNN_BACKEND_HALIDE ||
                  preferableTarget == DNN_TARGET_CPU ||
                  preferableTarget == DNN_TARGET_OPENCL);
        CV_Assert(preferableBackend != DNN_BACKEND_INFERENCE_ENGINE ||
                  preferableTarget == DNN_TARGET_CPU ||
                  preferableTarget == DNN_TARGET_OPENCL ||
                  preferableTarget == DNN_TARGET_OPENCL_FP16 ||
                  preferableTarget == DNN_TARGET_MYRIAD);
        CV_Assert(preferableBackend != DNN_BACKEND_VKCOM ||
                  preferableTarget == DNN_TARGET_VULKAN);
        if (!netWasAllocated || this->blobsToKeep != blobsToKeep_)
        {
            if (preferableBackend == DNN_BACKEND_OPENCV && IS_DNN_OPENCL_TARGET(preferableTarget))
#ifndef HAVE_OPENCL
            {
                CV_LOG_WARNING(NULL, "DNN: OpenCL target is not available in this OpenCV build, switching to CPU.");
                preferableTarget = DNN_TARGET_CPU;
            }
#else
            {
                if (!DNN_OPENCL_ALLOW_ALL_DEVICES)
                {
                    // Current implementation is only valid for GPU (#11494)
                    if (ocl::Device::getDefault().type() != ocl::Device::TYPE_GPU)
                    {
                        CV_LOG_WARNING(NULL, "DNN: OpenCL target is not supported with current OpenCL device (tested with GPUs only), switching to CPU.");
                        preferableTarget = DNN_TARGET_CPU;
                    }
                    else if (preferableTarget == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
                    {
                        CV_LOG_WARNING(NULL,
                            "DNN: OpenCL target with fp16 precision is not supported "
                            "with current OpenCL device (tested with Intel GPUs only), "
                            "switching to OpenCL with fp32 precision.");
                        preferableTarget = DNN_TARGET_OPENCL;
                    }
                }
            }
#endif
            if (preferableBackend == DNN_BACKEND_VKCOM && !haveVulkan())
            {
                preferableBackend = DNN_BACKEND_OPENCV;
                preferableTarget = DNN_TARGET_CPU;
            }

            clear();

            allocateLayers(blobsToKeep_);

            MapIdToLayerData::iterator it = layers.find(0);
            CV_Assert(it != layers.end());
            it->second.skip = netInputLayer->skip;

            initBackend();

            if (!netWasAllocated )
            {
#ifdef HAVE_HALIDE
                if (preferableBackend == DNN_BACKEND_HALIDE)
                    compileHalide();
#else
                CV_Assert(preferableBackend != DNN_BACKEND_HALIDE);
#endif
            }

            netWasAllocated = true;
            this->blobsToKeep = blobsToKeep_;
        }
    }

    int getLayerId(const String &layerName)
    {
        std::map<String, int>::iterator it = layerNameToId.find(layerName);
        return (it != layerNameToId.end()) ? it->second : -1;
    }

    int getLayerId(int id)
    {
        MapIdToLayerData::iterator it = layers.find(id);
        return (it != layers.end()) ? id : -1;
    }

    int getLayerId(DictValue &layerDesc)
    {
        if (layerDesc.isInt())
            return getLayerId(layerDesc.get<int>());
        else if (layerDesc.isString())
            return getLayerId(layerDesc.get<String>());

        CV_Assert(layerDesc.isInt() || layerDesc.isString());
        return -1;
    }

    String getLayerName(int id)
    {
        MapIdToLayerData::iterator it = layers.find(id);
        return (it != layers.end()) ? it->second.name : "(unknown layer)";
    }

    LayerData& getLayerData(int id)
    {
        MapIdToLayerData::iterator it = layers.find(id);

        if (it == layers.end())
            CV_Error(Error::StsObjectNotFound, format("Layer with requested id=%d not found", id));

        return it->second;
    }

    LayerData& getLayerData(const String &layerName)
    {
        int id = getLayerId(layerName);

        if (id < 0)
            CV_Error(Error::StsError, "Requested layer \"" + layerName + "\" not found");

        return getLayerData(id);
    }

    LayerData& getLayerData(const DictValue &layerDesc)
    {
        CV_Assert(layerDesc.isInt() || layerDesc.isString());
        if (layerDesc.isInt())
            return getLayerData(layerDesc.get<int>());
        else /*if (layerDesc.isString())*/
            return getLayerData(layerDesc.get<String>());
    }

    static void addLayerInput(LayerData &ld, int inNum, LayerPin from)
    {
        if ((int)ld.inputBlobsId.size() <= inNum)
        {
            ld.inputBlobsId.resize(inNum + 1);
        }
        else
        {
            LayerPin storedFrom = ld.inputBlobsId[inNum];
            if (storedFrom.valid() && !storedFrom.equal(from))
                CV_Error(Error::StsError, format("Input #%d of layer \"%s\" already was connected",
                                                 inNum, ld.name.c_str()));
        }

        ld.inputBlobsId[inNum] = from;
    }

    int resolvePinOutputName(LayerData &ld, const String &outName)
    {
        if (outName.empty())
            return 0;
        return ld.getLayerInstance()->outputNameToIndex(outName);
    }

    LayerPin getPinByAlias(const String &layerName)
    {
        LayerPin pin;
        pin.lid = (layerName.empty()) ? 0 : getLayerId(layerName);

        if (pin.lid >= 0)
            pin.oid = resolvePinOutputName(getLayerData(pin.lid), layerName);

        return pin;
    }

    std::vector<LayerPin> getLayerOutPins(const String &layerName)
    {
        int lid = (layerName.empty()) ? 0 : getLayerId(layerName);

        std::vector<LayerPin> pins;

        for (int i = 0; i < layers[lid].outputBlobs.size(); i++)
        {
            pins.push_back(LayerPin(lid, i));
        }

        return pins;
    }

    void connect(int outLayerId, int outNum, int inLayerId, int inNum)
    {
        CV_Assert(outLayerId < inLayerId);
        LayerData &ldOut = getLayerData(outLayerId);
        LayerData &ldInp = getLayerData(inLayerId);

        addLayerInput(ldInp, inNum, LayerPin(outLayerId, outNum));
        ldOut.requiredOutputs.insert(outNum);
        ldOut.consumers.push_back(LayerPin(inLayerId, outNum));
    }

    void initBackend()
    {
        CV_TRACE_FUNCTION();
        if (preferableBackend == DNN_BACKEND_OPENCV)
            CV_Assert(preferableTarget == DNN_TARGET_CPU || IS_DNN_OPENCL_TARGET(preferableTarget));
        else if (preferableBackend == DNN_BACKEND_HALIDE)
            initHalideBackend();
        else if (preferableBackend == DNN_BACKEND_INFERENCE_ENGINE)
            initInfEngineBackend();
        else if (preferableBackend == DNN_BACKEND_VKCOM)
            initVkComBackend();
        else
            CV_Error(Error::StsNotImplemented, "Unknown backend identifier");
    }

    void initHalideBackend()
    {
        CV_TRACE_FUNCTION();
        CV_Assert_N(preferableBackend == DNN_BACKEND_HALIDE, haveHalide());

        // Iterator to current layer.
        MapIdToLayerData::iterator it = layers.begin();
        // Iterator to base layer for fusion. In example, in case of conv+bn+relu
        // it'll be a conv layer.
        MapIdToLayerData::iterator baseIt = layers.begin();
        for (; it != layers.end(); it++)
        {
            LayerData &ldTop = it->second;
            Ptr<Layer> layerTop = ldTop.layerInstance;
            if (!layerTop->supportBackend(preferableBackend))
            {
                // Move base iterator to layer that don't support preferable
                // backend to prevent fusion over layer of different backend.
                baseIt = it;
                continue;
            }
            // Try to do layers fusion.
            LayerData &ldBot = baseIt->second;
            Ptr<Layer> layerBot = ldBot.layerInstance;
            // 1. Check that bottom and top from the same backends.
            if (it != layers.begin() && layerBot->supportBackend(preferableBackend))
            {
                // 2. Check that current layer works in-place.
                bool inPlace = ldTop.inputBlobs.size() == 1 &&
                               ldBot.outputBlobs.size() == 1 &&
                               ldTop.inputBlobs[0]->data ==
                               ldBot.outputBlobs[0].data;
                if (inPlace)
                {
                    // 3. Try to attach node.
                    CV_Assert(!ldBot.backendNodes[preferableBackend].empty());
                    Ptr<BackendNode> fusedNode =
                        layerTop->tryAttach(ldBot.backendNodes[preferableBackend]);
                    if (!fusedNode.empty())
                    {
                        ldTop.skip = true;
                        ldBot.backendNodes[preferableBackend] = fusedNode;
                        ldBot.outputBlobsWrappers = ldTop.outputBlobsWrappers;
                        continue;
                    }
                }
            }
            // No layers fusion.
            ldTop.skip = false;
            ldTop.backendNodes[DNN_BACKEND_HALIDE] =
                layerTop->initHalide(ldTop.inputBlobsWrappers);
            baseIt = it;
        }
    }

#ifdef HAVE_INF_ENGINE
    // Before launching Inference Engine graph we need to specify output blobs.
    // This function requests output blobs based on inputs references of
    // layers from default backend or layers from different graphs.
    void addInfEngineNetOutputs(LayerData &ld)
    {
        Ptr<InfEngineBackendNet> layerNet;
        if (ld.backendNodes.find(preferableBackend) != ld.backendNodes.end())
        {
            Ptr<BackendNode> node = ld.backendNodes[preferableBackend];
            if (!node.empty())
            {
                Ptr<InfEngineBackendNode> ieNode = node.dynamicCast<InfEngineBackendNode>();
                CV_Assert(!ieNode.empty()); CV_Assert(!ieNode->net.empty());
                layerNet = ieNode->net;
            }
        }
        // For an every input reference we check that it belongs to one of
        // the Inference Engine backend graphs. Request an output blob if it is.
        // Do nothing if layer's input is from the same graph.
        for (int i = 0; i < ld.inputBlobsId.size(); ++i)
        {
            LayerData &inpLd = layers[ld.inputBlobsId[i].lid];
            Ptr<BackendNode> inpNode = inpLd.backendNodes[preferableBackend];
            if (!inpNode.empty())
            {
                Ptr<InfEngineBackendNode> ieInpNode = inpNode.dynamicCast<InfEngineBackendNode>();
                CV_Assert(!ieInpNode.empty()); CV_Assert(!ieInpNode->net.empty());
                if (layerNet != ieInpNode->net)
                {
                    // layerNet is empty or nodes are from different graphs.
                    ieInpNode->net->addOutput(ieInpNode->layer->name);
                }
            }
        }
    }
#endif  // HAVE_INF_ENGINE

    void initVkComBackend()
    {
        CV_TRACE_FUNCTION();
        CV_Assert(preferableBackend == DNN_BACKEND_VKCOM);
#ifdef HAVE_VULKAN
        if (!haveVulkan())
            return;

        MapIdToLayerData::iterator it = layers.begin();
        for (; it != layers.end(); it++)
        {
            LayerData &ld = it->second;
            Ptr<Layer> layer = ld.layerInstance;
            if (!layer->supportBackend(preferableBackend))
            {
                continue;
            }

            if (ld.type == "Convolution")
            {
                std::vector<MatShape> in_shapes;
                std::vector<MatShape> out_shapes;
                CV_Assert(ld.inputBlobs.size() == ld.outputBlobs.size());

                for (int i = 0; i < ld.inputBlobs.size(); i++)
                {
                    in_shapes.push_back(shape(*ld.inputBlobs[i]));
                    out_shapes.push_back(shape(ld.outputBlobs[i]));
                }
                int64 flops = layer->getFLOPS(in_shapes, out_shapes);
                // FIXME
                //
                // This is a workaround for GPU hang on heavy convolution workload ( > 10 GFLOPS).
                // For the long time task, vkWaitForFences() return without error but next call on
                // vkQueueSubmit() return -4, i.e. "VK_ERROR_DEVICE_LOST" and driver reports GPU hang.
                //
                // Need more investigation on root cause of GPU hang and need to optimize convolution shader
                // to reduce process time.
                if (flops > CV_BIG_INT(10) * 1000 * 1000 * 1000)
                {
                    continue;
                }
            }

            ld.skip = false;

            try
            {
                ld.backendNodes[DNN_BACKEND_VKCOM] =
                    layer->initVkCom(ld.inputBlobsWrappers);
            }
            catch (const cv::Exception& e)
            {
                CV_LOG_ERROR(NULL, "initVkCom failed, fallback to CPU implementation. " << e.what());
                ld.backendNodes[DNN_BACKEND_VKCOM] = Ptr<BackendNode>();
            }
        }
#endif
    }

    void initInfEngineBackend()
    {
        CV_TRACE_FUNCTION();
        CV_Assert_N(preferableBackend == DNN_BACKEND_INFERENCE_ENGINE, haveInfEngine());
#ifdef HAVE_INF_ENGINE
        MapIdToLayerData::iterator it;
        Ptr<InfEngineBackendNet> net;

        for (it = layers.begin(); it != layers.end(); ++it)
        {
            LayerData &ld = it->second;
            if (ld.id == 0)
            {
                CV_Assert((netInputLayer->outNames.empty() && ld.outputBlobsWrappers.size() == 1) ||
                          (netInputLayer->outNames.size() == ld.outputBlobsWrappers.size()));
                for (int i = 0; i < ld.outputBlobsWrappers.size(); ++i)
                {
                    InferenceEngine::DataPtr dataPtr = infEngineDataNode(ld.outputBlobsWrappers[i]);
                    dataPtr->name = netInputLayer->outNames.empty() ? ld.name : netInputLayer->outNames[i];
                }
            }
            else
            {
                for (int i = 0; i < ld.outputBlobsWrappers.size(); ++i)
                {
                    InferenceEngine::DataPtr dataPtr = infEngineDataNode(ld.outputBlobsWrappers[i]);
                    dataPtr->name = ld.name;
                }
            }
        }

        if (skipInfEngineInit)
        {
            Ptr<BackendNode> node = layers[lastLayerId].backendNodes[preferableBackend];
            CV_Assert(!node.empty());

            Ptr<InfEngineBackendNode> ieNode = node.dynamicCast<InfEngineBackendNode>();
            CV_Assert(!ieNode.empty());

            for (it = layers.begin(); it != layers.end(); ++it)
            {
                LayerData &ld = it->second;
                if (ld.id == 0)
                {
                    for (int i = 0; i < ld.inputBlobsWrappers.size(); ++i)
                    {
                        InferenceEngine::DataPtr dataPtr = infEngineDataNode(ld.inputBlobsWrappers[i]);
                        dataPtr->name = netInputLayer->outNames[i];
                    }
                }
                else
                {
                    for (int i = 0; i < ld.outputBlobsWrappers.size(); ++i)
                    {
                        InferenceEngine::DataPtr dataPtr = infEngineDataNode(ld.outputBlobsWrappers[i]);
                        dataPtr->name = ld.name;
                    }
                }
                ieNode->net->addBlobs(ld.inputBlobsWrappers);
                ieNode->net->addBlobs(ld.outputBlobsWrappers);
                ld.skip = true;
            }
            layers[lastLayerId].skip = false;
            ieNode->net->init(preferableTarget);
            return;
        }

        // Build Inference Engine networks from sets of layers that support this
        // backend. Split a whole model on several Inference Engine networks if
        // some of layers is not implemented.

        // Set of all input and output blobs wrappers for current network.
        std::map<LayerPin, Ptr<BackendWrapper> > netBlobsWrappers;
        for (it = layers.begin(); it != layers.end(); ++it)
        {
            LayerData &ld = it->second;
            if (ld.id == 0 && ld.skip)
                continue;
            bool fused = ld.skip;

            Ptr<Layer> layer = ld.layerInstance;
            if (!fused && !layer->supportBackend(preferableBackend))
            {
                addInfEngineNetOutputs(ld);
                net = Ptr<InfEngineBackendNet>();
                netBlobsWrappers.clear();
                layer->preferableTarget = DNN_TARGET_CPU;
                continue;
            }
            ld.skip = true;  // Initially skip all Inference Engine supported layers.

            // Create a new network if one of inputs from different Inference Engine graph.
            for (int i = 0; i < ld.inputBlobsId.size(); ++i)
            {
                LayerData &inpLd = layers[ld.inputBlobsId[i].lid];
                Ptr<BackendNode> inpNode = inpLd.backendNodes[preferableBackend];
                if (!inpNode.empty())
                {
                    Ptr<InfEngineBackendNode> ieInpNode = inpNode.dynamicCast<InfEngineBackendNode>();
                    CV_Assert(!ieInpNode.empty()); CV_Assert(!ieInpNode->net.empty());
                    if (ieInpNode->net != net)
                    {
                        net = Ptr<InfEngineBackendNet>();
                        netBlobsWrappers.clear();
                        break;
                    }
                }
            }

            // The same blobs wrappers cannot be shared between two Inference Engine
            // networks because of explicit references between layers and blobs.
            // So we need to rewrap all the external blobs.
            for (int i = 0; i < ld.inputBlobsId.size(); ++i)
            {
                LayerPin inPin = ld.inputBlobsId[i];
                auto it = netBlobsWrappers.find(inPin);
                if (it == netBlobsWrappers.end())
                {
                    ld.inputBlobsWrappers[i] = InfEngineBackendWrapper::create(ld.inputBlobsWrappers[i]);
                    netBlobsWrappers[inPin] = ld.inputBlobsWrappers[i];
                }
                else
                    ld.inputBlobsWrappers[i] = it->second;
            }
            netBlobsWrappers[LayerPin(ld.id, 0)] = ld.outputBlobsWrappers[0];

            Ptr<BackendNode> node;
            if (!net.empty())
            {
                if (fused)
                {
                    bool inPlace = ld.inputBlobsId.size() == 1 && ld.outputBlobs.size() == 1 &&
                                   ld.inputBlobs[0]->data == ld.outputBlobs[0].data;
                    CV_Assert(inPlace);
                    node = layers[ld.inputBlobsId[0].lid].backendNodes[preferableBackend];
                    ld.inputBlobsWrappers = layers[ld.inputBlobsId[0].lid].inputBlobsWrappers;
                }
            }
            else
                net = Ptr<InfEngineBackendNet>(new InfEngineBackendNet());

            if (!fused)
            {
                node = layer->initInfEngine(ld.inputBlobsWrappers);
            }
            else if (node.empty())
                continue;

            CV_Assert(!node.empty());
            ld.backendNodes[preferableBackend] = node;

            Ptr<InfEngineBackendNode> ieNode = node.dynamicCast<InfEngineBackendNode>();
            CV_Assert(!ieNode.empty());
            ieNode->net = net;

            auto weightableLayer = std::dynamic_pointer_cast<InferenceEngine::WeightableLayer>(ieNode->layer);
            if ((preferableTarget == DNN_TARGET_OPENCL_FP16 || preferableTarget == DNN_TARGET_MYRIAD) && !fused)
            {
                ieNode->layer->precision = InferenceEngine::Precision::FP16;
                if (weightableLayer)
                {
                    if (weightableLayer->_weights)
                        weightableLayer->_weights = convertFp16(weightableLayer->_weights);
                    if (weightableLayer->_biases)
                        weightableLayer->_biases = convertFp16(weightableLayer->_biases);
                }
                else
                {
                    for (const auto& weights : {"weights", "biases"})
                    {
                        auto it = ieNode->layer->blobs.find(weights);
                        if (it != ieNode->layer->blobs.end())
                            it->second = convertFp16(it->second);
                    }
                }
            }
            if (weightableLayer)
            {
                if (weightableLayer->_weights)
                    weightableLayer->blobs["weights"] = weightableLayer->_weights;
                if (weightableLayer->_biases)
                    weightableLayer->blobs["biases"] = weightableLayer->_biases;
            }
            ieNode->connect(ld.inputBlobsWrappers, ld.outputBlobsWrappers);
            net->addBlobs(ld.inputBlobsWrappers);
            net->addBlobs(ld.outputBlobsWrappers);

            if (!fused)
                net->addLayer(ieNode->layer);
            addInfEngineNetOutputs(ld);
        }

        // Initialize all networks.
        std::set<InfEngineBackendNet> initializedNets;
        for (MapIdToLayerData::reverse_iterator it = layers.rbegin(); it != layers.rend(); ++it)
        {
            LayerData &ld = it->second;
            if (ld.backendNodes.find(preferableBackend) == ld.backendNodes.end())
                continue;

            Ptr<BackendNode> node = ld.backendNodes[preferableBackend];
            if (node.empty())
                continue;

            Ptr<InfEngineBackendNode> ieNode = node.dynamicCast<InfEngineBackendNode>();
            if (ieNode.empty())
                continue;

            CV_Assert(!ieNode->net.empty());

            if (!ieNode->net->isInitialized())
            {
                ieNode->net->init(preferableTarget);
                ld.skip = false;
            }
        }
#endif  // HAVE_INF_ENGINE
    }

    void allocateLayer(int lid, const LayersShapesMap& layersShapes)
    {
        CV_TRACE_FUNCTION();

        LayerData &ld = layers[lid];

        //already allocated
        if (ld.flag)
            return;

        size_t ninputs = ld.inputBlobsId.size();
#if 0
        printf("layer %s:", ld.name.c_str());
        for (size_t i = 0; i < ninputs; i++)
        {
            int inp_lid = ld.inputBlobsId[i].lid;
            LayerData &inp_ld = layers[inp_lid];
            int inp_outputs = (int)inp_ld.outputBlobs.size();
            std::cout << " " << inp_ld.name << "(" << inp_outputs;

            for( int j = 0; j < inp_outputs; j++ )
            {
                std::cout << (j == 0 ? ": " : ", ") << inp_ld.outputBlobs[j].size;
            }
            std::cout << ")";
        }
        printf("\n");
#endif

        //determine parent layers
        for (size_t i = 0; i < ninputs; i++)
            ld.inputLayersId.insert(ld.inputBlobsId[i].lid);

        //allocate parents
        for (set<int>::iterator i = ld.inputLayersId.begin(); i != ld.inputLayersId.end(); i++)
            allocateLayer(*i, layersShapes);

        //bind inputs
        if (ld.id == 0)  // DataLayer
        {
            ninputs = netInputLayer->inputsData.size();
            ld.inputBlobsWrappers.resize(ninputs);
            for (size_t i = 0; i < ninputs; i++)
            {
                ld.inputBlobsWrappers[i] = wrap(netInputLayer->inputsData[i]);
            }
        }
        else
        {
            ld.inputBlobs.resize(ninputs);
            ld.inputBlobsWrappers.resize(ninputs);
            for (size_t i = 0; i < ninputs; i++)
            {
                LayerPin from = ld.inputBlobsId[i];
                CV_Assert(from.valid());
                CV_DbgAssert(layers.count(from.lid) && (int)layers[from.lid].outputBlobs.size() > from.oid);
                ld.inputBlobs[i] = &layers[from.lid].outputBlobs[from.oid];
                ld.inputBlobsWrappers[i] = layers[from.lid].outputBlobsWrappers[from.oid];
            }
        }

        LayersShapesMap::const_iterator layerShapesIt = layersShapes.find(lid);

        CV_Assert(layerShapesIt != layersShapes.end());

        std::vector<LayerPin> pinsForInternalBlobs;
        blobManager.allocateBlobsForLayer(ld, layerShapesIt->second, pinsForInternalBlobs,
                                          preferableBackend == DNN_BACKEND_OPENCV &&
                                          preferableTarget == DNN_TARGET_OPENCL_FP16);
        ld.outputBlobsWrappers.resize(ld.outputBlobs.size());
        for (int i = 0; i < ld.outputBlobs.size(); ++i)
        {
            ld.outputBlobsWrappers[i] = wrap(ld.outputBlobs[i]);
        }
        ld.internalBlobsWrappers.resize(ld.internals.size());
        for (int i = 0; i < ld.internals.size(); ++i)
        {
            ld.internalBlobsWrappers[i] = wrap(ld.internals[i]);
        }

        Ptr<Layer> layerPtr = ld.getLayerInstance();
        {
            std::vector<Mat> inps(ld.inputBlobs.size());
            for (int i = 0; i < ld.inputBlobs.size(); ++i)
            {
                inps[i] = *ld.inputBlobs[i];
            }
            layerPtr->finalize(inps, ld.outputBlobs);
            layerPtr->preferableTarget = preferableTarget;
#if 0
            std::cout << "\toutputs:";
            size_t noutputs = ld.outputBlobs.size();
            for (size_t j = 0; j < noutputs; j++)
            {
                std::cout << (j == 0 ? " " : ", ") << ld.outputBlobs[j].size;
            }
            std::cout << "\n";
#endif
        }

        // After allocation of layer, we decrease counters to it's input blobs.
        blobManager.releaseReferences(ld.inputBlobsId);
        blobManager.releaseReferences(pinsForInternalBlobs);

        ld.flag = 1;
    }

#if 0
#define printf_(args) printf args
#else
#define printf_(args)
#endif

    void fuseLayers(const std::vector<LayerPin>& blobsToKeep_)
    {
        if( !fusion || preferableBackend != DNN_BACKEND_OPENCV &&
                       preferableBackend != DNN_BACKEND_INFERENCE_ENGINE)
            return;

        CV_TRACE_FUNCTION();

        // scan through all the layers. If there is convolution layer followed by the activation layer,
        // we try to embed this activation into the convolution and disable separate execution of the activation
        std::set<LayerPin> pinsToKeep(blobsToKeep_.begin(),
                                      blobsToKeep_.end());
        MapIdToLayerData::iterator it;
        for (it = layers.begin(); it != layers.end(); it++)
        {
            int lid = it->first;
            LayerData& ld = layers[lid];
            if( ld.skip )
            {
                printf_(("skipped %s: %s\n", ld.layerInstance->name.c_str(), ld.layerInstance->type.c_str()));
                continue;
            }
            printf_(("analyzing %s: %s\n", ld.layerInstance->name.c_str(), ld.layerInstance->type.c_str()));

            // the optimization #1. try to fuse batch norm, scaling and/or activation layers
            // with the current layer if they follow it. Normally, the are fused with the convolution layer,
            // but some of them (like activation) may be fused with fully-connected, elemwise (+) and
            // some other layers.
            Ptr<Layer>& currLayer = ld.layerInstance;
            if( ld.consumers.size() == 1 && pinsToKeep.count(LayerPin(lid, 0)) == 0 )
            {
                LayerData* nextData = &layers[ld.consumers[0].lid];
                LayerPin lpNext(ld.consumers[0].lid, 0);
                while (nextData)
                {
                    Ptr<Layer> nextLayer = nextData->layerInstance;
                    if (currLayer->tryFuse(nextLayer))
                    {
                        printf_(("\tfused with %s\n", nextLayer->name.c_str()));
                        nextData->skip = true;
                        ld.outputBlobs = layers[lpNext.lid].outputBlobs;
                        ld.outputBlobsWrappers = layers[lpNext.lid].outputBlobsWrappers;
                        if (nextData->consumers.size() == 1)
                        {
                            int nextLayerId = nextData->consumers[0].lid;
                            nextData = &layers[nextLayerId];
                            lpNext = LayerPin(nextLayerId, 0);
                        }
                        else
                        {
                            nextData = 0;
                            break;
                        }
                    }
                    else
                        break;
                }

                if (preferableBackend != DNN_BACKEND_OPENCV)
                    continue;  // Go to the next layer.

                // TODO: OpenCL target support more fusion styles.
                if ( preferableBackend == DNN_BACKEND_OPENCV && IS_DNN_OPENCL_TARGET(preferableTarget) &&
                     (!cv::ocl::useOpenCL() || (ld.layerInstance->type != "Convolution" &&
                     ld.layerInstance->type != "MVN" && ld.layerInstance->type != "Pooling" &&
                     ld.layerInstance->type != "Concat")) )
                    continue;

                while (nextData)
                {
                    // For now, OpenCL target support fusion with activation of ReLU/ChannelsPReLU/Power/Tanh
                    if (IS_DNN_OPENCL_TARGET(preferableTarget) &&
                        nextData->type != "ReLU" &&
                        nextData->type != "ChannelsPReLU" &&
                        nextData->type != "ReLU6" &&
                        nextData->type != "TanH" &&
                        nextData->type != "Power")
                        break;

                    Ptr<ActivationLayer> nextActivLayer = nextData->layerInstance.dynamicCast<ActivationLayer>();
                    if (nextActivLayer.empty())
                        break;

                    if (currLayer->setActivation(nextActivLayer))
                    {
                        printf_(("\tfused with %s\n", nextActivLayer->name.c_str()));
                        nextData->skip = true;
                        ld.outputBlobs = layers[lpNext.lid].outputBlobs;
                        ld.outputBlobsWrappers = layers[lpNext.lid].outputBlobsWrappers;
                        if (nextData->consumers.size() == 1)
                        {
                            int nextLayerId = nextData->consumers[0].lid;
                            nextData = &layers[nextLayerId];
                            lpNext = LayerPin(nextLayerId, 0);
                        }
                        else
                        {
                            nextData = 0;
                            break;
                        }
                    }
                    else
                        break;
                }

                // fuse convolution layer followed by eltwise + relu
                if ( IS_DNN_OPENCL_TARGET(preferableTarget) && ld.layerInstance->type == "Convolution" )
                {
                    Ptr<EltwiseLayer> nextEltwiseLayer;
                    if( nextData )
                        nextEltwiseLayer = nextData->layerInstance.dynamicCast<EltwiseLayer>();

                    if( !nextEltwiseLayer.empty() && pinsToKeep.count(lpNext) == 0 &&
                        nextData && nextData->inputBlobsId.size() == 2 )
                    {
                        LayerData *eltwiseData = nextData;

                        // Eltwise layer has two inputs. We need to determine which
                        // is a base convolution layer and which could be used as it's bias.
                        LayerData* biasLayerData = 0;
                        for (int i = 0; i < 2; ++i)
                        {
                            LayerData *downLayerData = &layers[eltwiseData->inputBlobsId[i].lid];
                            CV_Assert(downLayerData);
                            while (downLayerData->skip)
                            {
                                if (downLayerData->inputBlobsId.size() == 1)
                                    downLayerData = &layers[downLayerData->inputBlobsId[0].lid];
                                else
                                {
                                    downLayerData = 0;
                                    break;
                                }
                            }
                            if (downLayerData && ld.id == downLayerData->id)
                            {
                                biasLayerData = &layers[eltwiseData->inputBlobsId[1 - i].lid];
                                break;
                            }
                        }
                        CV_Assert(biasLayerData);
                        {
                            if( eltwiseData->consumers.size() == 1 )
                            {
                                // fuse eltwise + activation layer
                                if (biasLayerData->id < ld.id)
                                {
                                    nextData = &layers[eltwiseData->consumers[0].lid];
                                    lpNext = LayerPin(eltwiseData->consumers[0].lid, 0);
                                    Ptr<ActivationLayer> nextActivLayer;
                                    if( nextData )
                                        nextActivLayer = nextData->layerInstance.dynamicCast<ActivationLayer>();

                                    if( !nextActivLayer.empty() && pinsToKeep.count(lpNext) == 0 &&
                                            (!nextData->type.compare("ReLU") ||
                                             !nextData->type.compare("ChannelsPReLU") ||
                                             !nextData->type.compare("Power")) &&
                                            currLayer->setActivation(nextActivLayer) )
                                    {
                                        CV_Assert_N(biasLayerData->outputBlobsWrappers.size() == 1, ld.inputBlobsWrappers.size() == 1);
                                        ld.inputBlobsWrappers.push_back(biasLayerData->outputBlobsWrappers[0]);
                                        printf_(("\tfused with %s\n", nextEltwiseLayer->name.c_str()));
                                        printf_(("\tfused with %s\n", nextActivLayer->name.c_str()));
                                        eltwiseData->skip = true;
                                        nextData->skip = true;
                                        // This optimization for cases like
                                        // some_layer   conv
                                        //   |             |
                                        //   +-- eltwise --+
                                        //          |
                                        //        activ
                                        // This way all the element-wise computations
                                        // (i.e. some_layer+conv or some_layer*conv)
                                        // would be done at [conv] layer. So we need to
                                        // replace [conv]'s output blob to [eltwise]'s one
                                        // considering that [activ] is an in-place layer.
                                        // Also we need to move all the consumers' references.
                                        // To prevent memory collisions (i.e. when input of
                                        // [conv] and output of [eltwise] is the same blob)
                                        // we allocate a new blob.
                                        CV_Assert_N(ld.outputBlobs.size() == 1, ld.outputBlobsWrappers.size() == 1);
                                        ld.outputBlobs[0] = ld.outputBlobs[0].clone();
                                        ld.outputBlobsWrappers[0] = wrap(ld.outputBlobs[0]);

                                        eltwiseData->outputBlobs = ld.outputBlobs;
                                        nextData->outputBlobs = ld.outputBlobs;
                                        eltwiseData->outputBlobsWrappers = ld.outputBlobsWrappers;
                                        nextData->outputBlobsWrappers = ld.outputBlobsWrappers;

                                        // Move references of [activ] layer consumers to the newly allocated blob.
                                        for (int i = 0; i < nextData->consumers.size(); ++i)
                                        {
                                            LayerData& consumer = layers[nextData->consumers[i].lid];
                                            for (int j = 0; j < consumer.inputBlobsId.size(); ++j)
                                            {
                                                if (consumer.inputBlobsId[j].lid == lpNext.lid)
                                                {
                                                    consumer.inputBlobs[j] = &ld.outputBlobs[0];
                                                    consumer.inputBlobsWrappers[j] = ld.outputBlobsWrappers[0];
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (preferableBackend != DNN_BACKEND_OPENCV)
                continue;  // Go to the next layer.

            // the optimization #2. if there is no layer that takes max pooling layer's computed
            // max indices (and only some semantical segmentation networks might need this;
            // many others only take the maximum values), then we switch the max pooling
            // layer to the faster operating mode.
            Ptr<PoolingLayer> poolingLayer = ld.layerInstance.dynamicCast<PoolingLayer>();
            if( !poolingLayer.empty() && !ld.consumers.empty() )
            {
                size_t i = 0, nconsumers = ld.consumers.size();
                for( ; i < nconsumers; i++ )
                    if( ld.consumers[i].oid > 0 )
                        break;
                // if there is no layer that takes the second output pin of the pooling layer
                // on input then we don't need to compute the indices
                if( i >= nconsumers )
                {
                    poolingLayer->computeMaxIdx = false;
                    printf_(("\tsimplified pooling layer %s\n", poolingLayer->name.c_str()));
                }
            }

            // the optimization #3. if there is concat layer that concatenates channels
            // from the inputs together (i.e. axis == 1) then we make the inputs of
            // the concat layer to write to the concatenation output buffer
            // (and so we eliminate the concatenation layer, because the channels
            // are concatenated implicitly).
            Ptr<ConcatLayer> concatLayer = ld.layerInstance.dynamicCast<ConcatLayer>();
            if( !concatLayer.empty() && concatLayer->axis == 1 && !concatLayer->padding &&
                ld.outputBlobs.size() == 1 )
            {
                Mat& output = ld.outputBlobs[0];
                UMat umat_output;
                if (!ld.outputBlobsWrappers.empty() &&
                    (preferableBackend == DNN_BACKEND_OPENCV && IS_DNN_OPENCL_TARGET(preferableTarget)))
                {
                    size_t i, ninputs = ld.inputBlobsId.size();
                    bool conv_layer = true;
                    for( i = 0; i < ninputs; i++ )
                    {
                        LayerPin pin = ld.inputBlobsId[i];
                        LayerData* inp_i_data = &layers[pin.lid];
                        while(inp_i_data->skip &&
                              inp_i_data->inputBlobsId.size() == 1 &&
                              inp_i_data->consumers.size() == 1)
                        {
                            pin = inp_i_data->inputBlobsId[0];
                            inp_i_data = &layers[pin.lid];
                        }
                        conv_layer = conv_layer && (inp_i_data->getLayerInstance()->type == "Convolution");
                    }
                    if (!conv_layer)
                        continue;
                    std::vector<UMat> umat_outputBlobs;
                    umat_outputBlobs = OpenCLBackendWrapper::getUMatVector(ld.outputBlobsWrappers);
                    umat_output = umat_outputBlobs[0];
                }

                // TODO: in general, this optimization can always be done, but
                // many layers currently check that the input/output blobs are
                // continuous arrays. Unfortunately, this is not true when
                // the concatenation optimization is applied with batch_size > 1.
                // so, for now, we only apply this optimization in the most popular
                // case batch_size == 1.
                if( output.dims == 4 && output.size[0] == 1 )
                {
                    size_t i, ninputs = ld.inputBlobsId.size();
                    std::vector<LayerPin> realinputs(ninputs);
                    for( i = 0; i < ninputs; i++ )
                    {
                        LayerPin pin = ld.inputBlobsId[i];
                        LayerData* inp_i_data = &layers[pin.lid];
                        while(inp_i_data->skip &&
                              inp_i_data->inputBlobsId.size() == 1 &&
                              inp_i_data->consumers.size() == 1)
                        {
                            pin = inp_i_data->inputBlobsId[0];
                            inp_i_data = &layers[pin.lid];
                        }
                        printf_(("\treal input for %s is %s\n",
                               layers[ld.inputBlobsId[i].lid].getLayerInstance()->name.c_str(),
                               inp_i_data->getLayerInstance()->name.c_str()));

                        if(inp_i_data->skip || inp_i_data->consumers.size() != 1)
                            break;
                        realinputs[i] = pin;
                    }

                    if( i >= ninputs )
                    {
                        // Allocate new memory to prevent collisions during memory
                        // reusing (see https://github.com/opencv/opencv/pull/10456).
                        output = output.clone();
                        if (preferableBackend == DNN_BACKEND_OPENCV &&
                            IS_DNN_OPENCL_TARGET(preferableTarget))
                        {
                            std::vector<UMat> umats(1);
                            umat_output = umat_output.clone();
                            umats[0] = umat_output;
                            OpenCLBackendWrapper::update(ld.outputBlobsWrappers, umats);
                        }
                        Range chrange[] = { Range::all(), Range::all(), Range::all(), Range::all() };
                        int ofs = 0;
                        for( i = 0; i < ninputs; i++ )
                        {
                            LayerPin pin = realinputs[i];
                            LayerData* inp_i_data = &layers[pin.lid];
                            int channels_i = ld.inputBlobs[i]->size[1];
                            chrange[1] = Range(ofs, ofs + channels_i);
                            printf_(("\toutput %s(%d) to channels (%d, %d)\n", inp_i_data->layerInstance->name.c_str(),
                                   pin.oid, ofs, ofs + channels_i));
                            ofs += channels_i;
                            Mat output_slice = output(chrange);
                            Mat& curr_output = inp_i_data->outputBlobs[pin.oid];
                            CV_Assert(output_slice.isContinuous() && output_slice.size == curr_output.size);
                            Mat* oldPtr = &curr_output;
                            curr_output = output_slice;
                            if (preferableBackend == DNN_BACKEND_OPENCV && IS_DNN_OPENCL_TARGET(preferableTarget))
                            {
                                std::vector<UMat> umats(inp_i_data->outputBlobsWrappers.size());
                                umats[pin.oid] = umat_output(chrange);
                                OpenCLBackendWrapper::update(inp_i_data->outputBlobsWrappers, umats);
                            }
                            // Layers that refer old input Mat will refer to the
                            // new data but the same Mat object.
                            CV_Assert_N(curr_output.data == output_slice.data, oldPtr == &curr_output);
                        }
                        ld.skip = true;
                        printf_(("\toptimized out Concat layer %s\n", concatLayer->name.c_str()));
                    }
                }
            }
        }
    }

    void allocateLayers(const std::vector<LayerPin>& blobsToKeep_)
    {
        CV_TRACE_FUNCTION();

        MapIdToLayerData::iterator it;
        for (it = layers.begin(); it != layers.end(); it++)
            it->second.flag = 0;

        CV_Assert(!layers[0].outputBlobs.empty());
        ShapesVec inputShapes;
        for(int i = 0; i < layers[0].outputBlobs.size(); i++)
        {
            Mat& inp = layers[0].outputBlobs[i];
            CV_Assert(inp.total());
            if (preferableBackend == DNN_BACKEND_OPENCV &&
                preferableTarget == DNN_TARGET_OPENCL_FP16)
            {
                layers[0].outputBlobs[i].create(inp.dims, inp.size, CV_16S);
            }
            inputShapes.push_back(shape(inp));
        }
        LayersShapesMap layersShapes;
        getLayersShapes(inputShapes, layersShapes);

        blobManager.reset();
        backendWrappers.clear();
        // Fake references to input blobs.
        for (int i = 0; i < layers[0].outputBlobs.size(); ++i)
            blobManager.addReference(LayerPin(0, i));
        for (it = layers.begin(); it != layers.end(); ++it)
        {
            const LayerData& ld = it->second;
            blobManager.addReferences(ld.inputBlobsId);
        }

        for (int i = 0; i < blobsToKeep_.size(); i++)
        {
            blobManager.addReference(blobsToKeep_[i]);
        }

        for (it = layers.begin(); it != layers.end(); it++)
        {
            int lid = it->first;
            allocateLayer(lid, layersShapes);
        }

        layersTimings.resize(lastLayerId + 1, 0);
        fuseLayers(blobsToKeep_);
    }

    void forwardLayer(LayerData &ld)
    {
        CV_TRACE_FUNCTION();

        Ptr<Layer> layer = ld.layerInstance;

        TickMeter tm;
        tm.start();

        if( !ld.skip )
        {
            std::map<int, Ptr<BackendNode> >::iterator it = ld.backendNodes.find(preferableBackend);
            if (preferableBackend == DNN_BACKEND_OPENCV || it == ld.backendNodes.end() || it->second.empty())
            {
                if (preferableBackend == DNN_BACKEND_OPENCV && IS_DNN_OPENCL_TARGET(preferableTarget))
                {
                    std::vector<UMat> umat_inputBlobs = OpenCLBackendWrapper::getUMatVector(ld.inputBlobsWrappers);
                    std::vector<UMat> umat_outputBlobs = OpenCLBackendWrapper::getUMatVector(ld.outputBlobsWrappers);
                    std::vector<UMat> umat_internalBlobs = OpenCLBackendWrapper::getUMatVector(ld.internalBlobsWrappers);
                    layer->forward(umat_inputBlobs,
                                   umat_outputBlobs,
                                   umat_internalBlobs);
                    if (DNN_CHECK_NAN_INF)
                    {
                        bool fail = false;
                        for (size_t i = 0; i < umat_outputBlobs.size(); ++i)
                        {
                            UMat& u = umat_outputBlobs[i];
                            Mat m;
                            if (u.depth() == CV_16S) // FP16
                                convertFp16(u, m);
                            else
                                m = u.getMat(ACCESS_READ);
                            if (!checkRange(m))
                            {
                                std::cerr << "WARNING: NaN detected in layer output: id=" << ld.id << " name=" << layer->name << std::endl;
                                std::cerr << "output id=" << i << " output shape=" << shape(m) << std::endl;
                                fail = true;
                            }
                            else if (!checkRange(m, true, NULL, -1e6, 1e6))
                            {
                                std::cerr << "WARNING: Inf detected in layer output: id=" << ld.id << " name=" << layer->name << std::endl;
                                std::cerr << "output id=" << i << " output shape=" << shape(m) << std::endl;
                                fail = true;
                            }
                        }
                        if (fail)
                        {
                            for (size_t i = 0; i < umat_inputBlobs.size(); ++i)
                            {
                                UMat& u = umat_inputBlobs[i];
                                Mat m;
                                if (u.depth() == CV_16S) // FP16
                                    convertFp16(u, m);
                                else
                                    m = u.getMat(ACCESS_READ);
                                std::cout << "INPUT " << i << " " << cv::typeToString(u.type()) << " " << shape(m) << std::endl;
                                if (DNN_CHECK_NAN_INF_DUMP) std::cout << m.reshape(1, 1) << std::endl;
                            }
                            for (size_t i = 0; i < umat_outputBlobs.size(); ++i)
                            {
                                UMat& u = umat_outputBlobs[i];
                                Mat m;
                                if (u.depth() == CV_16S) // FP16
                                    convertFp16(u, m);
                                else
                                    m = u.getMat(ACCESS_READ);
                                std::cout << "OUTPUT " << i << " " << cv::typeToString(u.type()) << " " << shape(m) << std::endl;
                                if (DNN_CHECK_NAN_INF_DUMP) std::cout << m.reshape(1, 1) << std::endl;
                            }
                            for (size_t i = 0; i < umat_internalBlobs.size(); ++i)
                            {
                                UMat& u = umat_internalBlobs[i];
                                Mat m;
                                if (u.depth() == CV_16S) // FP16
                                    convertFp16(u, m);
                                else
                                    m = u.getMat(ACCESS_READ);
                                std::cout << "INTERNAL " << i << " " << shape(m) << std::endl;
                                if (DNN_CHECK_NAN_INF_DUMP) std::cout << cv::typeToString(u.type()) << " " << m.reshape(1, 1) << std::endl;
                            }
                            if (DNN_CHECK_NAN_INF_RAISE_ERROR)
                                CV_Assert(!fail);
                        }
                    }
                    OpenCLBackendWrapper::update(ld.outputBlobsWrappers, umat_outputBlobs);
                }
                else
                {
                    for (int i = 0, n = ld.inputBlobsWrappers.size(); i < n; ++i)
                    {
                        if (!ld.inputBlobsWrappers[i].empty())
                            ld.inputBlobsWrappers[i]->copyToHost();
                    }

                    std::vector<Mat> inps(ld.inputBlobs.size());
                    for (int i = 0; i < ld.inputBlobs.size(); ++i)
                    {
                        inps[i] = *ld.inputBlobs[i];
                    }
                    layer->forward(inps, ld.outputBlobs, ld.internals);

                    if (DNN_CHECK_NAN_INF)
                    {
                        bool fail = false;
                        for (size_t i = 0; i < ld.outputBlobs.size(); ++i)
                        {
                            const Mat& m = ld.outputBlobs[i];
                            if (!checkRange(m))
                            {
                                std::cerr << "WARNING: NaN detected in layer output: id=" << ld.id << " name=" << layer->name << std::endl;
                                std::cerr << "output id=" << i << " output shape=" << shape(m) << std::endl;
                                fail = true;
                            }
                            else if (!checkRange(m, true, NULL, -1e6, 1e6))
                            {
                                std::cerr << "WARNING: Inf detected in layer output: id=" << ld.id << " name=" << layer->name << std::endl;
                                std::cerr << "output id=" << i << " output shape=" << shape(m) << std::endl;
                                fail = true;
                            }
                        }
                        if (fail)
                        {
                            for (size_t i = 0; i < ld.inputBlobs.size(); ++i)
                            {
                                const Mat* pM = ld.inputBlobs[i];
                                if (!pM)
                                {
                                    std::cout << "INPUT " << i << " is NULL" << std::endl;
                                    continue;
                                }
                                const Mat& m = *pM;
                                std::cout << "INPUT " << i << " " << cv::typeToString(m.type()) << " " << shape(m) << std::endl;
                                if (DNN_CHECK_NAN_INF_DUMP) std::cout << m.reshape(1, 1) << std::endl;
                            }
                            for (size_t i = 0; i < ld.outputBlobs.size(); ++i)
                            {
                                const Mat& m = ld.outputBlobs[i];
                                std::cout << "OUTPUT " << i << " " << cv::typeToString(m.type()) << " " << shape(m) << std::endl;
                                if (DNN_CHECK_NAN_INF_DUMP) std::cout << m.reshape(1, 1) << std::endl;
                            }
                            for (size_t i = 0; i < ld.internals.size(); ++i)
                            {
                                const Mat& m = ld.internals[i];
                                std::cout << "INTERNAL " << i << " " << cv::typeToString(m.type()) << " " << shape(m) << std::endl;
                                if (DNN_CHECK_NAN_INF_DUMP) std::cout << m.reshape(1, 1) << std::endl;
                            }
                            if (DNN_CHECK_NAN_INF_RAISE_ERROR)
                                CV_Assert(!fail);
                        }
                    }

                    for (int i = 0, n = ld.outputBlobsWrappers.size(); i < n; ++i)
                    {
                        if (!ld.outputBlobsWrappers[i].empty())
                            ld.outputBlobsWrappers[i]->setHostDirty();
                    }
                }
            }
            else
            {
                Ptr<BackendNode> node = it->second;
                CV_Assert(!node.empty());
                if (preferableBackend == DNN_BACKEND_HALIDE)
                {
                    forwardHalide(ld.outputBlobsWrappers, node);
                }
                else if (preferableBackend == DNN_BACKEND_INFERENCE_ENGINE)
                {
                    forwardInfEngine(node);
                }
                else if (preferableBackend == DNN_BACKEND_VKCOM)
                {
                    try
                    {
                        forwardVkCom(ld.outputBlobsWrappers, node);
                    }
                    catch (const cv::Exception& e)
                    {
                        CV_LOG_ERROR(NULL, "forwardVkCom failed, fallback to CPU implementation. " << e.what());
                        it->second = Ptr<BackendNode>();
                        forwardLayer(ld);
                    }
                }
                else
                {
                    CV_Error(Error::StsNotImplemented, "Unknown backend identifier");
                }
            }
        }
        else
            tm.reset();

        tm.stop();
        layersTimings[ld.id] = tm.getTimeTicks();

        ld.flag = 1;
    }

    void forwardToLayer(LayerData &ld, bool clearFlags = true)
    {
        CV_TRACE_FUNCTION();

        if (clearFlags)
        {
            MapIdToLayerData::iterator it;
            for (it = layers.begin(); it != layers.end(); it++)
                it->second.flag = 0;
        }

        //already was forwarded
        if (ld.flag)
            return;

        //forward parents
        MapIdToLayerData::iterator it;
        for (it = layers.begin(); it != layers.end() && (it->second.id < ld.id); ++it)
        {
            LayerData &ld = it->second;
            if (ld.flag)
                continue;
            forwardLayer(ld);
        }

        //forward itself
        forwardLayer(ld);
    }

    void forwardAll()
    {
        CV_TRACE_FUNCTION();

        MapIdToLayerData::reverse_iterator last_layer = layers.rbegin();
        CV_Assert(last_layer != layers.rend());
        forwardToLayer(last_layer->second, true);
    }

    void getLayerShapesRecursively(int id, LayersShapesMap& inOutShapes)
    {
        std::vector<LayerPin>& inputLayerIds = layers[id].inputBlobsId;

        if (inOutShapes[id].in.empty())
        {
            for(int i = 0; i < inputLayerIds.size(); i++)
            {
                int layerId = inputLayerIds[i].lid;
                LayersShapesMap::iterator it =
                        inOutShapes.find(layerId);
                if(it == inOutShapes.end() ||
                        it->second.out.empty())
                {
                    getLayerShapesRecursively(layerId, inOutShapes);
                }
                const MatShape& shape = inOutShapes[layerId].out[inputLayerIds[i].oid];
                inOutShapes[id].in.push_back(shape);
            }
        }
        const ShapesVec& is = inOutShapes[id].in;
        ShapesVec& os = inOutShapes[id].out;
        ShapesVec& ints = inOutShapes[id].internal;
        int requiredOutputs = layers[id].requiredOutputs.size();
        inOutShapes[id].supportInPlace =
                layers[id].getLayerInstance()->getMemoryShapes(is, requiredOutputs, os, ints);
    }

    void getLayersShapes(const ShapesVec& netInputShapes,
                         LayersShapesMap& inOutShapes)
    {
        inOutShapes.clear();

        inOutShapes[0].in = netInputShapes; //insert shape for first input layer
        for (MapIdToLayerData::iterator it = layers.begin();
             it != layers.end(); it++)
        {
            getLayerShapesRecursively(it->first, inOutShapes);
        }
    }

    void getLayerShapes(const ShapesVec& netInputShapes,
                        const int layerId,
                        LayerShapes& shapes)
    {
        LayersShapesMap inOutShapes;
        inOutShapes[0].in = netInputShapes; //insert shape for first input layer
        getLayerShapesRecursively(layerId, inOutShapes);
        shapes = inOutShapes[layerId];
    }

    LayerPin getLatestLayerPin(const std::vector<LayerPin>& pins)
    {
        return *std::max_element(pins.begin(), pins.end());
    }

    Mat getBlob(const LayerPin& pin)
    {
        CV_TRACE_FUNCTION();

        if (!pin.valid())
            CV_Error(Error::StsObjectNotFound, "Requested blob not found");

        LayerData &ld = layers[pin.lid];
        if ((size_t)pin.oid >= ld.outputBlobs.size())
        {
            CV_Error(Error::StsOutOfRange, format("Layer \"%s\" produce only %zu outputs, "
                                           "the #%d was requested", ld.name.c_str(),
                                           ld.outputBlobs.size(), pin.oid));
        }
        if (preferableTarget != DNN_TARGET_CPU)
        {
            CV_Assert(!ld.outputBlobsWrappers.empty() && !ld.outputBlobsWrappers[pin.oid].empty());
            // Transfer data to CPU if it's require.
            ld.outputBlobsWrappers[pin.oid]->copyToHost();
        }

        if (ld.outputBlobs[pin.oid].depth() == CV_16S)
        {
            convertFp16(ld.outputBlobs[pin.oid], output_blob);
            return output_blob;
        }
        else
            return ld.outputBlobs[pin.oid];
    }

    Mat getBlob(String outputName)
    {
        return getBlob(getPinByAlias(outputName));
    }
};

Net::Net() : impl(new Net::Impl)
{
}

Net Net::readFromModelOptimizer(const String& xml, const String& bin)
{
#ifndef HAVE_INF_ENGINE
    CV_Error(Error::StsError, "Build OpenCV with Inference Engine to enable loading models from Model Optimizer.");
#else
    InferenceEngine::CNNNetReader reader;
    reader.ReadNetwork(xml);
    reader.ReadWeights(bin);

    InferenceEngine::CNNNetwork ieNet = reader.getNetwork();

    std::vector<String> inputsNames;
    for (auto& it : ieNet.getInputsInfo())
    {
        inputsNames.push_back(it.first);
    }

    Net cvNet;
    cvNet.setInputsNames(inputsNames);

    Ptr<InfEngineBackendNode> backendNode(new InfEngineBackendNode(0));
    backendNode->net = Ptr<InfEngineBackendNet>(new InfEngineBackendNet(ieNet));
    for (auto& it : ieNet.getOutputsInfo())
    {
        Ptr<Layer> cvLayer(new InfEngineBackendLayer(it.second));
        InferenceEngine::CNNLayerPtr ieLayer = ieNet.getLayerByName(it.first.c_str());
        CV_Assert(ieLayer);

        LayerParams lp;
        int lid = cvNet.addLayer(it.first, "", lp);

        LayerData& ld = cvNet.impl->layers[lid];
        cvLayer->name = it.first;
        cvLayer->type = ieLayer->type;
        ld.layerInstance = cvLayer;
        ld.backendNodes[DNN_BACKEND_INFERENCE_ENGINE] = backendNode;

        for (int i = 0; i < inputsNames.size(); ++i)
            cvNet.connect(0, i, lid, i);
    }
    cvNet.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);

    cvNet.impl->skipInfEngineInit = true;
    return cvNet;
#endif  // HAVE_INF_ENGINE
}

Net::~Net()
{
}

int Net::addLayer(const String &name, const String &type, LayerParams &params)
{
    CV_TRACE_FUNCTION();

    if (impl->getLayerId(name) >= 0)
    {
        CV_Error(Error::StsBadArg, "Layer \"" + name + "\" already into net");
        return -1;
    }

    int id = ++impl->lastLayerId;
    impl->layerNameToId.insert(std::make_pair(name, id));
    impl->layers.insert(std::make_pair(id, LayerData(id, name, type, params)));

    return id;
}

int Net::addLayerToPrev(const String &name, const String &type, LayerParams &params)
{
    CV_TRACE_FUNCTION();

    int prvLid = impl->lastLayerId;
    int newLid = this->addLayer(name, type, params);
    this->connect(prvLid, 0, newLid, 0);
    return newLid;
}

void Net::connect(int outLayerId, int outNum, int inpLayerId, int inpNum)
{
    CV_TRACE_FUNCTION();

    impl->connect(outLayerId, outNum, inpLayerId, inpNum);
}

void Net::connect(String _outPin, String _inPin)
{
    CV_TRACE_FUNCTION();

    LayerPin outPin = impl->getPinByAlias(_outPin);
    LayerPin inpPin = impl->getPinByAlias(_inPin);

    CV_Assert(outPin.valid() && inpPin.valid());

    impl->connect(outPin.lid, outPin.oid, inpPin.lid, inpPin.oid);
}

Mat Net::forward(const String& outputName)
{
    CV_TRACE_FUNCTION();

    String layerName = outputName;

    if (layerName.empty())
        layerName = getLayerNames().back();

    std::vector<LayerPin> pins(1, impl->getPinByAlias(layerName));
    impl->setUpNet(pins);
    impl->forwardToLayer(impl->getLayerData(layerName));

    return impl->getBlob(layerName);
}

void Net::forward(OutputArrayOfArrays outputBlobs, const String& outputName)
{
    CV_TRACE_FUNCTION();

    String layerName = outputName;

    if (layerName.empty())
        layerName = getLayerNames().back();

    std::vector<LayerPin> pins(1, impl->getPinByAlias(layerName));
    impl->setUpNet(pins);
    impl->forwardToLayer(impl->getLayerData(layerName));

    LayerPin pin = impl->getPinByAlias(layerName);
    LayerData &ld = impl->layers[pin.lid];

    if (outputBlobs.isUMat())
    {
        impl->getBlob(layerName).copyTo(outputBlobs);
    }
    else if (outputBlobs.isMat())
    {
        outputBlobs.assign(impl->getBlob(layerName));
    }
    else if (outputBlobs.isMatVector())
    {
        if (impl->preferableTarget != DNN_TARGET_CPU)
        {
            for (int i = 0; i < ld.outputBlobsWrappers.size(); ++i)
            {
                CV_Assert(!ld.outputBlobsWrappers[i].empty());
                ld.outputBlobsWrappers[i]->copyToHost();
            }
        }
        if (ld.outputBlobs[0].depth() == CV_32F)
        {
            std::vector<Mat> & outputvec = *(std::vector<Mat> *)outputBlobs.getObj();
            outputvec = ld.outputBlobs;
        } else {
            std::vector<Mat> & outputvec = *(std::vector<Mat> *)outputBlobs.getObj();
            outputvec.resize(ld.outputBlobs.size());
            for (int i = 0; i < outputvec.size(); i++)
                convertFp16(ld.outputBlobs[i], outputvec[i]);
        }
    }
    else if (outputBlobs.isUMatVector())
    {
        std::vector<UMat> & outputvec = *(std::vector<UMat> *)outputBlobs.getObj();

        if (impl->preferableBackend == DNN_BACKEND_OPENCV &&
            IS_DNN_OPENCL_TARGET(impl->preferableTarget))
        {
            if (impl->preferableTarget == DNN_TARGET_OPENCL)
                outputvec = OpenCLBackendWrapper::getUMatVector(ld.outputBlobsWrappers);
            else if (impl->preferableTarget == DNN_TARGET_OPENCL_FP16)
            {
                std::vector<UMat> out_vec = OpenCLBackendWrapper::getUMatVector(ld.outputBlobsWrappers);
                outputvec.resize(out_vec.size());
                for (int i = 0; i < out_vec.size(); i++)
                    convertFp16(out_vec[i], outputvec[i]);
            }
        }
        else
        {
            outputvec.resize(ld.outputBlobs.size());
            for (int i = 0; i < outputvec.size(); ++i)
                ld.outputBlobs[i].copyTo(outputvec[i]);
        }
    }
}

void Net::forward(OutputArrayOfArrays outputBlobs,
                  const std::vector<String>& outBlobNames)
{
    CV_TRACE_FUNCTION();

    std::vector<LayerPin> pins;
    for (int i = 0; i < outBlobNames.size(); i++)
    {
        pins.push_back(impl->getPinByAlias(outBlobNames[i]));
    }

    impl->setUpNet(pins);

    LayerPin out = impl->getLatestLayerPin(pins);

    impl->forwardToLayer(impl->getLayerData(out.lid));

    std::vector<Mat> matvec;
    for (int i = 0; i < pins.size(); i++)
    {
        matvec.push_back(impl->getBlob(pins[i]));
    }

    std::vector<Mat> & outputvec = *(std::vector<Mat> *)outputBlobs.getObj();
    outputvec = matvec;
}

void Net::forward(std::vector<std::vector<Mat> >& outputBlobs,
                     const std::vector<String>& outBlobNames)
{
    CV_TRACE_FUNCTION();

    std::vector<LayerPin> pins;
    for (int i = 0; i < outBlobNames.size(); i++)
    {
        std::vector<LayerPin> lp = impl->getLayerOutPins(outBlobNames[i]);
        pins.insert(pins.end(), lp.begin(), lp.end());
    }

    impl->setUpNet(pins);

    LayerPin out = impl->getLatestLayerPin(pins);

    impl->forwardToLayer(impl->getLayerData(out.lid));

    outputBlobs.resize(outBlobNames.size());
    for (int i = 0; i < outBlobNames.size(); i++)
    {
        std::vector<LayerPin> lp = impl->getLayerOutPins(outBlobNames[i]);
        for (int i = 0; i < lp.size(); i++)
        {
            outputBlobs[i].push_back(impl->getBlob(lp[i]));
        }
    }
}

void Net::setPreferableBackend(int backendId)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG(backendId);

    if( impl->preferableBackend != backendId )
    {
        impl->preferableBackend = backendId;
        impl->netWasAllocated = false;
        impl->clear();
    }
}

void Net::setPreferableTarget(int targetId)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG(targetId);

    if( impl->preferableTarget != targetId )
    {
        impl->preferableTarget = targetId;
        if (IS_DNN_OPENCL_TARGET(targetId))
        {
#ifndef HAVE_OPENCL
#ifdef HAVE_INF_ENGINE
            if (impl->preferableBackend == DNN_BACKEND_OPENCV)
#else
            if (impl->preferableBackend == DNN_BACKEND_DEFAULT ||
                impl->preferableBackend == DNN_BACKEND_OPENCV)
#endif  // HAVE_INF_ENGINE
                impl->preferableTarget = DNN_TARGET_CPU;
#else
            bool fp16 = ocl::Device::getDefault().isExtensionSupported("cl_khr_fp16");
            if (!fp16 && targetId == DNN_TARGET_OPENCL_FP16)
                impl->preferableTarget = DNN_TARGET_OPENCL;
#endif
        }
        impl->netWasAllocated = false;
        impl->clear();
    }
}

void Net::setInputsNames(const std::vector<String> &inputBlobNames)
{
    CV_TRACE_FUNCTION();

    impl->netInputLayer->setNames(inputBlobNames);
}

void Net::setInput(InputArray blob, const String& name, double scalefactor, const Scalar& mean)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG_VALUE(name, "name", name.c_str());

    LayerPin pin;
    pin.lid = 0;
    pin.oid = impl->resolvePinOutputName(impl->getLayerData(pin.lid), name);

    if (!pin.valid())
        CV_Error(Error::StsObjectNotFound, "Requested blob \"" + name + "\" not found");

    LayerData &ld = impl->layers[pin.lid];
    const int numInputs = std::max(pin.oid+1, (int)ld.requiredOutputs.size());
    ld.outputBlobs.resize(numInputs);
    ld.outputBlobsWrappers.resize(numInputs);
    impl->netInputLayer->inputsData.resize(numInputs);
    impl->netInputLayer->scaleFactors.resize(numInputs);
    impl->netInputLayer->means.resize(numInputs);

    MatShape prevShape = shape(impl->netInputLayer->inputsData[pin.oid]);
    Mat blob_ = blob.getMat();
    bool oldShape = prevShape == shape(blob_);
    if (oldShape)
    {
        blob_.copyTo(impl->netInputLayer->inputsData[pin.oid]);
    }
    else
    {
        ld.outputBlobs[pin.oid] = blob_.clone();
        impl->netInputLayer->inputsData[pin.oid] = ld.outputBlobs[pin.oid];
    }

    if (!ld.outputBlobsWrappers[pin.oid].empty())
    {
        ld.outputBlobsWrappers[pin.oid]->setHostDirty();
    }
    impl->netInputLayer->scaleFactors[pin.oid] = scalefactor;
    impl->netInputLayer->means[pin.oid] = mean;
    impl->netWasAllocated = impl->netWasAllocated && oldShape;
}

Mat Net::getParam(LayerId layer, int numParam)
{
    LayerData &ld = impl->getLayerData(layer);
    std::vector<Mat> &layerBlobs = ld.getLayerInstance()->blobs;
    CV_Assert(numParam < (int)layerBlobs.size());
    return layerBlobs[numParam];
}

void Net::setParam(LayerId layer, int numParam, const Mat &blob)
{
    LayerData &ld = impl->getLayerData(layer);

    std::vector<Mat> &layerBlobs = ld.getLayerInstance()->blobs;
    CV_Assert(numParam < (int)layerBlobs.size());
    //we don't make strong checks, use this function carefully
    layerBlobs[numParam] = blob;
}

int Net::getLayerId(const String &layer)
{
    return impl->getLayerId(layer);
}

Ptr<Layer> Net::getLayer(LayerId layerId)
{
    LayerData &ld = impl->getLayerData(layerId);
    return ld.getLayerInstance();
}

std::vector<Ptr<Layer> > Net::getLayerInputs(LayerId layerId)
{
    LayerData &ld = impl->getLayerData(layerId);
    if (!ld.layerInstance)
        CV_Error(Error::StsNullPtr, format("Requested layer \"%s\" was not initialized", ld.name.c_str()));

    std::vector<Ptr<Layer> > inputLayers;
    inputLayers.reserve(ld.inputLayersId.size());
    std::set<int>::iterator it;
    for (it = ld.inputLayersId.begin(); it != ld.inputLayersId.end(); ++it) {
        inputLayers.push_back(getLayer(*it));
    }
    return inputLayers;
}

std::vector<String> Net::getLayerNames() const
{
    std::vector<String> res;
    res.reserve(impl->layers.size());

    Impl::MapIdToLayerData::iterator it;
    for (it = impl->layers.begin(); it != impl->layers.end(); it++)
    {
        if (it->second.id) //skip Data layer
            res.push_back(it->second.name);
    }

    return res;
}

bool Net::empty() const
{
    return impl->layers.size() <= 1; //first layer is default Data layer
}

std::vector<int> Net::getUnconnectedOutLayers() const
{
    std::vector<int> layersIds;

    Impl::MapIdToLayerData::iterator it;
    for (it = impl->layers.begin(); it != impl->layers.end(); it++)
    {
        int lid = it->first;
        LayerData &ld = it->second;

        if (ld.requiredOutputs.size() == 0)
            layersIds.push_back(lid);
    }

    return layersIds;
}

std::vector<String> Net::getUnconnectedOutLayersNames() const
{
    std::vector<int> ids = getUnconnectedOutLayers();
    const size_t n = ids.size();
    std::vector<String> names(n);
    for (size_t i = 0; i < n; ++i)
    {
        names[i] = impl->layers[ids[i]].name;
    }
    return names;
}

void Net::getLayersShapes(const ShapesVec& netInputShapes,
                          std::vector<int>& layersIds,
                          std::vector<ShapesVec>& inLayersShapes,
                          std::vector<ShapesVec>& outLayersShapes) const
{
    layersIds.clear();
    inLayersShapes.clear();
    outLayersShapes.clear();

    Impl::LayersShapesMap inOutShapes;
    impl->getLayersShapes(netInputShapes, inOutShapes);

    for(Impl::LayersShapesMap::const_iterator it = inOutShapes.begin();
        it != inOutShapes.end(); it++)
    {
        layersIds.push_back(it->first);
        inLayersShapes.push_back(it->second.in);
        outLayersShapes.push_back(it->second.out);
    }
}

void Net::getLayersShapes(const MatShape& netInputShape,
                          std::vector<int>& layerIds,
                          std::vector<ShapesVec>& inLayersShapes,
                          std::vector<ShapesVec>& outLayersShapes) const
{
    getLayersShapes(ShapesVec(1, netInputShape),
                    layerIds, inLayersShapes, outLayersShapes);
}

void Net::getLayerShapes(const MatShape& netInputShape,
                         const int layerId,
                         ShapesVec& inLayerShapes,
                         ShapesVec& outLayerShapes) const
{
    getLayerShapes(ShapesVec(1, netInputShape),
                   layerId, inLayerShapes, outLayerShapes);

}

void Net::getLayerShapes(const ShapesVec& netInputShapes,
                    const int layerId,
                    ShapesVec& inLayerShapes,
                    ShapesVec& outLayerShapes) const
{
    LayerShapes shapes;
    impl->getLayerShapes(netInputShapes, layerId, shapes);
    inLayerShapes = shapes.in;
    outLayerShapes = shapes.out;
}

int64 Net::getFLOPS(const std::vector<MatShape>& netInputShapes) const
{
    CV_TRACE_FUNCTION();

    int64 flops = 0;
    std::vector<int> ids;
    std::vector<std::vector<MatShape> > inShapes, outShapes;
    getLayersShapes(netInputShapes, ids, inShapes, outShapes);
    CV_Assert(inShapes.size() == outShapes.size());
    CV_Assert(inShapes.size() == ids.size());

    for(int i = 0; i < ids.size(); i++)
    {
        flops += impl->layers[ids[i]].getLayerInstance()->getFLOPS(inShapes[i],
                                                                   outShapes[i]);
    }

    return flops;
}

int64 Net::getFLOPS(const MatShape& netInputShape) const
{
    return getFLOPS(std::vector<MatShape>(1, netInputShape));
}

int64 Net::getFLOPS(const int layerId,
              const std::vector<MatShape>& netInputShapes) const
{
    Impl::MapIdToLayerData::iterator layer = impl->layers.find(layerId);
    CV_Assert(layer != impl->layers.end());

    LayerShapes shapes;
    impl->getLayerShapes(netInputShapes, layerId, shapes);

    return layer->second.getLayerInstance()->getFLOPS(shapes.in, shapes.out);
}

int64 Net::getFLOPS(const int layerId,
              const MatShape& netInputShape) const
{
    return getFLOPS(layerId, std::vector<MatShape>(1, netInputShape));
}

void Net::getLayerTypes(std::vector<String>& layersTypes) const
{
    layersTypes.clear();

    std::map<String, int> layers;
    for (Impl::MapIdToLayerData::iterator it = impl->layers.begin();
         it != impl->layers.end(); it++)
    {
        if (layers.find(it->second.type) == layers.end())
            layers[it->second.type] = 0;
        layers[it->second.type]++;
    }

    for (std::map<String, int>::iterator it = layers.begin();
         it != layers.end(); it++)
    {
        layersTypes.push_back(it->first);
    }
}

int Net::getLayersCount(const String& layerType) const
{
    int count = 0;
    for (Impl::MapIdToLayerData::iterator it = impl->layers.begin();
         it != impl->layers.end(); it++)
    {
        if (it->second.type == layerType)
            count++;
    }
    return count;
}

void Net::getMemoryConsumption(const int layerId,
                               const std::vector<MatShape>& netInputShapes,
                               size_t& weights, size_t& blobs) const
{
    CV_TRACE_FUNCTION();

    Impl::MapIdToLayerData::iterator layer = impl->layers.find(layerId);
    CV_Assert(layer != impl->layers.end());

    weights = blobs = 0;

    for(int i = 0; i < layer->second.params.blobs.size(); i++)
    {
        const Mat& weightsBlob = layer->second.params.blobs[i];
        weights += weightsBlob.total()*weightsBlob.elemSize();
    }

    ShapesVec inLayerShapes, outLayerShapes;
    getLayerShapes(netInputShapes, layerId, inLayerShapes, outLayerShapes);
    for(int i = 0; i < outLayerShapes.size(); i++)
    {
        blobs += total(outLayerShapes[i]) * sizeof(float);
    }
}

void Net::getMemoryConsumption(const std::vector<MatShape>& netInputShapes,
                               size_t& weights, size_t& blobs) const
{
    CV_TRACE_FUNCTION();

    std::vector<int> layerIds;
    std::vector<size_t> w, b;
    getMemoryConsumption(netInputShapes, layerIds, w, b);

    weights = blobs = 0;
    for(int i = 0; i < layerIds.size(); i++)
    {
        weights += w[i];
        blobs += b[i];
    }
}

void Net::getMemoryConsumption(const int layerId,
                               const MatShape& netInputShape,
                               size_t& weights, size_t& blobs) const
{
    getMemoryConsumption(layerId, std::vector<MatShape>(1, netInputShape),
                         weights, blobs);
}

void Net::getMemoryConsumption(const MatShape& netInputShape,
                               size_t& weights, size_t& blobs) const
{
    getMemoryConsumption(std::vector<MatShape>(1, netInputShape),
                         weights, blobs);
}

void Net::getMemoryConsumption(const std::vector<MatShape>& netInputShapes,
                                  std::vector<int>& layerIds, std::vector<size_t>& weights,
                                  std::vector<size_t>& blobs) const
{
    CV_TRACE_FUNCTION();

    layerIds.clear();
    weights.clear();
    blobs.clear();

    std::vector<std::vector<MatShape> > inLayerShapes, outLayerShapes;

    getLayersShapes(netInputShapes, layerIds, inLayerShapes, outLayerShapes);

    for(int i = 0; i < layerIds.size(); i++)
    {
        int w = 0, b = 0;
        Impl::MapIdToLayerData::iterator layer = impl->layers.find(layerIds[i]);
        CV_Assert(layer != impl->layers.end());

        for(int j = 0; j < layer->second.params.blobs.size(); j++)
        {
            const Mat& weightsBlob = layer->second.params.blobs[j];
            w += weightsBlob.total()*weightsBlob.elemSize();
        }

        for(int j = 0; j < outLayerShapes[i].size(); j++)
        {
            b += total(outLayerShapes[i][j]) * sizeof(float);
        }

        weights.push_back(w);
        blobs.push_back(b);
    }
}

void Net::getMemoryConsumption(const MatShape& netInputShape, std::vector<int>& layerIds,
                               std::vector<size_t>& weights, std::vector<size_t>& blobs) const
{
    getMemoryConsumption(std::vector<MatShape>(1, netInputShape), layerIds,
                         weights, blobs);
}

void Net::enableFusion(bool fusion)
{
    if( impl->fusion != fusion )
    {
        impl->fusion = fusion;
        impl->netWasAllocated = false;
        impl->clear();
    }
}

void Net::setHalideScheduler(const String& scheduler)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG_VALUE(scheduler, "scheduler", scheduler.c_str());

    impl->halideConfigFile = scheduler;
}

int64 Net::getPerfProfile(std::vector<double>& timings)
{
    timings = std::vector<double>(impl->layersTimings.begin() + 1, impl->layersTimings.end());
    int64 total = (int64)std::accumulate(timings.begin(), timings.end(), 0.0);
    return total;
}

//////////////////////////////////////////////////////////////////////////

Layer::Layer() { preferableTarget = DNN_TARGET_CPU; }

Layer::Layer(const LayerParams &params)
    : blobs(params.blobs), name(params.name), type(params.type)
{
    preferableTarget = DNN_TARGET_CPU;
}

void Layer::setParamsFrom(const LayerParams &params)
{
    blobs = params.blobs;
    name = params.name;
    type = params.type;
}

int Layer::inputNameToIndex(String)
{
    return -1;
}

int Layer::outputNameToIndex(const String&)
{
    return 0;
}

bool Layer::supportBackend(int backendId)
{
    return backendId == DNN_BACKEND_OPENCV;
}

Ptr<BackendNode> Layer::initVkCom(const std::vector<Ptr<BackendWrapper> > &)
{
    CV_Error(Error::StsNotImplemented, "VkCom pipeline of " + type +
                                       " layers is not defined.");
    return Ptr<BackendNode>();
}

Ptr<BackendNode> Layer::initHalide(const std::vector<Ptr<BackendWrapper> > &)
{
    CV_Error(Error::StsNotImplemented, "Halide pipeline of " + type +
                                       " layers is not defined.");
    return Ptr<BackendNode>();
}

Ptr<BackendNode> Layer::initInfEngine(const std::vector<Ptr<BackendWrapper> > &)
{
    CV_Error(Error::StsNotImplemented, "Inference Engine pipeline of " + type +
                                       " layers is not defined.");
    return Ptr<BackendNode>();
}

void Layer::applyHalideScheduler(Ptr<BackendNode>& node, const std::vector<Mat*> &inputs,
                                 const std::vector<Mat> &outputs, int targetId) const
{
#ifdef  HAVE_HALIDE
    CV_TRACE_FUNCTION();

    Halide::Var x("x"), y("y"), c("c"), n("n"), co("co"), ci("ci"),
                xo("xo"), xi("xi"), yo("yo"), yi("yi"), tile("tile");
    Halide::Func& top = node.dynamicCast<HalideBackendNode>()->funcs.back();

    int outW, outH, outC, outN;
    getCanonicalSize(outputs[0].size, &outW, &outH, &outC, &outN);

    if (targetId == DNN_TARGET_CPU)
    {
        if (outW == 1 && outH == 1)
        {
            if (outC + outN == 1)
                return;

            if (outC > 8)
              top.split(c, co, ci, 8)
                 .fuse(x, y, tile).fuse(co, tile, tile).fuse(n, tile, tile)
                 .parallel(tile)
                 .vectorize(ci, 8);
            else
              top.fuse(x, y, tile).fuse(c, tile, tile).fuse(n, tile, tile)
                 .parallel(tile);
        }
        else
        {
            if (outH > 2)
            {
                top.reorder(x, c, y)
                   .split(y, yo, yi, 2)
                   .fuse(yo, n, tile)
                   .parallel(tile)
                   .unroll(yi)
                   .vectorize(x, outW >= 16 ? 16 : outW);
            }
        }
    }
    else if (targetId == DNN_TARGET_OPENCL)
    {
        if (outW == 1 && outH == 1)
        {
            int c_split = outC > 8 ? (outC > 16 ? 8 : 4) : outC;
            top.split(c, co, ci, c_split)
               .fuse(x, y, tile).fuse(co, tile, tile).fuse(n, tile, tile)
               .gpu_blocks(tile)
               .gpu_threads(ci);
        }
        else
        {
            int x_split = outW > 8 ? (outW >= 32 ? 16 : 8) : outW;
            int y_split = outH > 8 ? (outH >= 32 ? 16 : 8) : outH;
            // Supported vectorization widths: 2, 3, 4, 8, 16
            int c_split = outC > 8 ? (outC > 16 ? 8 : 4) : std::min(4, outC);
            top.split(x, xo, xi, x_split).split(y, yo, yi, y_split)
               .split(c, co, ci, c_split)
               .gpu_blocks(xo, yo, co)
               .gpu_threads(xi, yi)
               .reorder(xi, yi, ci, xo, yo, co)
               .vectorize(ci);
        }
    }
    else
        CV_Error(Error::StsNotImplemented, "Unknown target identifier");
#endif  // HAVE_HALIDE
}

Ptr<BackendNode> Layer::tryAttach(const Ptr<BackendNode>& node)
{
    return Ptr<BackendNode>();
}

bool Layer::setActivation(const Ptr<ActivationLayer>&) { return false; }
bool Layer::tryFuse(Ptr<Layer>&) { return false; }
void Layer::getScaleShift(Mat& scale, Mat& shift) const
{
    scale = Mat();
    shift = Mat();
}

void Layer::unsetAttached()
{
    setActivation(Ptr<ActivationLayer>());
}

template <typename T>
static void vecToPVec(const std::vector<T> &v, std::vector<T*> &pv)
{
    pv.resize(v.size());
    for (size_t i = 0; i < v.size(); i++)
        pv[i] = const_cast<T*>(&v[i]);
}

void Layer::finalize(const std::vector<Mat> &inputs, std::vector<Mat> &outputs)
{
    CV_TRACE_FUNCTION();
    this->finalize((InputArrayOfArrays)inputs, (OutputArrayOfArrays)outputs);
}

void Layer::finalize(const std::vector<Mat*> &input, std::vector<Mat> &output)
{
    CV_UNUSED(input);CV_UNUSED(output);
}

void Layer::finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr)
{
    CV_TRACE_FUNCTION();
    std::vector<Mat> inputs, outputs;
    inputs_arr.getMatVector(inputs);
    outputs_arr.getMatVector(outputs);

    std::vector<Mat*> inputsp;
    vecToPVec(inputs, inputsp);
    this->finalize(inputsp, outputs);
}

std::vector<Mat> Layer::finalize(const std::vector<Mat> &inputs)
{
    CV_TRACE_FUNCTION();

    std::vector<Mat> outputs;
    this->finalize(inputs, outputs);
    return outputs;
}

void Layer::forward(std::vector<Mat*> &input, std::vector<Mat> &output, std::vector<Mat> &internals)
{
    // We kept this method for compatibility. DNN calls it now only to support users' implementations.
}

void Layer::forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG_VALUE(name, "name", name.c_str());

    Layer::forward_fallback(inputs_arr, outputs_arr, internals_arr);
}

void Layer::forward_fallback(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG_VALUE(name, "name", name.c_str());

    if (preferableTarget == DNN_TARGET_OPENCL_FP16 && inputs_arr.depth() == CV_16S)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;
        std::vector<UMat> internals;

        std::vector<UMat> orig_inputs;
        std::vector<UMat> orig_outputs;
        std::vector<UMat> orig_internals;

        inputs_arr.getUMatVector(orig_inputs);
        outputs_arr.getUMatVector(orig_outputs);
        internals_arr.getUMatVector(orig_internals);

        inputs.resize(orig_inputs.size());
        for (size_t i = 0; i < orig_inputs.size(); i++)
            convertFp16(orig_inputs[i], inputs[i]);

        outputs.resize(orig_outputs.size());
        for (size_t i = 0; i < orig_outputs.size(); i++)
            outputs[i].create(shape(orig_outputs[i]), CV_32F);

        internals.resize(orig_internals.size());
        for (size_t i = 0; i < orig_internals.size(); i++)
            internals[i].create(shape(orig_internals[i]), CV_32F);

        forward(inputs, outputs, internals);

        for (size_t i = 0; i < outputs.size(); i++)
            convertFp16(outputs[i], orig_outputs[i]);

        // sync results back
        outputs_arr.assign(orig_outputs);
        internals_arr.assign(orig_internals);
        return;
    }
    std::vector<Mat> inpvec;
    std::vector<Mat> outputs;
    std::vector<Mat> internals;

    inputs_arr.getMatVector(inpvec);
    outputs_arr.getMatVector(outputs);
    internals_arr.getMatVector(internals);

    std::vector<Mat*> inputs(inpvec.size());
    for (int i = 0; i < inpvec.size(); i++)
        inputs[i] = &inpvec[i];

    this->forward(inputs, outputs, internals);

    // sync results back
    outputs_arr.assign(outputs);
    internals_arr.assign(internals);
}

void Layer::run(const std::vector<Mat> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
{
    CV_TRACE_FUNCTION();

    this->finalize(inputs, outputs);
    this->forward(inputs, outputs, internals);
}

Layer::~Layer() {}

bool Layer::getMemoryShapes(const std::vector<MatShape> &inputs,
                            const int requiredOutputs,
                            std::vector<MatShape> &outputs,
                            std::vector<MatShape> &internals) const
{
    CV_Assert(inputs.size());
    outputs.assign(std::max(requiredOutputs, (int)inputs.size()), inputs[0]);
    return false;
}

//////////////////////////////////////////////////////////////////////////

static Mutex& getLayerFactoryMutex()
{
    static Mutex* volatile instance = NULL;
    if (instance == NULL)
    {
        cv::AutoLock lock(getInitializationMutex());
        if (instance == NULL)
            instance = new Mutex();
    }
    return *instance;
}

typedef std::map<String, std::vector<LayerFactory::Constructor> > LayerFactory_Impl;

static LayerFactory_Impl& getLayerFactoryImpl_()
{
    static LayerFactory_Impl impl;
    return impl;
}

static LayerFactory_Impl& getLayerFactoryImpl()
{
    static LayerFactory_Impl* volatile instance = NULL;
    if (instance == NULL)
    {
        cv::AutoLock lock(getLayerFactoryMutex());
        if (instance == NULL)
        {
            instance = &getLayerFactoryImpl_();
            initializeLayerFactory();
        }
    }
    return *instance;
}

void LayerFactory::registerLayer(const String &type, Constructor constructor)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG_VALUE(type, "type", type.c_str());

    cv::AutoLock lock(getLayerFactoryMutex());
    String type_ = toLowerCase(type);
    LayerFactory_Impl::iterator it = getLayerFactoryImpl().find(type_);

    if (it != getLayerFactoryImpl().end())
    {
        if (it->second.back() == constructor)
            CV_Error(cv::Error::StsBadArg, "Layer \"" + type_ + "\" already was registered");
        it->second.push_back(constructor);
    }
    getLayerFactoryImpl().insert(std::make_pair(type_, std::vector<Constructor>(1, constructor)));
}

void LayerFactory::unregisterLayer(const String &type)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG_VALUE(type, "type", type.c_str());

    cv::AutoLock lock(getLayerFactoryMutex());
    String type_ = toLowerCase(type);

    LayerFactory_Impl::iterator it = getLayerFactoryImpl().find(type_);
    if (it != getLayerFactoryImpl().end())
    {
        if (it->second.size() > 1)
            it->second.pop_back();
        else
            getLayerFactoryImpl().erase(it);
    }
}

Ptr<Layer> LayerFactory::createLayerInstance(const String &type, LayerParams& params)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG_VALUE(type, "type", type.c_str());

    cv::AutoLock lock(getLayerFactoryMutex());
    String type_ = toLowerCase(type);
    LayerFactory_Impl::const_iterator it = getLayerFactoryImpl().find(type_);

    if (it != getLayerFactoryImpl().end())
    {
        CV_Assert(!it->second.empty());
        return it->second.back()(params);
    }
    else
    {
        return Ptr<Layer>(); //NULL
    }
}

BackendNode::BackendNode(int backendId) : backendId(backendId) {}

BackendNode::~BackendNode() {};

BackendWrapper::BackendWrapper(int backendId, int targetId)
    : backendId(backendId), targetId(targetId) {}

BackendWrapper::BackendWrapper(int targetId, const cv::Mat& m)
{
    CV_Error(Error::StsNotImplemented,
             "Constructor of backend wrapper must be implemented");
}

BackendWrapper::BackendWrapper(const Ptr<BackendWrapper>& base, const MatShape& shape)
{
    CV_Error(Error::StsNotImplemented,
             "Constructor of backend wrapper must be implemented");
}

BackendWrapper::~BackendWrapper() {}

Net readNet(const String& _model, const String& _config, const String& _framework)
{
    String framework = toLowerCase(_framework);
    String model = _model;
    String config = _config;
    const std::string modelExt = model.substr(model.rfind('.') + 1);
    const std::string configExt = config.substr(config.rfind('.') + 1);
    if (framework == "caffe" || modelExt == "caffemodel" || configExt == "caffemodel" ||
                                modelExt == "prototxt" || configExt == "prototxt")
    {
        if (modelExt == "prototxt" || configExt == "caffemodel")
            std::swap(model, config);
        return readNetFromCaffe(config, model);
    }
    if (framework == "tensorflow" || modelExt == "pb" || configExt == "pb" ||
                                     modelExt == "pbtxt" || configExt == "pbtxt")
    {
        if (modelExt == "pbtxt" || configExt == "pb")
            std::swap(model, config);
        return readNetFromTensorflow(model, config);
    }
    if (framework == "torch" || modelExt == "t7" || modelExt == "net" ||
                                configExt == "t7" || configExt == "net")
    {
        return readNetFromTorch(model.empty() ? config : model);
    }
    if (framework == "darknet" || modelExt == "weights" || configExt == "weights" ||
                                  modelExt == "cfg" || configExt == "cfg")
    {
        if (modelExt == "cfg" || configExt == "weights")
            std::swap(model, config);
        return readNetFromDarknet(config, model);
    }
    if (framework == "dldt" || modelExt == "bin" || configExt == "bin" ||
                               modelExt == "xml" || configExt == "xml")
    {
        if (modelExt == "xml" || configExt == "bin")
            std::swap(model, config);
        return readNetFromModelOptimizer(config, model);
    }
    if (framework == "onnx" || modelExt == "onnx")
    {
        return readNetFromONNX(model);
    }
    CV_Error(Error::StsError, "Cannot determine an origin framework of files: " +
                                      model + (config.empty() ? "" : ", " + config));
}

Net readNet(const String& _framework, const std::vector<uchar>& bufferModel,
            const std::vector<uchar>& bufferConfig)
{
    String framework = toLowerCase(_framework);
    if (framework == "caffe")
        return readNetFromCaffe(bufferConfig, bufferModel);
    else if (framework == "tensorflow")
        return readNetFromTensorflow(bufferModel, bufferConfig);
    else if (framework == "darknet")
        return readNetFromDarknet(bufferConfig, bufferModel);
    else if (framework == "torch")
        CV_Error(Error::StsNotImplemented, "Reading Torch models from buffers");
    else if (framework == "dldt")
        CV_Error(Error::StsNotImplemented, "Reading Intel's Model Optimizer models from buffers");
    CV_Error(Error::StsError, "Cannot determine an origin framework with a name " + framework);
}

Net readNetFromModelOptimizer(const String &xml, const String &bin)
{
    return Net::readFromModelOptimizer(xml, bin);
}

CV__DNN_INLINE_NS_END
}} // namespace
