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
#include "halide_scheduler.hpp"
#include "memory_manager.hpp"
#include <set>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iterator>
#include <numeric>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>

namespace cv {
namespace dnn {
CV__DNN_EXPERIMENTAL_NS_BEGIN

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

template<typename T>
static String toString(const T &v)
{
    std::ostringstream ss;
    ss << v;
    return ss.str();
}

Mat blobFromImage(const Mat& image, double scalefactor, const Size& size,
                  const Scalar& mean, bool swapRB, bool crop)
{
    CV_TRACE_FUNCTION();
    std::vector<Mat> images(1, image);
    return blobFromImages(images, scalefactor, size, mean, swapRB, crop);
}

Mat blobFromImages(const std::vector<Mat>& images_, double scalefactor, Size size,
                   const Scalar& mean_, bool swapRB, bool crop)
{
    CV_TRACE_FUNCTION();
    std::vector<Mat> images = images_;
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
              resize(images[i], images[i], Size(), resizeFactor, resizeFactor);
              Rect crop(Point(0.5 * (images[i].cols - size.width),
                              0.5 * (images[i].rows - size.height)),
                        size);
              images[i] = images[i](crop);
            }
            else
              resize(images[i], images[i], size);
        }
        if(images[i].depth() == CV_8U)
            images[i].convertTo(images[i], CV_32F);
        Scalar mean = mean_;
        if (swapRB)
            std::swap(mean[0], mean[2]);

        images[i] -= mean;
        images[i] *= scalefactor;
    }

    size_t i, nimages = images.size();
    if(nimages == 0)
        return Mat();
    Mat image0 = images[0];
    int nch = image0.channels();
    CV_Assert(image0.dims == 2);
    Mat blob, image;
    if (nch == 3 || nch == 4)
    {
        int sz[] = { (int)nimages, nch, image0.rows, image0.cols };
        blob = Mat(4, sz, CV_32F);
        Mat ch[4];

        for( i = 0; i < nimages; i++ )
        {
            image = images[i];
            CV_Assert(image.depth() == CV_32F);
            nch = image.channels();
            CV_Assert(image.dims == 2 && (nch == 3 || nch == 4));
            CV_Assert(image.size() == image0.size());

            for( int j = 0; j < nch; j++ )
                ch[j] = Mat(image.rows, image.cols, CV_32F, blob.ptr((int)i, j));
            if(swapRB)
                std::swap(ch[0], ch[2]);
            split(image, ch);
        }
    }
    else
    {
       CV_Assert(nch == 1);
       int sz[] = { (int)nimages, 1, image0.rows, image0.cols };
       blob = Mat(4, sz, CV_32F);

       for( i = 0; i < nimages; i++ )
       {
           Mat image = images[i];
           CV_Assert(image.depth() == CV_32F);
           nch = image.channels();
           CV_Assert(image.dims == 2 && (nch == 1));
           CV_Assert(image.size() == image0.size());

           image.copyTo(Mat(image.rows, image.cols, CV_32F, blob.ptr((int)i, 0)));
       }
    }
    return blob;
}

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
    LayerData() : id(-1), flag(0) {}
    LayerData(int _id, const String &_name, const String &_type, LayerParams &_params)
        : id(_id), name(_name), type(_type), params(_params), flag(0)
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

    Ptr<Layer> layerInstance;
    std::vector<Mat> outputBlobs;
    std::vector<Mat*> inputBlobs;
    std::vector<Mat> internals;
    // Computation nodes of implemented backends (except DEFAULT).
    std::map<int, Ptr<BackendNode> > backendNodes;
    // Flag for skip layer computation for specific backend.
    std::map<int, bool> skipFlags;

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
    void finalize(const std::vector<Mat*>&, std::vector<Mat>&) {}
    void forward(std::vector<Mat*>&, std::vector<Mat>&, std::vector<Mat> &) {}

    int outputNameToIndex(String tgtName)
    {
        int idx = (int)(std::find(outNames.begin(), outNames.end(), tgtName) - outNames.begin());
        return (idx < (int)outNames.size()) ? idx : -1;
    }

    void setNames(const std::vector<String> &names)
    {
        outNames.assign(names.begin(), names.end());
    }

private:
    std::vector<String> outNames;
};

static Ptr<BackendWrapper> wrapMat(int backendId, int targetId, const cv::Mat& m)
{
    if (backendId == DNN_BACKEND_DEFAULT)
    {
        return Ptr<BackendWrapper>();
    }
    else if (backendId == DNN_BACKEND_HALIDE)
    {
        CV_Assert(haveHalide());
#ifdef HAVE_HALIDE
        return Ptr<BackendWrapper>(new HalideBackendWrapper(targetId, m));
#endif  // HAVE_HALIDE
    }
    else
        CV_Error(Error::StsNotImplemented, "Unknown backend identifier");
    return Ptr<BackendWrapper>();
}

// Network's output layers memory allocator.
struct BlobManager
{
public:
    // Clear state of blob manager and resolve new one memory management task.
    void reset(const std::map<int, LayerData>& layers,
               const std::map<int, LayerShapes>& layersShapes,
               const std::vector<LayerPin>& blobsToKeep,
               int backendId_, int targetId_)
    {
        CV_TRACE_FUNCTION();

        memUsers.clear();
        inPlaceMap.clear();
        hostMemPoses.clear();
        devMemPoses.clear();
        devMemoryBuffers.clear();
        memoryBuffer.release();

        backendId = backendId_;
        targetId = targetId_;
        // References counter to layer outputs.
        std::map<LayerPin, int> refCounter;
        std::map<int, LayerData>::const_iterator it;

        // Represent memory management task as a schedule where every memory
        // user is characterized by start iteration, end iteration and required
        // memory size (in bytes).

        // Collect references counters, register output and internal blobs.
        for (it = layers.begin(); it != layers.end(); ++it)
        {
            const LayerData& ld = it->second;
            if (ld.id == 0)  // Don't process input layer.
                continue;
            const LayerShapes& shapes = layersShapes.find(ld.id)->second;
            incRefsToInputs(ld, refCounter);
            registerOutputs(ld, shapes);
            registerInternals(ld, shapes);
        }

        // Fake references to blobsToKeep.
        for (int i = 0; i < blobsToKeep.size(); ++i)
        {
            if (refCounter.find(blobsToKeep[i]) != refCounter.end())
            {
                refCounter[blobsToKeep[i]] += 1;
            }
        }

        // Set end iterations of users. It's a moment when references counter to
        // them equals zero.
        for (it = layers.begin(); it != layers.end(); ++it)
        {
            const LayerData& ld = it->second;
            if (ld.id == 0)  // Don't process input layer.
                continue;
            const LayerShapes& shapes = layersShapes.find(ld.id)->second;

            // Prolong input blobs up to current layer.
            for (int i = 0; i < ld.inputBlobsId.size(); ++i)
            {
                LayerPin id = ld.inputBlobsId[i];
                if (id.lid == 0)  // Don't process input layer.
                    continue;
                jumpOverInPlace(id);
                CV_Assert(refCounter.find(id) != refCounter.end());
                CV_Assert(refCounter[id] >= 1);
                CV_Assert(memUsers[id].endIter <= ld.id);

                // If layer work in-place and it's input is not used anymore,
                // we should keep only one memory user entity.
                bool inPlace = shapes.supportInPlace &&
                               ld.inputBlobsId.size() == 1 &&
                               refCounter[id] == 1;
                if (inPlace)
                {
                    for (int i = 0; i < shapes.out.size(); ++i)
                    {
                        LayerPin pin(ld.id, i);
                        CV_Assert(inPlaceMap.find(pin) == inPlaceMap.end());
                        CV_Assert(memUsers.find(pin) != memUsers.end());
                        inPlaceMap[pin] = id;
                        refCounter[id] += refCounter[pin];
                        memUsers.erase(pin);
                    }
                }
                memUsers[id].endIter = ld.id;
                refCounter[id] -= 1;
            }

            // Special case - concat layer.
            // There is some optimization that skip concat layer in case of
            // concatenation by channels. That means input layer of concat layer
            // are processed directly in right positions of concat layer blob.
            //
            // To protect concat layer memory during all it's inputs processing,
            // we manually decreace concat layer start iteration.
            // std::map<LayerPin, MemoryUser>::iterator inputUserIt;
            Ptr<ConcatLayer> concatLayer = ld.layerInstance.dynamicCast<ConcatLayer>();
            if(!concatLayer.empty() && shapes.out.size() == 1)
            {
                CV_Assert(memUsers.find(LayerPin(ld.id, 0)) != memUsers.end());
                MemoryUser& concatMemUser = memUsers[LayerPin(ld.id, 0)];
                for (int i = 0; i < ld.inputBlobsId.size(); ++i)
                {
                    if (ld.inputBlobsId[i].lid != 0)  // Not input layer.
                    {
                        LayerPin id = ld.inputBlobsId[i];
                        jumpOverInPlace(id);
                        concatMemUser.startIter = std::min(concatMemUser.startIter,
                                                           memUsers[id].startIter);
                    }
                }
            }
        }

        // For blobsToKeep we increase end iteration up to the end of pipeline
        // (make them impossible for reusing).
        for (int i = 0; i < blobsToKeep.size(); ++i)
        {
            LayerPin pin = blobsToKeep[i];
            jumpOverInPlace(pin);
            memUsers[pin].endIter = layers.size() - 1;
        }

        // Solve the task.
        std::vector<MemoryUser> memUsersVec;
        memUsersVec.reserve(memUsers.size());
        for (std::map<LayerPin, MemoryUser>::iterator it = memUsers.begin();
             it != memUsers.end(); ++it)
        {
            it->second.id = memUsersVec.size();
            memUsersVec.push_back(it->second);
        }

        uint64_t hostMemUsage;
        MemoryManager::solveOpt(memUsersVec, hostMemPoses, &hostMemUsage);
        memoryBuffer.create(std::vector<int>(1, hostMemUsage), CV_32F);

        if (targetId != DNN_TARGET_CPU)
        {
            uint64_t devMemUsage;
            std::vector<int> buffersIds;
            MemoryManager::solveReuseOrCreate(memUsersVec, devMemPoses, buffersIds, &devMemUsage);
            for (int i = 0; i < buffersIds.size(); ++i)
            {
                int id = buffersIds[i];
                uint64_t hostMemPos = hostMemPoses[id];
                uint64_t devMemPos = devMemPoses[id];
                unsigned char* host = memoryBuffer.data + hostMemPos;
                CV_Assert(devMemoryBuffers.find(devMemPos) == devMemoryBuffers.end());

                std::vector<int> shape(1, memUsersVec[id].memSize);
                devMemoryBuffers[devMemPos] =
                    wrapMat(backendId, targetId, Mat(shape, CV_32F, host));
            }
        }
    }

    // Assign memory into output blobs. Also wrap to specific backend and target.
    void allocateBlobsForLayer(LayerData &ld, const LayerShapes& layerShapes)
    {
        CV_TRACE_FUNCTION();

        std::vector<Mat>& outputBlobs = ld.outputBlobs,
                &internalBlobs = ld.internals;

        const ShapesVec& outShapes = layerShapes.out,
                internalShapes = layerShapes.internal;

        outputBlobs.resize(std::max((size_t)1, outShapes.size())); //layer produce at least one output blob
        internalBlobs.resize(internalShapes.size());

        CV_Assert(ld.requiredOutputs.size() <= outShapes.size());

        ld.outputBlobsWrappers.assign(outputBlobs.size(), Ptr<BackendWrapper>());
        for(int i = 0; i < outputBlobs.size(); i++)
        {
            allocate(outShapes[i], LayerPin(ld.id, i), outputBlobs[i]);
            if (backendId != DNN_BACKEND_DEFAULT)
            {
                ld.outputBlobsWrappers[i] = wrap(outShapes[i], LayerPin(ld.id, i), outputBlobs[i]);
            }
        }
        for(int i = 0; i < internalBlobs.size(); i++)
        {
            allocate(internalShapes[i], LayerPin(ld.id, outputBlobs.size() + i),
                     internalBlobs[i]);
        }
    }

private:
    static void incRefsToInputs(const LayerData& ld, std::map<LayerPin, int>& refCounter)
    {
        for (int i = 0; i < ld.inputBlobsId.size(); ++i)
        {
            const LayerPin& id = ld.inputBlobsId[i];
            std::map<LayerPin, int>::iterator it = refCounter.find(id);
            if (it == refCounter.end())
            {
                refCounter[id] = 1;
            }
            else
            {
                it->second += 1;
            }
        }
    }

    void registerOutputs(const LayerData& ld, const LayerShapes& layerShapes)
    {
        const int numOuts = layerShapes.out.size();
        for (int i = 0; i < numOuts; ++i)
        {
            LayerPin pin(ld.id, i);
            uint64_t memSize = total(layerShapes.out[i]) * sizeof(float);
            CV_Assert(memUsers.find(pin) == memUsers.end());
            memUsers[pin] = MemoryUser(ld.id, ld.id, memSize);
        }
    }

    void registerInternals(const LayerData& ld, const LayerShapes& layerShapes)
    {
        const int numOuts = layerShapes.out.size();
        for (int i = 0; i < layerShapes.internal.size(); ++i)
        {
            LayerPin pin(ld.id, numOuts + i);
            uint64_t memSize = total(layerShapes.internal[i]) * sizeof(float);
            CV_Assert(memUsers.find(pin) == memUsers.end());
            memUsers[pin] = MemoryUser(ld.id, ld.id, memSize);
        }
    }

    void jumpOverInPlace(LayerPin& lp)
    {
        if (inPlaceMap.find(lp) != inPlaceMap.end())
        {
            CV_Assert(memUsers.find(lp) == memUsers.end());
            lp = inPlaceMap[lp];
            CV_Assert(inPlaceMap.find(lp) == inPlaceMap.end());
        }
        CV_Assert(memUsers.find(lp) != memUsers.end());
    }

    void allocate(const MatShape& shape, LayerPin lp, Mat& dst)
    {
        jumpOverInPlace(lp);
        uint64_t memPos = hostMemPoses[memUsers[lp].id];
        dst = Mat(shape, CV_32F, memoryBuffer.data + memPos);
    }

    Ptr<BackendWrapper> wrap(const MatShape& shape, LayerPin lp, const Mat& host)
    {
        jumpOverInPlace(lp);

        Ptr<BackendWrapper> wrapper;
        if (targetId == DNN_TARGET_CPU)
        {
            wrapper = wrapMat(backendId, targetId, host);
        }
        else
        {
            uint64_t memPos = devMemPoses[memUsers[lp].id];
            CV_Assert(devMemoryBuffers.find(memPos) != devMemoryBuffers.end());
            Ptr<BackendWrapper> baseBuffer = devMemoryBuffers[memPos];

            if (backendId == DNN_BACKEND_HALIDE)
            {
                CV_Assert(haveHalide());
#ifdef HAVE_HALIDE
                wrapper = Ptr<BackendWrapper>(new HalideBackendWrapper(baseBuffer, shape, host));
#endif  // HAVE_HALIDE
            }
            else
                CV_Error(Error::StsNotImplemented, "Unknown backend identifier");
        }
        return wrapper;
    }

    std::map<LayerPin, MemoryUser> memUsers;
    std::map<LayerPin, LayerPin> inPlaceMap;

    std::vector<uint64_t> hostMemPoses;
    std::vector<uint64_t> devMemPoses;
    std::map<uint64_t, Ptr<BackendWrapper> > devMemoryBuffers;
    cv::Mat memoryBuffer;

    int backendId, targetId;
};

struct Net::Impl
{
    typedef std::map<int, LayerShapes> LayersShapesMap;
    typedef std::map<int, LayerData> MapIdToLayerData;

    Impl()
    {
        //allocate fake net input layer
        netInputLayer = Ptr<DataLayer>(new DataLayer());
        LayerData &inpl = layers.insert( make_pair(0, LayerData()) ).first->second;
        inpl.id = 0;
        inpl.name = "_input";
        inpl.type = "__NetInputLayer__";
        inpl.layerInstance = netInputLayer;
        layerNameToId.insert(std::make_pair(inpl.name, inpl.id));

        lastLayerId = 0;
        netWasAllocated = false;
        fusion = true;
        preferableBackend = DNN_BACKEND_DEFAULT;
        preferableTarget = DNN_TARGET_CPU;
    }

    Ptr<DataLayer> netInputLayer;
    std::vector<int> netOutputs;
    std::vector<LayerPin> blobsToKeep;
    MapIdToLayerData layers;
    std::map<String, int> layerNameToId;
    BlobManager blobManager;
    int preferableBackend;
    int preferableTarget;
    String halideConfigFile;

    int lastLayerId;

    bool netWasAllocated;
    bool fusion;
    std::vector<int64> layersTimings;

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
            if (layer->supportBackend(DNN_BACKEND_HALIDE) && !ld.skipFlags[DNN_BACKEND_HALIDE])
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
            it->second.skipFlags.clear();
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
        it = layers.find(0);
        CV_Assert(it != layers.end());
        it->second.skipFlags[DNN_BACKEND_DEFAULT] = true;

        layersTimings.clear();
    }

    void setUpNet(const std::vector<LayerPin>& blobsToKeep_ = std::vector<LayerPin>())
    {
        CV_TRACE_FUNCTION();

        if (!netWasAllocated || this->blobsToKeep != blobsToKeep_)
        {
            clear();

            allocateLayers(blobsToKeep_);
            computeNetOutputLayers();
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
            CV_Error(Error::StsError, "Requsted layer \"" + layerName + "\" not found");

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
                CV_Error(Error::StsError, "Input #" + toString(inNum) + "of layer \"" + ld.name + "\" already was connected");
        }

        ld.inputBlobsId[inNum] = from;
    }

    static void splitPin(const String &pinAlias, String &layerName, String &outName)
    {
        size_t delimPos = pinAlias.find('.');
        layerName = pinAlias.substr(0, delimPos);
        outName = (delimPos == String::npos) ? String() : pinAlias.substr(delimPos + 1);
    }

    int resolvePinOutputName(LayerData &ld, const String &outName)
    {
        if (outName.empty())
            return 0;

        if (std::isdigit(outName[0]))
        {
            char *lastChar;
            long inum = std::strtol(outName.c_str(), &lastChar, 10);

            if (*lastChar == 0)
            {
                CV_Assert(inum == (int)inum);
                return (int)inum;
            }
        }

        return ld.getLayerInstance()->outputNameToIndex(outName);
    }

    LayerPin getPinByAlias(const String &pinAlias)
    {
        LayerPin pin;
        String layerName, outName;
        splitPin(pinAlias, layerName, outName);

        pin.lid = (layerName.empty()) ? 0 : getLayerId(layerName);

        if (pin.lid >= 0)
            pin.oid = resolvePinOutputName(getLayerData(pin.lid), outName);

        return pin;
    }

    std::vector<LayerPin> getLayerOutPins(const String &pinAlias)
    {
        String layerName, outName;
        splitPin(pinAlias, layerName, outName);

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

    void computeNetOutputLayers()
    {
        CV_TRACE_FUNCTION();

        netOutputs.clear();

        MapIdToLayerData::iterator it;
        for (it = layers.begin(); it != layers.end(); it++)
        {
            int lid = it->first;
            LayerData &ld = it->second;

            if (ld.requiredOutputs.size() == 0)
                netOutputs.push_back(lid);
        }

        #ifndef NDEBUG
        std::cout << "\nNet Outputs(" << netOutputs.size() << "):\n";
        for (size_t i = 0; i < netOutputs.size(); i++)
            std::cout << layers[netOutputs[i]].name << "\n";
        #endif
    }

    void initBackend()
    {
        CV_TRACE_FUNCTION();

        if (preferableBackend == DNN_BACKEND_DEFAULT)
        {
            CV_Assert(preferableTarget == DNN_TARGET_CPU || preferableTarget == DNN_TARGET_OPENCL);
            return;
        }

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
                        ldTop.skipFlags[preferableBackend] = true;
                        ldBot.backendNodes[preferableBackend] = fusedNode;
                        continue;
                    }
                }
            }
            // No layers fusion.
            ldTop.skipFlags[preferableBackend] = false;
            if (preferableBackend == DNN_BACKEND_HALIDE)
            {
                ldTop.backendNodes[DNN_BACKEND_HALIDE] =
                    layerTop->initHalide(ldTop.inputBlobsWrappers);
                baseIt = it;
            }
            else
            {
                CV_Error(Error::StsNotImplemented, "Unknown backend identifier");
            }
        }
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

        LayersShapesMap::const_iterator layerShapesIt = layersShapes.find(lid);

        CV_Assert(layerShapesIt != layersShapes.end());

        if (ld.id != 0)
            blobManager.allocateBlobsForLayer(ld, layerShapesIt->second);
        else
        {
            ld.outputBlobsWrappers.resize(ld.outputBlobs.size());
            for (int i = 0; i < ld.outputBlobs.size(); ++i)
            {
                ld.outputBlobsWrappers[i] = wrapMat(preferableBackend,
                                                    preferableTarget,
                                                    ld.outputBlobs[i]);
            }
        }

        Ptr<Layer> layerPtr = ld.getLayerInstance();
        {
            layerPtr->finalize(ld.inputBlobs, ld.outputBlobs);
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

        ld.flag = 1;
    }

#if 0
#define printf_(args) printf args
#else
#define printf_(args)
#endif

    void fuseLayers(const std::vector<LayerPin>& blobsToKeep_)
    {
        if( !fusion || preferableBackend != DNN_BACKEND_DEFAULT)
            return;

        CV_TRACE_FUNCTION();

        // scan through all the layers. If there is convolution layer followed by the activation layer,
        // we try to embed this activation into the convolution and disable separate execution of the activation
        std::vector<String> outnames;
        std::set<LayerPin> pinsToKeep(blobsToKeep_.begin(),
                                      blobsToKeep_.end());
        MapIdToLayerData::iterator it;
        for (it = layers.begin(); it != layers.end(); it++)
        {
            int lid = it->first;
            LayerData& ld = layers[lid];
            if( ld.skipFlags[DNN_BACKEND_DEFAULT] )
            {
                printf_(("skipped %s: %s\n", ld.layerInstance->name.c_str(), ld.layerInstance->type.c_str()));
                continue;
            }
            printf_(("analyzing %s: %s\n", ld.layerInstance->name.c_str(), ld.layerInstance->type.c_str()));
            if( ld.consumers.size() == 0 )
                outnames.push_back(ld.layerInstance->name);

            // the optimization #1. try to fuse batch norm, scaling and/or activation layers
            // with the current layer if they follow it. Normally, the are fused with the convolution layer,
            // but some of them (like activation) may be fused with fully-connected, elemwise (+) and
            // some other layers.

            // TODO: OpenCL target support more fusion styles.
            if ( preferableTarget == DNN_TARGET_OPENCL && ld.layerInstance->type.compare("Convolution") )
                continue;

            Ptr<Layer>& currLayer = ld.layerInstance;
            if( ld.consumers.size() == 1 && pinsToKeep.count(LayerPin(lid, 0)) == 0 )
            {
                LayerData* nextData = &layers[ld.consumers[0].lid];
                Ptr<BatchNormLayer> nextBNormLayer =
                    nextData->layerInstance.dynamicCast<BatchNormLayer>();
                LayerPin lpNext(ld.consumers[0].lid, 0);
                if( !nextBNormLayer.empty() && pinsToKeep.count(lpNext) == 0 )
                {
                    LayerData* bnormData = nextData;
                    nextData = 0;
                    if( currLayer->setBatchNorm(nextBNormLayer) )
                    {
                        printf_(("\tfused with %s\n", nextBNormLayer->name.c_str()));
                        bnormData->skipFlags[DNN_BACKEND_DEFAULT] = true;
                        ld.outputBlobs = layers[lpNext.lid].outputBlobs;
                        if( bnormData->consumers.size() == 1 )
                        {
                            nextData = &layers[bnormData->consumers[0].lid];
                            lpNext = LayerPin(bnormData->consumers[0].lid, 0);
                        }
                    }
                }

                Ptr<ScaleLayer> nextScaleLayer;
                if( nextData )
                    nextScaleLayer = nextData->layerInstance.dynamicCast<ScaleLayer>();
                if( !nextScaleLayer.empty() && pinsToKeep.count(lpNext) == 0 )
                {
                    LayerData* scaleData = nextData;
                    nextData = 0;
                    if( currLayer->setScale(nextScaleLayer) )
                    {
                        printf_(("\tfused with %s\n", nextScaleLayer->name.c_str()));
                        scaleData->skipFlags[DNN_BACKEND_DEFAULT] = true;
                        ld.outputBlobs = layers[lpNext.lid].outputBlobs;
                        if( scaleData->consumers.size() == 1 )
                        {
                            nextData = &layers[scaleData->consumers[0].lid];
                            lpNext = LayerPin(scaleData->consumers[0].lid, 0);
                        }
                    }
                }

                // For now,  OpenCL target only support fusion with activation of ReLU/ChannelsPReLU
                if ( preferableTarget != DNN_TARGET_OPENCL ||
                        (preferableTarget == DNN_TARGET_OPENCL &&
                         nextData &&
                        (!nextData->type.compare("ReLU") ||
                         !nextData->type.compare("ChannelsPReLU"))) )
                {

                    Ptr<ActivationLayer> nextActivLayer;

                    if( nextData )
                        nextActivLayer = nextData->layerInstance.dynamicCast<ActivationLayer>();

                    if( !nextActivLayer.empty() && pinsToKeep.count(lpNext) == 0
                            && currLayer->setActivation(nextActivLayer) )
                    {
                        LayerData *activData = nextData;
                        printf_(("\tfused with %s\n", nextActivLayer->name.c_str()));
                        activData->skipFlags[DNN_BACKEND_DEFAULT] = true;
                        ld.outputBlobs = layers[lpNext.lid].outputBlobs;
                    }
                }
            }

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
            // the concat layer to write to the concatetion output buffer
            // (and so we eliminate the concatenation layer, because the channels
            // are concatenated implicitly).
            Ptr<ConcatLayer> concatLayer = ld.layerInstance.dynamicCast<ConcatLayer>();
            if( !concatLayer.empty() && concatLayer->axis == 1 && !concatLayer->padding &&
                ld.outputBlobs.size() == 1 )
            {
                Mat& output = ld.outputBlobs[0];

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
                        while(inp_i_data->skipFlags[DNN_BACKEND_DEFAULT] &&
                              inp_i_data->inputBlobsId.size() == 1)
                        {
                            pin = inp_i_data->inputBlobsId[0];
                            inp_i_data = &layers[pin.lid];
                        }
                        printf_(("\treal input for %s is %s\n",
                               layers[ld.inputBlobsId[i].lid].getLayerInstance()->name.c_str(),
                               inp_i_data->getLayerInstance()->name.c_str()));

                        if(inp_i_data->skipFlags[DNN_BACKEND_DEFAULT])
                            break;
                        realinputs[i] = pin;
                    }

                    if( i >= ninputs )
                    {
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
                            curr_output = output_slice;
                        }
                        ld.skipFlags[DNN_BACKEND_DEFAULT] = true;
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
            CV_Assert(layers[0].outputBlobs[i].total());
            inputShapes.push_back(shape(layers[0].outputBlobs[i]));
        }
        LayersShapesMap layersShapes;
        getLayersShapes(inputShapes, layersShapes);

        blobManager.reset(layers, layersShapes, blobsToKeep_, preferableBackend,
                          preferableTarget);

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

        if (preferableBackend == DNN_BACKEND_DEFAULT ||
            !layer->supportBackend(preferableBackend))
        {
            if( !ld.skipFlags[DNN_BACKEND_DEFAULT] )
            {
                for (int i = 0, n = ld.inputBlobsWrappers.size(); i < n; ++i)
                {
                    if (!ld.inputBlobsWrappers[i].empty())
                        ld.inputBlobsWrappers[i]->copyToHost();
                }
                layer->forward(ld.inputBlobs, ld.outputBlobs, ld.internals);
                for (int i = 0, n = ld.outputBlobsWrappers.size(); i < n; ++i)
                {
                    if (!ld.outputBlobsWrappers[i].empty())
                        ld.outputBlobsWrappers[i]->setHostDirty();
                }
            }
            else
                tm.reset();
        }
        else if (!ld.skipFlags[preferableBackend])
        {
            Ptr<BackendNode> node = ld.backendNodes[preferableBackend];
            if (preferableBackend == DNN_BACKEND_HALIDE)
            {
                forwardHalide(ld.outputBlobsWrappers, node);
            }
            else
            {
                CV_Error(Error::StsNotImplemented, "Unknown backend identifier");
            }
        }

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
            CV_Error(Error::StsOutOfRange, "Layer \"" + ld.name + "\" produce only " + toString(ld.outputBlobs.size()) +
                                           " outputs, the #" + toString(pin.oid) + " was requsted");
        }
        if (!ld.outputBlobsWrappers[pin.oid].empty())
        {
            // Transfer data to CPU if it's require.
            ld.outputBlobsWrappers[pin.oid]->copyToHost();
        }
        else
        {
            CV_Assert(preferableTarget == DNN_TARGET_CPU ||
                      (preferableBackend == DNN_BACKEND_DEFAULT && preferableTarget == DNN_TARGET_OPENCL));
        }
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

Net::~Net()
{
}

int Net::addLayer(const String &name, const String &type, LayerParams &params)
{
    CV_TRACE_FUNCTION();

    if (name.find('.') != String::npos)
    {
        CV_Error(Error::StsBadArg, "Added layer name \"" + name + "\" must not contain dot symbol");
        return -1;
    }

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

    impl->setUpNet();
    impl->forwardToLayer(impl->getLayerData(layerName));

    return impl->getBlob(layerName);
}

void Net::forward(std::vector<Mat>& outputBlobs, const String& outputName)
{
    CV_TRACE_FUNCTION();

    impl->setUpNet();

    String layerName = outputName;

    if (layerName.empty())
        layerName = getLayerNames().back();

    impl->forwardToLayer(impl->getLayerData(layerName));

    LayerPin pin = impl->getPinByAlias(layerName);
    LayerData &ld = impl->layers[pin.lid];
    outputBlobs = ld.outputBlobs;
}

void Net::forward(std::vector<Mat>& outputBlobs,
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

    outputBlobs.clear();
    for (int i = 0; i < pins.size(); i++)
    {
        outputBlobs.push_back(impl->getBlob(pins[i]));
    }
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
        impl->netWasAllocated = false;
        impl->clear();
    }
}

void Net::setInputsNames(const std::vector<String> &inputBlobNames)
{
    CV_TRACE_FUNCTION();

    impl->netInputLayer->setNames(inputBlobNames);
}

void Net::setInput(const Mat &blob_, const String& name)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG_VALUE(name, "name", name.c_str());

    LayerPin pin;
    pin.lid = 0;
    pin.oid = impl->resolvePinOutputName(impl->getLayerData(pin.lid), name);

    if (!pin.valid())
        CV_Error(Error::StsObjectNotFound, "Requested blob \"" + name + "\" not found");

    LayerData &ld = impl->layers[pin.lid];
    ld.outputBlobs.resize( std::max(pin.oid+1, (int)ld.requiredOutputs.size()) );
    ld.outputBlobsWrappers.resize(ld.outputBlobs.size());
    MatShape prevShape = shape(ld.outputBlobs[pin.oid]);
    bool oldShape = prevShape == shape(blob_);
    if (oldShape)
        blob_.copyTo(ld.outputBlobs[pin.oid]);
    else
        ld.outputBlobs[pin.oid] = blob_.clone();

    if (!ld.outputBlobsWrappers[pin.oid].empty())
    {
        ld.outputBlobsWrappers[pin.oid]->setHostDirty();
    }
    impl->netWasAllocated = impl->netWasAllocated && oldShape;
}

Mat Net::getParam(LayerId layer, int numParam)
{
    LayerData &ld = impl->getLayerData(layer);

    std::vector<Mat> &layerBlobs = ld.layerInstance->blobs;
    CV_Assert(numParam < (int)layerBlobs.size());
    return layerBlobs[numParam];
}

void Net::setParam(LayerId layer, int numParam, const Mat &blob)
{
    LayerData &ld = impl->getLayerData(layer);

    std::vector<Mat> &layerBlobs = ld.layerInstance->blobs;
    CV_Assert(numParam < (int)layerBlobs.size());
    //we don't make strong checks, use this function carefully
    layerBlobs[numParam] = blob;
}

int Net::getLayerId(const String &layer)
{
    return impl->getLayerId(layer);
}

void Net::deleteLayer(LayerId)
{
    CV_Error(Error::StsNotImplemented, "");
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
    int64 total = std::accumulate(timings.begin(), timings.end(), 0);
    return total;
}

//////////////////////////////////////////////////////////////////////////

Importer::~Importer() {}

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

int Layer::outputNameToIndex(String)
{
    return -1;
}

bool Layer::supportBackend(int backendId)
{
    return backendId == DNN_BACKEND_DEFAULT;
}

Ptr<BackendNode> Layer::initHalide(const std::vector<Ptr<BackendWrapper> > &)
{
    CV_Error(Error::StsNotImplemented, "Halide pipeline of " + type +
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
        int c_split = outC > 8 ? (outC > 16 ? 8 : 4) : outC;
        if (outW == 1 && outH == 1)
        {
            top.split(c, co, ci, c_split)
               .fuse(x, y, tile).fuse(co, tile, tile).fuse(n, tile, tile)
               .gpu_blocks(tile)
               .gpu_threads(ci);
        }
        else
        {
            int x_split = outW > 8 ? (outW >= 32 ? 16 : 8) : outW;
            int y_split = outH > 8 ? (outH >= 32 ? 16 : 8) : outH;
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
bool Layer::setBatchNorm(const Ptr<BatchNormLayer>&) { return false; }
bool Layer::setScale(const Ptr<ScaleLayer>&) { return false; }
void Layer::unsetAttached()
{
    setActivation(Ptr<ActivationLayer>());
    setBatchNorm(Ptr<BatchNormLayer>());
    setScale(Ptr<ScaleLayer>());
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

    std::vector<Mat*> inputsp;
    vecToPVec(inputs, inputsp);
    this->finalize(inputsp, outputs);
}

void Layer::finalize(const std::vector<Mat*> &input, std::vector<Mat> &output)
{
    (void)input;(void)output;
}

std::vector<Mat> Layer::finalize(const std::vector<Mat> &inputs)
{
    CV_TRACE_FUNCTION();

    std::vector<Mat> outputs;
    this->finalize(inputs, outputs);
    return outputs;
}

void Layer::forward(const std::vector<Mat> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
{
    CV_TRACE_FUNCTION();

    std::vector<Mat*> inputsp;
    vecToPVec(inputs, inputsp);
    this->forward(inputsp, outputs, internals);
}

void Layer::run(const std::vector<Mat> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
{
    CV_TRACE_FUNCTION();

    std::vector<Mat*> inputsp;
    vecToPVec(inputs, inputsp);
    this->finalize(inputsp, outputs);
    this->forward(inputsp, outputs, internals);
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

typedef std::map<String, LayerFactory::Constuctor> LayerFactory_Impl;

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

void LayerFactory::registerLayer(const String &type, Constuctor constructor)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG_VALUE(type, "type", type.c_str());

    cv::AutoLock lock(getLayerFactoryMutex());
    String type_ = type.toLowerCase();
    LayerFactory_Impl::const_iterator it = getLayerFactoryImpl().find(type_);

    if (it != getLayerFactoryImpl().end() && it->second != constructor)
    {
        CV_Error(cv::Error::StsBadArg, "Layer \"" + type_ + "\" already was registered");
    }

    getLayerFactoryImpl().insert(std::make_pair(type_, constructor));
}

void LayerFactory::unregisterLayer(const String &type)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG_VALUE(type, "type", type.c_str());

    cv::AutoLock lock(getLayerFactoryMutex());
    String type_ = type.toLowerCase();
    getLayerFactoryImpl().erase(type_);
}

Ptr<Layer> LayerFactory::createLayerInstance(const String &type, LayerParams& params)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG_VALUE(type, "type", type.c_str());

    cv::AutoLock lock(getLayerFactoryMutex());
    String type_ = type.toLowerCase();
    LayerFactory_Impl::const_iterator it = getLayerFactoryImpl().find(type_);

    if (it != getLayerFactoryImpl().end())
    {
        return it->second(params);
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

BackendWrapper::~BackendWrapper() {}

CV__DNN_EXPERIMENTAL_NS_END
}} // namespace
