// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include "op_halide.hpp"
#include "net_impl.hpp"

#ifdef HAVE_HALIDE
#include "halide_scheduler.hpp"

#include <HalideRuntimeOpenCL.h>
#include <thread>
#endif  // HAVE_HALIDE

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN


void Net::Impl::setHalideScheduler(const String& scheduler)
{
    halideConfigFile = scheduler;
}


#ifdef HAVE_HALIDE


void Net::Impl::compileHalide()
{
    CV_TRACE_FUNCTION();

    CV_Assert(preferableBackend == DNN_BACKEND_HALIDE);

    HalideScheduler scheduler(halideConfigFile);
    std::vector< std::reference_wrapper<LayerData> > compileList; compileList.reserve(64);
    for (MapIdToLayerData::iterator it = layers.begin(); it != layers.end(); ++it)
    {
        LayerData& ld = it->second;
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


void Net::Impl::initHalideBackend()
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


#endif  // HAVE_HALIDE
CV__DNN_INLINE_NS_END


#ifdef HAVE_HALIDE
static MatShape getBufferShape(const MatShape& shape)
{
    if (shape.size() == 2 || shape.size() == 4)
    {
        int w, h, c, n;
        getCanonicalSize(shape, &w, &h, &c, &n);
        return {w, h, c, n};
    }
    else
    {
        MatShape bufferShape(shape);
        std::reverse(bufferShape.begin(), bufferShape.end());
        return bufferShape;
    }
}

static MatShape getBufferShape(const MatSize& size)
{
    return getBufferShape(shape(size));
}

Halide::Buffer<float> wrapToHalideBuffer(const Mat& mat)
{
    return wrapToHalideBuffer(mat, getBufferShape(mat.size));
}

Halide::Buffer<float> wrapToHalideBuffer(const Mat& mat,
                                         const std::vector<int>& sizes)
{
    Halide::Buffer<float> buffer((float*)mat.data, sizes);
    buffer.set_host_dirty();  // Indicate that data is on CPU.
    return buffer;
}

Halide::Buffer<> halideBuffer(const Ptr<BackendWrapper>& ptr)
{
    CV_Assert(!ptr.empty());
    return ptr.dynamicCast<HalideBackendWrapper>()->buffer;
}

std::vector<Halide::Buffer<> > halideBuffers(const std::vector<Ptr<BackendWrapper> >& ptrs)
{
    std::vector<Halide::Buffer<> > vec;
    vec.reserve(ptrs.size());
    for (const Ptr<BackendWrapper>& ptr : ptrs)
    {
        vec.push_back(halideBuffer(ptr));
    }
    return vec;
}

void getCanonicalSize(const Halide::Buffer<>& buffer, int* width, int* height,
                      int* channels, int* batch)
{
    CV_Assert(buffer.dimensions() == 4);
    *width = buffer.extent(0);
    *height = buffer.extent(1);
    *channels = buffer.extent(2);
    *batch = buffer.extent(3);
}

HalideBackendNode::HalideBackendNode(const Halide::Func& func)
    : BackendNode(DNN_BACKEND_HALIDE), funcs(1, func) {}

HalideBackendNode::HalideBackendNode(const std::vector<Halide::Func>& funcs)
    : BackendNode(DNN_BACKEND_HALIDE), funcs(funcs) {}

HalideBackendNode::HalideBackendNode(const Ptr<HalideBackendNode>& base,
                                     const Halide::Func& top)
    : BackendNode(DNN_BACKEND_HALIDE), funcs(base->funcs)
{
    funcs.back() = top;
}

HalideBackendWrapper::HalideBackendWrapper(int targetId, const cv::Mat& m)
    : BackendWrapper(DNN_BACKEND_HALIDE, targetId)
{
    managesDevMemory = true;
    buffer = wrapToHalideBuffer(m);
    if (targetId == DNN_TARGET_CPU)
    {
        return;
    }
    else if (targetId == DNN_TARGET_OPENCL)
    {
        Halide::Target t = Halide::get_host_target();
        t.set_feature(Halide::Target::OpenCL);
        buffer.copy_to_device(t);
    }
    else
        CV_Error(Error::StsNotImplemented, "Unknown target identifier");
}

HalideBackendWrapper::HalideBackendWrapper(const Ptr<BackendWrapper>& base,
                                           const MatShape& shape)
    : BackendWrapper(DNN_BACKEND_HALIDE, base->targetId)
{
    managesDevMemory = false;
    Halide::Buffer<float> baseBuffer = halideBuffer(base);
    buffer = Halide::Buffer<float>((float*)baseBuffer.raw_buffer()->host,
                                   getBufferShape(shape));
    if (baseBuffer.has_device_allocation())
    {
        buffer.raw_buffer()->device = baseBuffer.raw_buffer()->device;
        buffer.raw_buffer()->device_interface = baseBuffer.raw_buffer()->device_interface;
        buffer.set_device_dirty();
    }
    else
    {
        buffer.set_host_dirty();  // Indicate that data is on CPU.
        CV_Assert(targetId == DNN_TARGET_CPU);
    }
}

HalideBackendWrapper::~HalideBackendWrapper()
{
    if (buffer.has_device_allocation() && !managesDevMemory)
    {
        buffer.raw_buffer()->device = 0;
        buffer.raw_buffer()->device_interface = 0;
        buffer.set_device_dirty(false);
    }
}

void HalideBackendWrapper::copyToHost()
{
    if (buffer.device_dirty())
    {
        buffer.device_sync();
        buffer.copy_to_host();
    }
}

void HalideBackendWrapper::setHostDirty()
{
    buffer.set_device_dirty(false);
    buffer.set_host_dirty();
}
#endif  // HAVE_HALIDE

void getCanonicalSize(const MatSize& size, int* w, int* h, int* c, int* n)
{
    getCanonicalSize(shape(size), w, h, c, n);
}

void getCanonicalSize(const MatShape& shape, int* width, int* height,
                      int* channels, int* batch)
{
    const int dims = shape.size();
    CV_Assert(dims == 2 || dims == 4);
    *batch = shape[0];
    *channels = shape[1];
    if (dims == 4)
    {
        *width = shape[3];
        *height = shape[2];
    }
    else
    {
        *width = 1;
        *height = 1;
    }
}

void compileHalide(const std::vector<Mat> &outputs, Ptr<BackendNode>& node, int targetId)
{
#ifdef HAVE_HALIDE
    CV_Assert(!node.empty());
    Halide::Func& top = node.dynamicCast<HalideBackendNode>()->funcs.back();

    int outW, outH, outC, outN;
    Halide::Var x("x"), y("y"), c("c"), n("n");
    getCanonicalSize(outputs[0].size, &outW, &outH, &outC, &outN);
    top.bound(x, 0, outW).bound(y, 0, outH)
       .bound(c, 0, outC).bound(n, 0, outN);

    Halide::Target target = Halide::get_host_target();
    target.set_feature(Halide::Target::NoAsserts);
    if (targetId == DNN_TARGET_OPENCL)
    {
        target.set_feature(Halide::Target::OpenCL);
    }
    CV_Assert(target.supported());
    top.compile_jit(target);
#endif  // HAVE_HALIDE
}

void forwardHalide(std::vector<Ptr<BackendWrapper> > &outputs,
                   const Ptr<BackendNode>& node)
{
#ifdef HAVE_HALIDE
    CV_Assert(!node.empty());
    Halide::Func& top = node.dynamicCast<HalideBackendNode>()->funcs.back();
    auto outputBuffers = halideBuffers(outputs);
    top.realize(Halide::Realization(outputBuffers));
#endif  // HAVE_HALIDE
}

bool haveHalide()
{
#ifdef HAVE_HALIDE
    return true;
#else
    return false;
#endif  // HAVE_HALIDE
}


CV__DNN_INLINE_NS_BEGIN


void Layer::applyHalideScheduler(Ptr<BackendNode>& node, const std::vector<Mat*> &inputs,
                                 const std::vector<Mat> &outputs, int targetId) const
{
#ifndef HAVE_HALIDE
    CV_Error(Error::StsNotImplemented, "");
#else
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


CV__DNN_INLINE_NS_END
}}  // namespace
