// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_OP_HALIDE_HPP__
#define __OPENCV_DNN_OP_HALIDE_HPP__

#include "precomp.hpp"

#ifdef HAVE_HALIDE
#include <Halide.h>
#endif  // HAVE_HALIDE

namespace cv
{
namespace dnn
{
#ifdef HAVE_HALIDE
    // Returns four-dimensional buffer with float32 type that wrap cv::Mat data.
    // No data copy here.
    Halide::Buffer<float> wrapToHalideBuffer(const Mat& mat);

    Halide::Buffer<float> wrapToHalideBuffer(const Mat& mat,
                                             const std::vector<int>& shape);

    // Extract batch size, number of channels, width and height from buffer.
    void getCanonicalSize(const Halide::Buffer<>& buffer, int* width, int* height,
                          int* channels, int* batch);

    // Cast pointer and create copy of Halide buffer. No data copy.
    Halide::Buffer<> halideBuffer(const Ptr<BackendWrapper>& ptr);

    std::vector<Halide::Buffer<> > halideBuffers(const std::vector<Ptr<BackendWrapper> >& ptrs);

    class HalideBackendNode : public BackendNode
    {
    public:
        HalideBackendNode(const Halide::Func& func);

        HalideBackendNode(const std::vector<Halide::Func>& funcs);

        // Initialize from the <base> node but replace last function to <top>.
        // It's using in case of layers fusing when we want to keep functions of
        // root layer but replace top by fused one (i.e. conv+padding to relu+padding).
        HalideBackendNode(const Ptr<HalideBackendNode>& base, const Halide::Func& top);

        std::vector<Halide::Func> funcs;
    };

    class HalideBackendWrapper : public BackendWrapper
    {
    public:
        HalideBackendWrapper(int targetId, const cv::Mat& m);

        HalideBackendWrapper(const Ptr<BackendWrapper>& base, const MatShape& shape);

        virtual void copyToHost();

        Halide::Buffer<float> buffer;
    };
#endif  // HAVE_HALIDE

    // Extract batch size, number of channels, width and height from MatSize.
    void getCanonicalSize(const MatSize& size, int* width, int* height,
                          int* channels, int* batch);

    void getCanonicalSize(const MatShape& shape, int* width, int* height,
                          int* channels, int* batch);

    // Realize Halide pipeline into output blobs.
    void forwardHalide(std::vector<Ptr<BackendWrapper> > &outputs,
                       const Ptr<BackendNode>& node);

    // Compile Halide pipeline to specific target. Use outputs to set bounds of functions.
    void compileHalide(std::vector<Mat> &outputs, Ptr<BackendNode>& node, int targetId);

    bool haveHalide();
}  // namespace dnn
}  // namespace cv

#endif  // __OPENCV_DNN_OP_HALIDE_HPP__
