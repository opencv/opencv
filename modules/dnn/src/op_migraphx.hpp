// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_OP_MIGRAPHX_HPP
#define OPENCV_DNN_OP_MIGRAPHX_HPP

#include <opencv2/core.hpp>

#ifdef HAVE_MIGRAPHX
#include <migraphx/migraphx.hpp>
#include <migraphx/version.h>
#include <vector>
#include <string>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

// Whole-model MIGraphX offload (Strategy A: hand the ONNX model to MIGraphX's
// native ONNX frontend, compile to the GPU target, and run the whole program).
class MIGraphXNet
{
public:
    // Parse ONNX bytes (fixing input dims), optionally quantize fp16, compile to gpu.
    bool build(const uchar* onnxData, size_t onnxSize,
               const std::vector<std::string>& inputNames,
               const std::vector<std::vector<int> >& inputShapes,
               bool fp16);

    // Bind host input Mats (by name, fallback by order) -> eval on GPU -> host output Mats.
    bool run(const std::vector<Mat>& inputBlobs,
             const std::vector<std::string>& inputNames,
             std::vector<Mat>& outputBlobs);

    bool compiled = false;
    bool offloadCopy = true;                 // Phase 1: host in/out, MIGraphX does H2D/D2H
    migraphx::program prog;
    std::vector<std::string> paramNames;     // program parameter (input) names
    std::vector<std::vector<int> > builtShapes;  // input shapes the program was compiled for
    bool builtFp16 = false;                  // fp16 flag the program was compiled with
};

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn

#endif // HAVE_MIGRAPHX
#endif // OPENCV_DNN_OP_MIGRAPHX_HPP
