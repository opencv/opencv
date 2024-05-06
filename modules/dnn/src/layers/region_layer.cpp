/*M ///////////////////////////////////////////////////////////////////////////////////////
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
// Copyright (C) 2017, Intel Corporation, all rights reserved.
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

#include "../precomp.hpp"
#include "../op_cuda.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include "../nms.inl.hpp"
#include "cpu_kernels/softmax.hpp"

#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
#endif

#ifdef HAVE_DNN_NGRAPH
#include "../ie_ngraph.hpp"
#endif

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/region.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv
{
namespace dnn
{

class RegionLayerImpl CV_FINAL : public RegionLayer
{
public:
    int coords, classes, anchors, classfix;
    float thresh, scale_x_y;
    int new_coords;
    bool useSoftmax, useLogistic;
#ifdef HAVE_OPENCL
    UMat blob_umat;
#endif

    RegionLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        CV_Assert(blobs.size() == 1);

        thresh = params.get<float>("thresh", 0.2);
        coords = params.get<int>("coords", 4);
        classes = params.get<int>("classes", 0);
        anchors = params.get<int>("anchors", 5);
        classfix = params.get<int>("classfix", 0);
        useSoftmax = params.get<bool>("softmax", false);
        useLogistic = params.get<bool>("logistic", false);
        nmsThreshold = params.get<float>("nms_threshold", 0.4);
        scale_x_y = params.get<float>("scale_x_y", 1.0); // Yolov4
        new_coords = params.get<int>("new_coords", 0); // Yolov4x-mish

        CV_Assert(nmsThreshold >= 0.);
        CV_Assert(coords == 4);
        CV_Assert(classes >= 1);
        CV_Assert(anchors >= 1);
        CV_Assert(useLogistic || useSoftmax);
        if (params.get<bool>("softmax_tree", false))
            CV_Error(cv::Error::StsNotImplemented, "Yolo9000 is not implemented");
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() > 0);
        // channels == cell_size*anchors
        CV_Assert(inputs[0][3] == (1 + coords + classes)*anchors);
        int batch_size = inputs[0][0];
        if(batch_size > 1)
            outputs = std::vector<MatShape>(1, shape(batch_size, inputs[0][1] * inputs[0][2] * anchors, inputs[0][3] / anchors));
        else
            outputs = std::vector<MatShape>(1, shape(inputs[0][1] * inputs[0][2] * anchors, inputs[0][3] / anchors));
        return false;
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
#ifdef HAVE_DNN_NGRAPH
    if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        return preferableTarget != DNN_TARGET_MYRIAD && new_coords == 0;
#endif
#ifdef HAVE_CUDA
        if (backendId == DNN_BACKEND_CUDA)
            return true;
#endif
        return backendId == DNN_BACKEND_OPENCV;
    }

    float logistic_activate(float x) { return 1.F / (1.F + exp(-x)); }

    void softmax_activate(const float* input, const int n, const float temp, float* output)
    {
        int i;
        float sum = 0;
        float largest = -FLT_MAX;
        for (i = 0; i < n; ++i) {
            if (input[i] > largest) largest = input[i];
        }
        for (i = 0; i < n; ++i) {
            float e = exp((input[i] - largest) / temp);
            sum += e;
            output[i] = e;
        }
        for (i = 0; i < n; ++i) {
            output[i] /= sum;
        }
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inps, OutputArrayOfArrays outs, OutputArrayOfArrays internals)
    {
        if (blob_umat.empty())
            blobs[0].copyTo(blob_umat);

        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        // TODO: implement a logistic activation to classification scores.
        if (useLogistic || inps.depth() == CV_16F)
            return false;

        inps.getUMatVector(inputs);
        outs.getUMatVector(outputs);

        CV_Assert(inputs.size() >= 1);
        int const cell_size = classes + coords + 1;

        for (size_t ii = 0; ii < outputs.size(); ii++)
        {
            UMat& inpBlob = inputs[ii];
            UMat& outBlob = outputs[ii];

            int batch_size = inpBlob.size[0];
            int rows = inpBlob.size[1];
            int cols = inpBlob.size[2];

            // channels == cell_size*anchors, see l. 94
            int sample_size = cell_size*rows*cols*anchors;

            ocl::Kernel logistic_kernel("logistic_activ", ocl::dnn::region_oclsrc);
            size_t nanchors = rows*cols*anchors*batch_size;
            logistic_kernel.set(0, (int)nanchors);
            logistic_kernel.set(1, ocl::KernelArg::PtrReadOnly(inpBlob));
            logistic_kernel.set(2, (int)cell_size);
            logistic_kernel.set(3, ocl::KernelArg::PtrWriteOnly(outBlob));
            logistic_kernel.run(1, &nanchors, NULL, false);

            if (useSoftmax)
            {
                // Yolo v2
                // softmax activation for Probability, for each grid cell (X x Y x Anchor-index)
                ocl::Kernel softmax_kernel("softmax_activ", ocl::dnn::region_oclsrc);
                size_t nanchors = rows*cols*anchors*batch_size;
                softmax_kernel.set(0, (int)nanchors);
                softmax_kernel.set(1, ocl::KernelArg::PtrReadOnly(inpBlob));
                softmax_kernel.set(2, ocl::KernelArg::PtrReadOnly(blob_umat));
                softmax_kernel.set(3, (int)cell_size);
                softmax_kernel.set(4, (int)classes);
                softmax_kernel.set(5, (int)classfix);
                softmax_kernel.set(6, (int)rows);
                softmax_kernel.set(7, (int)cols);
                softmax_kernel.set(8, (int)anchors);
                softmax_kernel.set(9, (float)thresh);
                softmax_kernel.set(10, ocl::KernelArg::PtrWriteOnly(outBlob));
                if (!softmax_kernel.run(1, &nanchors, NULL, false))
                    return false;
            }

            if (nmsThreshold > 0) {
                Mat mat = outBlob.getMat(ACCESS_WRITE);
                float *dstData = mat.ptr<float>();
                for (int b = 0; b < batch_size; ++b)
                    do_nms_sort(dstData + b*sample_size, rows*cols*anchors, thresh, nmsThreshold);
            }

        }

        return true;
    }
#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        if (inputs_arr.depth() == CV_16F)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs, internals;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        internals_arr.getMatVector(internals);

        CV_Assert(inputs.size() >= 1);
        CV_Assert(outputs.size() == 1);
        int const cell_size = classes + coords + 1;

        const float* biasData = blobs[0].ptr<float>();

        for (size_t ii = 0; ii < outputs.size(); ii++)
        {
            Mat &inpBlob = inputs[ii];
            Mat &outBlob = outputs[ii];

            int batch_size = inpBlob.size[0];
            int rows = inpBlob.size[1];
            int cols = inpBlob.size[2];

            // address length for one image in batch, both for input and output
            int sample_size = cell_size*rows*cols*anchors;

            // assert that the comment above is true
            CV_Assert(sample_size*batch_size == inpBlob.total());
            CV_Assert(sample_size*batch_size == outBlob.total());

            CV_Assert(inputs.size() < 2 || inputs[1].dims == 4);
            int hNorm = inputs.size() > 1 ? inputs[1].size[2] : rows;
            int wNorm = inputs.size() > 1 ? inputs[1].size[3] : cols;

            const float *srcData = inpBlob.ptr<float>();
            float *dstData = outBlob.ptr<float>();

            if (new_coords == 0) {
                // logistic activation for t0, for each grid cell (X x Y x Anchor-index)
                for (int i = 0; i < batch_size*rows*cols*anchors; ++i) {
                    int index = cell_size*i;
                    float x = srcData[index + 4];
                    dstData[index + 4] = logistic_activate(x);	// logistic activation
                }

                if (useSoftmax) {  // Yolo v2
                    Mat _inpBlob = inpBlob.reshape(0, outBlob.dims, outBlob.size);
                    softmax(outBlob, _inpBlob, -1, 5, classes);
                }
                else if (useLogistic) {  // Yolo v3
                    for (int i = 0; i < batch_size*rows*cols*anchors; ++i){
                        int index = cell_size*i;
                        const float* input = srcData + index + 5;
                        float* output = dstData + index + 5;
                        for (int c = 0; c < classes; ++c)
                            output[c] = logistic_activate(input[c]);
                    }
                }
            }
            for (int b = 0; b < batch_size; ++b)
                for (int x = 0; x < cols; ++x)
                    for(int y = 0; y < rows; ++y)
                        for (int a = 0; a < anchors; ++a) {
                            // relative start address for image b within the batch data
                            int index_sample_offset = sample_size*b;
                            int index = (y*cols + x)*anchors + a;  // index for each grid-cell & anchor
                            int p_index = index_sample_offset + index * cell_size + 4;
                            float scale = dstData[p_index];
                            if (classfix == -1 && scale < .5)
                            {
                                scale = 0;  // if(t0 < 0.5) t0 = 0;
                            }
                            int box_index = index_sample_offset + index * cell_size;

                            if (new_coords == 1) {
                                float x_tmp = (srcData[box_index + 0] - 0.5f) * scale_x_y + 0.5f;
                                float y_tmp = (srcData[box_index + 1] - 0.5f) * scale_x_y + 0.5f;
                                dstData[box_index + 0] = (x + x_tmp) / cols;
                                dstData[box_index + 1] = (y + y_tmp) / rows;
                                dstData[box_index + 2] = (srcData[box_index + 2]) * (srcData[box_index + 2]) * 4 * biasData[2 * a] / wNorm;
                                dstData[box_index + 3] = (srcData[box_index + 3]) * (srcData[box_index + 3]) * 4 * biasData[2 * a + 1] / hNorm;
                                dstData[box_index + 4] = srcData[p_index];

                                scale = srcData[p_index];
                                if (classfix == -1 && scale < thresh)
                                {
                                    scale = 0;  // if(t0 < 0.5) t0 = 0;
                                }

                                int class_index = index_sample_offset + index * cell_size + 5;
                                for (int j = 0; j < classes; ++j) {
                                    float prob = scale*srcData[class_index + j];  // prob = IoU(box, object) = t0 * class-probability
                                    dstData[class_index + j] = (prob > thresh) ? prob : 0;  // if (IoU < threshold) IoU = 0;
                                }
                            }
                            else
                            {
                                float x_tmp = (logistic_activate(srcData[box_index + 0]) - 0.5f) * scale_x_y + 0.5f;
                                float y_tmp = (logistic_activate(srcData[box_index + 1]) - 0.5f) * scale_x_y + 0.5f;
                                dstData[box_index + 0] = (x + x_tmp) / cols;
                                dstData[box_index + 1] = (y + y_tmp) / rows;
                                dstData[box_index + 2] = exp(srcData[box_index + 2]) * biasData[2 * a] / wNorm;
                                dstData[box_index + 3] = exp(srcData[box_index + 3]) * biasData[2 * a + 1] / hNorm;

                                int class_index = index_sample_offset + index * cell_size + 5;
                                for (int j = 0; j < classes; ++j) {
                                    float prob = scale*dstData[class_index + j];  // prob = IoU(box, object) = t0 * class-probability
                                    dstData[class_index + j] = (prob > thresh) ? prob : 0;  // if (IoU < threshold) IoU = 0;
                                }
                            }
                        }
            if (nmsThreshold > 0) {
                for (int b = 0; b < batch_size; ++b){
                    do_nms_sort(dstData+b*sample_size, rows*cols*anchors, thresh, nmsThreshold);
                }
            }
        }
    }

    void do_nms_sort(float *detections, int total, float score_thresh, float nms_thresh)
    {
        std::vector<Rect2d> boxes(total);
        std::vector<float> scores(total);

        for (int i = 0; i < total; ++i)
        {
            Rect2d &b = boxes[i];
            int box_index = i * (classes + coords + 1);
            b.width = detections[box_index + 2];
            b.height = detections[box_index + 3];
            b.x = detections[box_index + 0] - b.width / 2;
            b.y = detections[box_index + 1] - b.height / 2;
        }

        std::vector<int> indices;
        for (int k = 0; k < classes; ++k)
        {
            for (int i = 0; i < total; ++i)
            {
                int box_index = i * (classes + coords + 1);
                int class_index = box_index + 5;
                scores[i] = detections[class_index + k];
                detections[class_index + k] = 0;
            }
            NMSBoxes(boxes, scores, score_thresh, nms_thresh, indices);
            for (int i = 0, n = indices.size(); i < n; ++i)
            {
                int box_index = indices[i] * (classes + coords + 1);
                int class_index = box_index + 5;
                detections[class_index + k] = scores[indices[i]];
            }
        }
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(
        void *context_,
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs
    ) override
    {
        auto context = reinterpret_cast<csl::CSLContext*>(context_);

        if (coords != 4)
            CV_Error(Error::StsNotImplemented, "Only upright rectangular boxes are supported in RegionLayer.");

        std::size_t height_norm, width_norm;
        if (inputs.size() == 1)
        {
            auto input_wrapper = inputs[0].dynamicCast<CUDABackendWrapper>();
            auto input_shape = input_wrapper->getShape();
            height_norm = input_shape[1];
            width_norm = input_shape[2];
        }
        else
        {
            auto input_wrapper = inputs[1].dynamicCast<CUDABackendWrapper>();
            auto input_shape = input_wrapper->getShape();
            CV_Assert(input_shape.size() == 4);
            height_norm = input_shape[2];
            width_norm = input_shape[3];
        }

        cuda4dnn::SquashMethod squash_method;
        if(useLogistic)
            squash_method = cuda4dnn::SquashMethod::SIGMOID;
        else if (useSoftmax)
            squash_method = cuda4dnn::SquashMethod::SOFTMAX;

        /* exactly one must be true */
        CV_Assert((useLogistic || useSoftmax) && !(useLogistic && useSoftmax));

        cuda4dnn::RegionConfiguration<float> config;
        config.squash_method = squash_method;
        config.classes = classes;
        config.boxes_per_cell = anchors;

        config.height_norm = height_norm;
        config.width_norm = width_norm;

        config.scale_x_y = scale_x_y;

        config.object_prob_cutoff = (classfix == -1) ? thresh : 0.f;
        config.class_prob_cutoff = thresh;

        config.nms_iou_threshold = nmsThreshold;

        config.new_coords = (new_coords == 1);
        return make_cuda_node<cuda4dnn::RegionOp>(preferableTarget, std::move(context->stream), blobs[0], config);
    }
#endif

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_UNUSED(outputs); // suppress unused variable warning

        int64 flops = 0;
        for(int i = 0; i < inputs.size(); i++)
        {
            flops += 60*total(inputs[i]);
        }
        return flops;
    }

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> > &inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto& input = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        auto parent_shape = input.get_shape();
        int64_t b = parent_shape[0];
        int64_t h = parent_shape[1];
        int64_t w = parent_shape[2];
        int64_t c = parent_shape[3];

        int64_t cols = b * h * w * anchors;
        int64_t rows = c / anchors;
        auto shape_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2},  std::vector<int64_t>{cols, rows});
        auto tr_axes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 0});

        std::shared_ptr<ov::Node> input2d;
        {
            input2d = std::make_shared<ov::op::v1::Reshape>(input, shape_node, true);
            input2d = std::make_shared<ov::op::v1::Transpose>(input2d, tr_axes);
        }

        std::shared_ptr<ov::Node> region;
        {
            auto new_axes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
            auto tr_input = std::make_shared<ov::op::v1::Transpose>(input, new_axes);

            std::vector<float> anchors_vec(blobs[0].ptr<float>(), blobs[0].ptr<float>() + blobs[0].total());
            std::vector<int64_t> mask(anchors, 1);
            region = std::make_shared<ov::op::v0::RegionYolo>(tr_input, coords, classes, anchors, useSoftmax, mask, 1, 3, anchors_vec);

            auto tr_shape = tr_input->get_shape();
            auto shape_as_inp = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                       ov::Shape{tr_shape.size()},
                                                                       std::vector<int64_t>(tr_shape.begin(), tr_shape.end()));

            region = std::make_shared<ov::op::v1::Reshape>(region, shape_as_inp, true);
            new_axes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 3, 1});
            region = std::make_shared<ov::op::v1::Transpose>(region, new_axes);

            region = std::make_shared<ov::op::v1::Reshape>(region, shape_node, true);
            region = std::make_shared<ov::op::v1::Transpose>(region, tr_axes);
        }

        auto strides = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 1});
        std::vector<int64_t> boxes_shape{b, anchors, h, w};
        auto shape_3d = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{boxes_shape.size()}, boxes_shape.data());

        ov::Shape box_broad_shape{1, (size_t)anchors, (size_t)h, (size_t)w};
        auto scale_x_y_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, &scale_x_y);
        auto shift_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{0.5});

        auto axis = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{}, {0});
        auto splits = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{5}, {1, 1, 1, 1, rows - 4});
        auto split = std::make_shared<ov::op::v1::VariadicSplit>(input2d, axis, splits);
        std::shared_ptr<ov::Node> box_x;
        {
            box_x = std::make_shared<ov::op::v0::Sigmoid>(split->output(0));
            box_x = std::make_shared<ov::op::v1::Subtract>(box_x, shift_node, ov::op::AutoBroadcastType::NUMPY);
            box_x = std::make_shared<ov::op::v1::Multiply>(box_x, scale_x_y_node, ov::op::AutoBroadcastType::NUMPY);
            box_x = std::make_shared<ov::op::v1::Add>(box_x, shift_node, ov::op::AutoBroadcastType::NUMPY);
            box_x = std::make_shared<ov::op::v1::Reshape>(box_x, shape_3d, true);

            std::vector<float> x_indices(w * h * anchors);
            auto begin = x_indices.begin();
            for (int i = 0; i < w; i++)
            {
                std::fill(begin + i * anchors, begin + (i + 1) * anchors, i);
            }

            for (int j = 1; j < h; j++)
            {
                std::copy(begin, begin + w * anchors, begin + j * w * anchors);
            }
            auto horiz = std::make_shared<ov::op::v0::Constant>(ov::element::f32, box_broad_shape, x_indices.data());
            box_x = std::make_shared<ov::op::v1::Add>(box_x, horiz, ov::op::AutoBroadcastType::NUMPY);

            auto cols_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{float(w)});
            box_x = std::make_shared<ov::op::v1::Divide>(box_x, cols_node, ov::op::AutoBroadcastType::NUMPY);
        }

        std::shared_ptr<ov::Node> box_y;
        {
            box_y = std::make_shared<ov::op::v0::Sigmoid>(split->output(1));
            box_y = std::make_shared<ov::op::v1::Subtract>(box_y, shift_node, ov::op::AutoBroadcastType::NUMPY);
            box_y = std::make_shared<ov::op::v1::Multiply>(box_y, scale_x_y_node, ov::op::AutoBroadcastType::NUMPY);
            box_y = std::make_shared<ov::op::v1::Add>(box_y, shift_node, ov::op::AutoBroadcastType::NUMPY);
            box_y = std::make_shared<ov::op::v1::Reshape>(box_y, shape_3d, true);

            std::vector<float> y_indices(h * anchors);
            for (int i = 0; i < h; i++)
            {
                std::fill(y_indices.begin() + i * anchors, y_indices.begin() + (i + 1) * anchors, i);
            }

            auto vert = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, (size_t)anchors, (size_t)h, 1}, y_indices.data());
            box_y = std::make_shared<ov::op::v1::Add>(box_y, vert, ov::op::AutoBroadcastType::NUMPY);
            auto rows_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{float(h)});
            box_y = std::make_shared<ov::op::v1::Divide>(box_y, rows_node, ov::op::AutoBroadcastType::NUMPY);
        }

        std::shared_ptr<ov::Node> box_w, box_h;
        {
            int hNorm, wNorm;
            if (nodes.size() > 1)
            {
                auto node_1_shape = nodes[1].dynamicCast<InfEngineNgraphNode>()->node.get_shape();
                hNorm = node_1_shape[2];
                wNorm = node_1_shape[3];
            }
            else
            {
                hNorm = h;
                wNorm = w;
            }

            std::vector<float> anchors_w(anchors), anchors_h(anchors);
            for (size_t a = 0; a < anchors; ++a)
            {
                anchors_w[a] = blobs[0].at<float>(0, 2 * a) / wNorm;
                anchors_h[a] = blobs[0].at<float>(0, 2 * a + 1) / hNorm;
            }

            std::vector<float> bias_w(w * h * anchors), bias_h(w * h * anchors);
            for (int j = 0; j < h; j++)
            {
                std::copy(anchors_w.begin(), anchors_w.end(), bias_w.begin() + j * anchors);
                std::copy(anchors_h.begin(), anchors_h.end(), bias_h.begin() + j * anchors);
            }

            for (int i = 1; i < w; i++)
            {
                std::copy(bias_w.begin(), bias_w.begin() + h * anchors, bias_w.begin() + i * h * anchors);
                std::copy(bias_h.begin(), bias_h.begin() + h * anchors, bias_h.begin() + i * h * anchors);
            }

            box_w = std::make_shared<ov::op::v0::Exp>(split->output(2));
            box_w = std::make_shared<ov::op::v1::Reshape>(box_w, shape_3d, true);
            auto anchor_w_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, box_broad_shape, bias_w.data());
            box_w = std::make_shared<ov::op::v1::Multiply>(box_w, anchor_w_node, ov::op::AutoBroadcastType::NUMPY);

            box_h = std::make_shared<ov::op::v0::Exp>(split->output(3));
            box_h = std::make_shared<ov::op::v1::Reshape>(box_h, shape_3d, true);
            auto anchor_h_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, box_broad_shape, bias_h.data());
            box_h = std::make_shared<ov::op::v1::Multiply>(box_h, anchor_h_node, ov::op::AutoBroadcastType::NUMPY);
        }

        auto region_splits = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{3}, {4, 1, rows - 5});
        auto region_split = std::make_shared<ov::op::v1::VariadicSplit>(region, axis, region_splits);

        std::shared_ptr<ov::Node> scale;
        {
            float thr = classfix == -1 ? 0.5 : 0;
            auto thresh_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{thr});
            auto mask = std::make_shared<ov::op::v1::Less>(region_split->output(1), thresh_node);
            auto zero_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, mask->get_shape(), std::vector<float>(cols, 0));
            scale = std::make_shared<ov::op::v1::Select>(mask, zero_node, region_split->output(1));
        }

        std::shared_ptr<ov::Node> probs;
        {
            probs = std::make_shared<ov::op::v1::Multiply>(region_split->output(2), scale, ov::op::AutoBroadcastType::NUMPY);
            auto thresh_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, &thresh);
            auto mask = std::make_shared<ov::op::v1::Greater>(probs, thresh_node);
            auto zero_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, mask->get_shape(), std::vector<float>((rows - 5) * cols, 0));
            probs = std::make_shared<ov::op::v1::Select>(mask, probs, zero_node);
        }


        auto concat_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, cols});
        box_x = std::make_shared<ov::op::v1::Reshape>(box_x, concat_shape, true);
        box_y = std::make_shared<ov::op::v1::Reshape>(box_y, concat_shape, true);
        box_w = std::make_shared<ov::op::v1::Reshape>(box_w, concat_shape, true);
        box_h = std::make_shared<ov::op::v1::Reshape>(box_h, concat_shape, true);

        ov::NodeVector inp_nodes{box_x, box_y, box_w, box_h, scale, probs};
        std::shared_ptr<ov::Node> result = std::make_shared<ov::op::v0::Concat>(inp_nodes, 0);
        result = std::make_shared<ov::op::v1::Transpose>(result, tr_axes);
        if (b > 1)
        {
            std::vector<int64_t> sizes{b, static_cast<int64_t>(result->get_shape()[0]) / b, static_cast<int64_t>(result->get_shape()[1])};
            auto shape_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{sizes.size()}, sizes.data());
            result = std::make_shared<ov::op::v1::Reshape>(result, shape_node, true);
        }

        return Ptr<BackendNode>(new InfEngineNgraphNode(result));
    }
#endif  // HAVE_DNN_NGRAPH

};

Ptr<RegionLayer> RegionLayer::create(const LayerParams& params)
{
    return Ptr<RegionLayer>(new RegionLayerImpl(params));
}

}  // namespace dnn
}  // namespace cv
