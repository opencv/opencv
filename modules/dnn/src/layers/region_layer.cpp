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
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <iostream>
#include "opencl_kernels_dnn.hpp"

namespace cv
{
namespace dnn
{

class RegionLayerImpl : public RegionLayer
{
public:
    int coords, classes, anchors, classfix;
    float thresh, nmsThreshold;
    bool useSoftmaxTree, useSoftmax;

    RegionLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        CV_Assert(blobs.size() == 1);

        thresh = params.get<float>("thresh", 0.2);
        coords = params.get<int>("coords", 4);
        classes = params.get<int>("classes", 0);
        anchors = params.get<int>("anchors", 5);
        classfix = params.get<int>("classfix", 0);
        useSoftmaxTree = params.get<bool>("softmax_tree", false);
        useSoftmax = params.get<bool>("softmax", false);
        nmsThreshold = params.get<float>("nms_threshold", 0.4);

        CV_Assert(nmsThreshold >= 0.);
        CV_Assert(coords == 4);
        CV_Assert(classes >= 1);
        CV_Assert(anchors >= 1);
        CV_Assert(useSoftmaxTree || useSoftmax);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const
    {
        CV_Assert(inputs.size() > 0);
        CV_Assert(inputs[0][3] == (1 + coords + classes)*anchors);
        outputs = std::vector<MatShape>(inputs.size(), shape(inputs[0][1] * inputs[0][2] * anchors, inputs[0][3] / anchors));
        return false;
    }

    virtual bool supportBackend(int backendId)
    {
        return backendId == DNN_BACKEND_DEFAULT;
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
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        inps.getUMatVector(inputs);
        outs.getUMatVector(outputs);

        if (useSoftmaxTree) {   // Yolo 9000
            CV_Error(cv::Error::StsNotImplemented, "Yolo9000 is not implemented");
            return false;
        }

        CV_Assert(inputs.size() >= 1);
        int const cell_size = classes + coords + 1;
        UMat blob_umat = blobs[0].getUMat(ACCESS_READ);

        for (size_t ii = 0; ii < outputs.size(); ii++)
        {
            UMat& inpBlob = inputs[ii];
            UMat& outBlob = outputs[ii];

            int rows = inpBlob.size[1];
            int cols = inpBlob.size[2];

            ocl::Kernel logistic_kernel("logistic_activ", ocl::dnn::region_oclsrc);
            size_t global = rows*cols*anchors;
            logistic_kernel.set(0, (int)global);
            logistic_kernel.set(1, ocl::KernelArg::PtrReadOnly(inpBlob));
            logistic_kernel.set(2, (int)cell_size);
            logistic_kernel.set(3, ocl::KernelArg::PtrWriteOnly(outBlob));
            logistic_kernel.run(1, &global, NULL, false);

            if (useSoftmax)
            {
                // Yolo v2
                // softmax activation for Probability, for each grid cell (X x Y x Anchor-index)
                ocl::Kernel softmax_kernel("softmax_activ", ocl::dnn::region_oclsrc);
                size_t nthreads = rows*cols*anchors;
                softmax_kernel.set(0, (int)nthreads);
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
                if (!softmax_kernel.run(1, &nthreads, NULL, false))
                    return false;
            }

            if (nmsThreshold > 0) {
                Mat mat = outBlob.getMat(ACCESS_WRITE);
                float *dstData = mat.ptr<float>();
                do_nms_sort(dstData, rows*cols*anchors, nmsThreshold);
                //do_nms(dstData, rows*cols*anchors, nmsThreshold);
            }

        }

        return true;
    }
#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN((preferableTarget == DNN_TARGET_OPENCL) &&
                   OCL_PERFORMANCE_CHECK(ocl::Device::getDefault().isIntel()),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        Layer::forward_fallback(inputs_arr, outputs_arr, internals_arr);
    }

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_Assert(inputs.size() >= 1);
        int const cell_size = classes + coords + 1;

        const float* biasData = blobs[0].ptr<float>();

        for (size_t ii = 0; ii < outputs.size(); ii++)
        {
            Mat &inpBlob = *inputs[ii];
            Mat &outBlob = outputs[ii];

            int rows = inpBlob.size[1];
            int cols = inpBlob.size[2];

            const float *srcData = inpBlob.ptr<float>();
            float *dstData = outBlob.ptr<float>();

            // logistic activation for t0, for each grid cell (X x Y x Anchor-index)
            for (int i = 0; i < rows*cols*anchors; ++i) {
                int index = cell_size*i;
                float x = srcData[index + 4];
                dstData[index + 4] = logistic_activate(x);	// logistic activation
            }

            if (useSoftmaxTree) {   // Yolo 9000
                CV_Error(cv::Error::StsNotImplemented, "Yolo9000 is not implemented");
            }
            else if (useSoftmax) {  // Yolo v2
                // softmax activation for Probability, for each grid cell (X x Y x Anchor-index)
                for (int i = 0; i < rows*cols*anchors; ++i) {
                    int index = cell_size*i;
                    softmax_activate(srcData + index + 5, classes, 1, dstData + index + 5);
                }

                for (int x = 0; x < cols; ++x)
                    for(int y = 0; y < rows; ++y)
                        for (int a = 0; a < anchors; ++a) {
                            int index = (y*cols + x)*anchors + a;	// index for each grid-cell & anchor
                            int p_index = index * cell_size + 4;
                            float scale = dstData[p_index];
                            if (classfix == -1 && scale < .5) scale = 0;	// if(t0 < 0.5) t0 = 0;
                            int box_index = index * cell_size;

                            dstData[box_index + 0] = (x + logistic_activate(srcData[box_index + 0])) / cols;
                            dstData[box_index + 1] = (y + logistic_activate(srcData[box_index + 1])) / rows;
                            dstData[box_index + 2] = exp(srcData[box_index + 2]) * biasData[2 * a] / cols;
                            dstData[box_index + 3] = exp(srcData[box_index + 3]) * biasData[2 * a + 1] / rows;

                            int class_index = index * cell_size + 5;

                            if (useSoftmaxTree) {
                                CV_Error(cv::Error::StsNotImplemented, "Yolo9000 is not implemented");
                            }
                            else {
                                for (int j = 0; j < classes; ++j) {
                                    float prob = scale*dstData[class_index + j];	// prob = IoU(box, object) = t0 * class-probability
                                    dstData[class_index + j] = (prob > thresh) ? prob : 0;		// if (IoU < threshold) IoU = 0;
                                }
                            }
                        }

            }

            if (nmsThreshold > 0) {
                do_nms_sort(dstData, rows*cols*anchors, nmsThreshold);
                //do_nms(dstData, rows*cols*anchors, nmsThreshold);
            }

        }
    }


    struct box {
        float x, y, w, h;
        float *probs;
    };

    float overlap(float x1, float w1, float x2, float w2)
    {
        float l1 = x1 - w1 / 2;
        float l2 = x2 - w2 / 2;
        float left = l1 > l2 ? l1 : l2;
        float r1 = x1 + w1 / 2;
        float r2 = x2 + w2 / 2;
        float right = r1 < r2 ? r1 : r2;
        return right - left;
    }

    float box_intersection(box a, box b)
    {
        float w = overlap(a.x, a.w, b.x, b.w);
        float h = overlap(a.y, a.h, b.y, b.h);
        if (w < 0 || h < 0) return 0;
        float area = w*h;
        return area;
    }

    float box_union(box a, box b)
    {
        float i = box_intersection(a, b);
        float u = a.w*a.h + b.w*b.h - i;
        return u;
    }

    float box_iou(box a, box b)
    {
        return box_intersection(a, b) / box_union(a, b);
    }

    struct sortable_bbox {
        int index;
        float *probs;
    };

    struct nms_comparator {
        int k;
        nms_comparator(int _k) : k(_k) {}
        bool operator ()(sortable_bbox v1, sortable_bbox v2) {
            return v2.probs[k] < v1.probs[k];
        }
    };

    void do_nms_sort(float *detections, int total, float nms_thresh)
    {
        std::vector<box> boxes(total);
        for (int i = 0; i < total; ++i) {
            box &b = boxes[i];
            int box_index = i * (classes + coords + 1);
            b.x = detections[box_index + 0];
            b.y = detections[box_index + 1];
            b.w = detections[box_index + 2];
            b.h = detections[box_index + 3];
            int class_index = i * (classes + 5) + 5;
            b.probs = (detections + class_index);
        }

        std::vector<sortable_bbox> s(total);

        for (int i = 0; i < total; ++i) {
            s[i].index = i;
            int class_index = i * (classes + 5) + 5;
            s[i].probs = (detections + class_index);
        }

        for (int k = 0; k < classes; ++k) {
            std::stable_sort(s.begin(), s.end(), nms_comparator(k));
            for (int i = 0; i < total; ++i) {
                if (boxes[s[i].index].probs[k] == 0) continue;
                box a = boxes[s[i].index];
                for (int j = i + 1; j < total; ++j) {
                    box b = boxes[s[j].index];
                    if (box_iou(a, b) > nms_thresh) {
                        boxes[s[j].index].probs[k] = 0;
                    }
                }
            }
        }
    }

    void do_nms(float *detections, int total, float nms_thresh)
    {
        std::vector<box> boxes(total);
        for (int i = 0; i < total; ++i) {
            box &b = boxes[i];
            int box_index = i * (classes + coords + 1);
            b.x = detections[box_index + 0];
            b.y = detections[box_index + 1];
            b.w = detections[box_index + 2];
            b.h = detections[box_index + 3];
            int class_index = i * (classes + 5) + 5;
            b.probs = (detections + class_index);
        }

        for (int i = 0; i < total; ++i) {
            bool any = false;
            for (int k = 0; k < classes; ++k) any = any || (boxes[i].probs[k] > 0);
            if (!any) {
                continue;
            }
            for (int j = i + 1; j < total; ++j) {
                if (box_iou(boxes[i], boxes[j]) > nms_thresh) {
                    for (int k = 0; k < classes; ++k) {
                        if (boxes[i].probs[k] < boxes[j].probs[k]) boxes[i].probs[k] = 0;
                        else boxes[j].probs[k] = 0;
                    }
                }
            }
        }
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const
    {
        (void)outputs; // suppress unused variable warning

        int64 flops = 0;
        for(int i = 0; i < inputs.size(); i++)
        {
            flops += 60*total(inputs[i]);
        }
        return flops;
    }
};

Ptr<RegionLayer> RegionLayer::create(const LayerParams& params)
{
    return Ptr<RegionLayer>(new RegionLayerImpl(params));
}

}  // namespace dnn
}  // namespace cv
