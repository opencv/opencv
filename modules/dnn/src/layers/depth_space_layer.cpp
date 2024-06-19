// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>

// OpenCL backend
#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
#endif

namespace cv { namespace dnn {

class DepthSpaceLayerImpl CV_FINAL : public DepthSpaceLayer {
public:

    DepthSpaceLayerImpl(const LayerParams &params) {
        setParamsFrom(params);

        auto op_type = params.get<std::string>("op_type");
        auto mode = params.get<std::string>("mode", "DCR");
        CV_CheckFalse(op_type.empty(), "DepthSpaceLayer: op_type cannot be empty");
        if (op_type == "DepthToSpace") {
            if (mode == "DCR") {
                op = OPERATION::DEPTH_TO_SPACE_DCR;
                permutation = {0, 3, 4, 1, 5, 2};
            } else if (mode == "CRD") {
                op = OPERATION::DEPTH_TO_SPACE_CRD;
                permutation = {0, 1, 4, 2, 5, 3};
            } else {
                CV_Error(Error::StsBadArg, cv::format("DepthSpaceLayer: unsupported mode %s\n", mode.c_str()));
            }
        } else if (op_type == "SpaceToDepth") {
            op = OPERATION::SPACE_TO_DPETH;
            permutation = {0, 3, 5, 1, 2, 4};
        } else {
            CV_Error(Error::StsBadArg, cv::format("DepthSpaceLayer: unsupported op_type %s\n", op_type.c_str()));
        }

        CV_CheckTrue(params.has("blocksize"), "DepthSpaceLayer: blocksize is required");
        blocksize = params.get<int>("blocksize");
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        // TODO: support other backends
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE {
        CV_CheckEQ(inputs.size(), static_cast<size_t>(1), "DepthSpaceLayer: accepts only one input");
        const auto &input = inputs.front();
        CV_CheckEQ(input.size(), static_cast<size_t>(4), "DepthSpaceLayer: input needs to be 4-dimensional [N, C, H, W]");
        int batch = input[0], input_depth = input[1], input_height = input[2], input_width = input[3];
        int output_depth = -1, output_height = -1, output_width = -1;
        if (op == OPERATION::DEPTH_TO_SPACE_DCR || op == OPERATION::DEPTH_TO_SPACE_CRD) {
            CV_CheckEQ(input_depth % (blocksize * blocksize), 0, "DepthSpaceLayer: DepthToSpace requires input depth to be a multiple of (blocksize * blocksize)");
            output_depth = input_depth / blocksize / blocksize;
            output_height = input_height * blocksize;
            output_width = input_width * blocksize;
        } else { // SPACE_TO_DEPTH
            CV_CheckEQ(input_height % blocksize, 0, "DepthSpaceLayer: SpaceToDepth requires input height to be a multiple of blocksize");
            CV_CheckEQ(input_width % blocksize, 0, "DepthSpaceLayer: SpaceToDepth requires input width to be a multiple of blocksize");
            output_depth = input_depth * blocksize * blocksize;
            output_height = input_height / blocksize;
            output_width = input_width / blocksize;
        }
        outputs.assign(1, MatShape{batch, output_depth, output_height, output_width});
        return false;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE {
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);

        auto input_shape = shape(inputs.front());
        int batch = input_shape[0], input_depth = input_shape[1], input_height = input_shape[2], input_width = input_shape[3];
        if (op == OPERATION::DEPTH_TO_SPACE_DCR) {
            internal_shape = MatShape{batch, blocksize, blocksize, input_depth / (blocksize * blocksize), input_height, input_width};
        } else if (op == OPERATION::DEPTH_TO_SPACE_CRD) {
            internal_shape = MatShape{batch, input_depth / (blocksize * blocksize), blocksize, blocksize, input_height, input_width};
        } else { // SPACE_TO_DEPTH
            internal_shape = MatShape{batch, input_depth, input_height / blocksize, blocksize, input_width / blocksize, blocksize};
        }

        transposed_internal_shape = MatShape(internal_shape.size());
        for (size_t i = 0; i < permutation.size(); i++) {
            transposed_internal_shape[i] = internal_shape[permutation[i]];
        }

#ifdef HAVE_OPENCL
        umat_permutation.release();
        umat_internal_strides.release();
        umat_transposed_internal_strides.release();
#endif
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        // TODO: support 8-bit int in permute kernel
        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget) && inputs_arr.depth() != CV_8S,
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        if (inputs_arr.depth() == CV_16F) {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const auto output_shape = shape(outputs.front());
        Mat tmp;
        cv::transposeND(inputs[0].reshape(1, internal_shape), permutation, tmp);
        tmp.reshape(1, output_shape).copyTo(outputs[0]);
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) {
        std::vector<UMat> inputs, outputs;

        inputs_arr.getUMatVector(inputs);
        outputs_arr.getUMatVector(outputs);

        if (umat_permutation.empty() || umat_internal_strides.empty() || umat_transposed_internal_strides.empty()) {
            Mat mat_permutation(1, permutation.size(), CV_32S, permutation.data());
            mat_permutation.copyTo(umat_permutation);

            std::vector<int> internal_strides(permutation.size(), 1), transposed_internal_stides(permutation.size(), 1);
            for (int i = static_cast<int>(permutation.size()) - 2; i >= 0; i--) {
                internal_strides[i] = internal_strides[i + 1] * internal_shape[i + 1];
                transposed_internal_stides[i] = transposed_internal_stides[i + 1] * transposed_internal_shape[i + 1];
            }
            Mat mat_internal_strides(1, internal_strides.size(), CV_32S, internal_strides.data());
            mat_internal_strides.copyTo(umat_internal_strides);
            Mat mat_transposed_internal_strides(1, transposed_internal_stides.size(), CV_32S, transposed_internal_stides.data());
            mat_transposed_internal_strides.copyTo(umat_transposed_internal_strides);
        }

        const auto output_shape = shape(outputs.front());
        UMat tmp = inputs.front().reshape(1, static_cast<int>(internal_shape.size()), internal_shape.data());

        bool use_half = (inputs_arr.depth() == CV_16F);
        std::string permute_options = cv::format("-DDtype=%s", use_half ? "half" : "float");
        ocl::Kernel permute_kernel("permute", ocl::dnn::permute_oclsrc, permute_options);
        if (permute_kernel.empty()) {
            return false;
        }
        UMat transposed_tmp(static_cast<int>(transposed_internal_shape.size()), transposed_internal_shape.data(), inputs_arr.depth());
        size_t num_element = static_cast<size_t>(std::accumulate(internal_shape.begin(), internal_shape.end(), 1, std::multiplies<int>()));
        permute_kernel.set(0, static_cast<int>(num_element));
        permute_kernel.set(1, ocl::KernelArg::PtrReadOnly(tmp));
        permute_kernel.set(2, ocl::KernelArg::PtrReadOnly(umat_permutation));
        permute_kernel.set(3, ocl::KernelArg::PtrReadOnly(umat_internal_strides));
        permute_kernel.set(4, ocl::KernelArg::PtrReadOnly(umat_transposed_internal_strides));
        permute_kernel.set(5, static_cast<int>(permutation.size()));
        permute_kernel.set(6, ocl::KernelArg::PtrWriteOnly(transposed_tmp));
        if (!permute_kernel.run(1, &num_element, NULL, false)) {
            return false;
        }

        transposed_tmp.reshape(1, static_cast<int>(output_shape.size()), output_shape.data()).copyTo(outputs.front());
        return true;
    }
#endif

private:
    enum class OPERATION {
        DEPTH_TO_SPACE_DCR = 0, // DepthToSpace, depth-colum-row order re-arrangement
        DEPTH_TO_SPACE_CRD, // DepthToSpace, column-row-depth roder re-arrangement
        SPACE_TO_DPETH,
    } op;
    int blocksize;

    MatShape internal_shape;
    MatShape transposed_internal_shape;
    std::vector<int> permutation;

#ifdef HAVE_OPENCL
    UMat umat_permutation;
    UMat umat_internal_strides;
    UMat umat_transposed_internal_strides;
#endif
};

Ptr<DepthSpaceLayer> DepthSpaceLayer::create(const LayerParams &params) {
    return makePtr<DepthSpaceLayerImpl>(params);
}

}} // namespace cv::dnn
