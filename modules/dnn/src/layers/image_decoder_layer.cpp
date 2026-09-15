// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the
// top-level directory of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#ifdef HAVE_OPENCV_IMGCODECS
#include <opencv2/imgcodecs.hpp>
#endif

namespace cv {
namespace dnn {

// ONNX ImageDecoder operator
// Spec: https://onnx.ai/onnx/operators/onnx__ImageDecoder.html
// Supported opsets: 20

class ImageDecoderLayerImpl CV_FINAL : public ImageDecoderLayer
{
public:
    enum PixelFormat { PF_RGB, PF_BGR, PF_GRAYSCALE };
    PixelFormat pixelFormat;

    ImageDecoderLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        String pf = params.get<String>("pixel_format", "RGB");
        if (pf == "RGB")
            pixelFormat = PF_RGB;
        else if (pf == "BGR")
            pixelFormat = PF_BGR;
        else if (pf == "Grayscale")
            pixelFormat = PF_GRAYSCALE;
        else
            CV_Error_(Error::StsBadArg, ("DNN/ImageDecoder: unsupported pixel_format '%s'", pf.c_str()));
    }

    virtual bool dynamicOutputShapes() const CV_OVERRIDE
    {
        return true;
    }

    bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs, const int /*requiredOutputs*/,
                          std::vector<MatShape>& outputs, std::vector<MatShape>& /*internals*/) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
        int channels = pixelFormat == PF_GRAYSCALE ? 1 : 3;
        MatShape out({-1, -1, channels});
        outputs.assign(1, out);
        return false;
    }

    void getTypes(const std::vector<MatType>& /*inputs*/, const int requiredOutputs,
                  const int /*requiredInternals*/, std::vector<MatType>& outputs,
                  std::vector<MatType>& /*internals*/) const CV_OVERRIDE
    {
        outputs.assign(requiredOutputs, CV_8U);
    }

    void forward(InputArrayOfArrays in_arr, OutputArrayOfArrays out_arr, OutputArrayOfArrays) CV_OVERRIDE
    {
        CV_Assert(in_arr.size().area() == 1);
        Mat encoded = in_arr.getMat(0);
        CV_CheckTypeEQ(encoded.type(), CV_8UC1, "DNN/ImageDecoder: input must be a uint8 tensor");
        CV_Assert(encoded.isContinuous());

#ifndef HAVE_OPENCV_IMGCODECS
        CV_UNUSED(out_arr);
        CV_Error(Error::StsNotImplemented, "DNN/ImageDecoder: OpenCV was built without imgcodecs support");
#else
        Mat buf(1, (int)encoded.total(), CV_8UC1, encoded.data);
        Mat decoded;
        if (pixelFormat == PF_GRAYSCALE)
        {
            Mat color = imdecode(buf, IMREAD_COLOR_BGR);
            cvtColor(color, decoded, COLOR_BGR2GRAY);
        }
        else
        {
            decoded = imdecode(buf, pixelFormat == PF_RGB ? IMREAD_COLOR_RGB : IMREAD_COLOR_BGR);
        }

        const int channels = pixelFormat == PF_GRAYSCALE ? 1 : 3;
        MatShape outShape({decoded.rows, decoded.cols, channels});

        // Zero-copy reshape; ternary guards the empty case since reshape() throws on it.
        Mat Y = decoded.empty() ? Mat(outShape, CV_8U) : decoded.reshape(1, outShape);

        auto kind = out_arr.kind();
        if (kind == _InputArray::STD_VECTOR_MAT)
        {
            std::vector<Mat>& out_mats = out_arr.getMatVecRef();
            out_mats.resize(1);
            out_mats[0] = Y;
        }
        else
        {
            CV_Assert(kind == _InputArray::STD_VECTOR_UMAT);
            std::vector<UMat>& out_umats = out_arr.getUMatVecRef();
            out_umats.resize(1);
            Y.copyTo(out_umats[0]);
        }
#endif
    }
};

Ptr<ImageDecoderLayer> ImageDecoderLayer::create(const LayerParams& params)
{
    return makePtr<ImageDecoderLayerImpl>(params);
}

}} // namespace cv::dnn
