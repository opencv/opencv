#ifndef __OPENCV_SAMPLES_DNN_CUSTOM_LAYERS__
#define __OPENCV_SAMPLES_DNN_CUSTOM_LAYERS__

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>    // getPlane

//! [InterpLayer]
class InterpLayer : public cv::dnn::Layer
{
public:
    InterpLayer(const cv::dnn::LayerParams &params) : Layer(params)
    {
        outWidth = params.get<int>("width", 0);
        outHeight = params.get<int>("height", 0);
    }

    static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
    {
        return cv::Ptr<cv::dnn::Layer>(new InterpLayer(params));
    }

    virtual bool getMemoryShapes(const std::vector<std::vector<int> > &inputs,
                                 const int requiredOutputs,
                                 std::vector<std::vector<int> > &outputs,
                                 std::vector<std::vector<int> > &internals) const CV_OVERRIDE
    {
        CV_UNUSED(requiredOutputs); CV_UNUSED(internals);
        std::vector<int> outShape(4);
        outShape[0] = inputs[0][0];  // batch size
        outShape[1] = inputs[0][1];  // number of channels
        outShape[2] = outHeight;
        outShape[3] = outWidth;
        outputs.assign(1, outShape);
        return false;
    }

    // Implementation of this custom layer is based on https://github.com/cdmh/deeplab-public/blob/master/src/caffe/layers/interp_layer.cpp
    virtual void forward(cv::InputArrayOfArrays inputs_arr,
                         cv::OutputArrayOfArrays outputs_arr,
                         cv::OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        if (inputs_arr.depth() == CV_16S)
        {
            // In case of DNN_TARGET_OPENCL_FP16 target the following method
            // converts data from FP16 to FP32 and calls this forward again.
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<cv::Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        cv::Mat& inp = inputs[0];
        cv::Mat& out = outputs[0];
        const float* inpData = (float*)inp.data;
        float* outData = (float*)out.data;

        const int batchSize = inp.size[0];
        const int numChannels = inp.size[1];
        const int inpHeight = inp.size[2];
        const int inpWidth = inp.size[3];

        const float rheight = (outHeight > 1) ? static_cast<float>(inpHeight - 1) / (outHeight - 1) : 0.f;
        const float rwidth = (outWidth > 1) ? static_cast<float>(inpWidth - 1) / (outWidth - 1) : 0.f;
        for (int h2 = 0; h2 < outHeight; ++h2)
        {
            const float h1r = rheight * h2;
            const int h1 = static_cast<int>(h1r);
            const int h1p = (h1 < inpHeight - 1) ? 1 : 0;
            const float h1lambda = h1r - h1;
            const float h0lambda = 1.f - h1lambda;
            for (int w2 = 0; w2 < outWidth; ++w2)
            {
                const float w1r = rwidth * w2;
                const int w1 = static_cast<int>(w1r);
                const int w1p = (w1 < inpWidth - 1) ? 1 : 0;
                const float w1lambda = w1r - w1;
                const float w0lambda = 1.f - w1lambda;
                const float* pos1 = inpData + h1 * inpWidth + w1;
                float* pos2 = outData + h2 * outWidth + w2;
                for (int c = 0; c < batchSize * numChannels; ++c)
                {
                    pos2[0] =
                      h0lambda * (w0lambda * pos1[0] + w1lambda * pos1[w1p]) +
                      h1lambda * (w0lambda * pos1[h1p * inpWidth] + w1lambda * pos1[h1p * inpWidth + w1p]);
                    pos1 += inpWidth * inpHeight;
                    pos2 += outWidth * outHeight;
                }
            }
        }
    }

private:
    int outWidth, outHeight;
};
//! [InterpLayer]

//! [ResizeBilinearLayer]
class ResizeBilinearLayer CV_FINAL : public cv::dnn::Layer
{
public:
    ResizeBilinearLayer(const cv::dnn::LayerParams &params) : Layer(params)
    {
        CV_Assert(!params.get<bool>("align_corners", false));
        CV_Assert(!blobs.empty());

        for (size_t i = 0; i < blobs.size(); ++i)
            CV_Assert(blobs[i].type() == CV_32SC1);

        // There are two cases of input blob: a single blob which contains output
        // shape and two blobs with scaling factors.
        if (blobs.size() == 1)
        {
            CV_Assert(blobs[0].total() == 2);
            outHeight = blobs[0].at<int>(0, 0);
            outWidth = blobs[0].at<int>(0, 1);
            factorHeight = factorWidth = 0;
        }
        else
        {
            CV_Assert(blobs.size() == 2); CV_Assert(blobs[0].total() == 1); CV_Assert(blobs[1].total() == 1);
            factorHeight = blobs[0].at<int>(0, 0);
            factorWidth = blobs[1].at<int>(0, 0);
            outHeight = outWidth = 0;
        }
    }

    static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
    {
        return cv::Ptr<cv::dnn::Layer>(new ResizeBilinearLayer(params));
    }

    virtual bool getMemoryShapes(const std::vector<std::vector<int> > &inputs,
                                 const int,
                                 std::vector<std::vector<int> > &outputs,
                                 std::vector<std::vector<int> > &) const CV_OVERRIDE
    {
        std::vector<int> outShape(4);
        outShape[0] = inputs[0][0];  // batch size
        outShape[1] = inputs[0][1];  // number of channels
        outShape[2] = outHeight != 0 ? outHeight : (inputs[0][2] * factorHeight);
        outShape[3] = outWidth != 0 ? outWidth : (inputs[0][3] * factorWidth);
        outputs.assign(1, outShape);
        return false;
    }

    virtual void finalize(cv::InputArrayOfArrays, cv::OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        std::vector<cv::Mat> outputs;
        outputs_arr.getMatVector(outputs);
        if (!outWidth && !outHeight)
        {
            outHeight = outputs[0].size[2];
            outWidth = outputs[0].size[3];
        }
    }

    // This implementation is based on a reference implementation from
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h
    virtual void forward(cv::InputArrayOfArrays inputs_arr,
                         cv::OutputArrayOfArrays outputs_arr,
                         cv::OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        if (inputs_arr.depth() == CV_16S)
        {
            // In case of DNN_TARGET_OPENCL_FP16 target the following method
            // converts data from FP16 to FP32 and calls this forward again.
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<cv::Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        cv::Mat& inp = inputs[0];
        cv::Mat& out = outputs[0];
        const float* inpData = (float*)inp.data;
        float* outData = (float*)out.data;

        const int batchSize = inp.size[0];
        const int numChannels = inp.size[1];
        const int inpHeight = inp.size[2];
        const int inpWidth = inp.size[3];

        float heightScale = static_cast<float>(inpHeight) / outHeight;
        float widthScale = static_cast<float>(inpWidth) / outWidth;
        for (int b = 0; b < batchSize; ++b)
        {
            for (int y = 0; y < outHeight; ++y)
            {
                float input_y = y * heightScale;
                int y0 = static_cast<int>(std::floor(input_y));
                int y1 = std::min(y0 + 1, inpHeight - 1);
                for (int x = 0; x < outWidth; ++x)
                {
                    float input_x = x * widthScale;
                    int x0 = static_cast<int>(std::floor(input_x));
                    int x1 = std::min(x0 + 1, inpWidth - 1);
                    for (int c = 0; c < numChannels; ++c)
                    {
                        float interpolation =
                            inpData[offset(inp.size, c, x0, y0, b)] * (1 - (input_y - y0)) * (1 - (input_x - x0)) +
                            inpData[offset(inp.size, c, x0, y1, b)] * (input_y - y0) * (1 - (input_x - x0)) +
                            inpData[offset(inp.size, c, x1, y0, b)] * (1 - (input_y - y0)) * (input_x - x0) +
                            inpData[offset(inp.size, c, x1, y1, b)] * (input_y - y0) * (input_x - x0);
                        outData[offset(out.size, c, x, y, b)] = interpolation;
                    }
                }
            }
        }
    }

private:
    static inline int offset(const cv::MatSize& size, int c, int x, int y, int b)
    {
        return x + size[3] * (y + size[2] * (c + size[1] * b));
    }

    int outWidth, outHeight, factorWidth, factorHeight;
};
//! [ResizeBilinearLayer]

//
// The following code is used only to generate tutorials documentation.
//

//! [A custom layer interface]
class MyLayer : public cv::dnn::Layer
{
public:
    //! [MyLayer::MyLayer]
    MyLayer(const cv::dnn::LayerParams &params);
    //! [MyLayer::MyLayer]

    //! [MyLayer::create]
    static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params);
    //! [MyLayer::create]

    //! [MyLayer::getMemoryShapes]
    virtual bool getMemoryShapes(const std::vector<std::vector<int> > &inputs,
                                 const int requiredOutputs,
                                 std::vector<std::vector<int> > &outputs,
                                 std::vector<std::vector<int> > &internals) const CV_OVERRIDE;
    //! [MyLayer::getMemoryShapes]

    //! [MyLayer::forward]
    virtual void forward(cv::InputArrayOfArrays inputs,
                         cv::OutputArrayOfArrays outputs,
                         cv::OutputArrayOfArrays internals) CV_OVERRIDE;
    //! [MyLayer::forward]

    //! [MyLayer::finalize]
    virtual void finalize(cv::InputArrayOfArrays inputs,
                          cv::OutputArrayOfArrays outputs) CV_OVERRIDE;
    //! [MyLayer::finalize]
};
//! [A custom layer interface]

//! [Register a custom layer]
#include <opencv2/dnn/layer.details.hpp>  // CV_DNN_REGISTER_LAYER_CLASS

static inline void loadNet()
{
    CV_DNN_REGISTER_LAYER_CLASS(Interp, InterpLayer);
    // ...
    //! [Register a custom layer]

    //! [Register InterpLayer]
    CV_DNN_REGISTER_LAYER_CLASS(Interp, InterpLayer);
    cv::dnn::Net caffeNet = cv::dnn::readNet("/path/to/config.prototxt", "/path/to/weights.caffemodel");
    //! [Register InterpLayer]

    //! [Register ResizeBilinearLayer]
    CV_DNN_REGISTER_LAYER_CLASS(ResizeBilinear, ResizeBilinearLayer);
    cv::dnn::Net tfNet = cv::dnn::readNet("/path/to/graph.pb");
    //! [Register ResizeBilinearLayer]

    if (false) loadNet();  // To prevent unused function warning.
}

#endif  // __OPENCV_SAMPLES_DNN_CUSTOM_LAYERS__
