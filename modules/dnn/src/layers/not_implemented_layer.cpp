// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../dnn_common.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

namespace detail {

class NotImplementedImpl CV_FINAL : public NotImplemented
{
public:
    NotImplementedImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        CV_Assert(params.has("type"));
        std::stringstream ss;
        ss << "Node for layer '" << params.name << "' of type '" << params.get("type") << "' wasn't initialized.";
        msg = ss.str();
    }

    CV_DEPRECATED_EXTERNAL
    virtual void finalize(const std::vector<Mat*> &input, std::vector<Mat> &output) CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented, msg);
    }

    virtual void finalize(InputArrayOfArrays inputs, OutputArrayOfArrays outputs) CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented, msg);
    }

    CV_DEPRECATED_EXTERNAL
    virtual void forward(std::vector<Mat*> &input, std::vector<Mat> &output, std::vector<Mat> &internals) CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented, msg);
    }

    virtual void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented, msg);
    }

    void forward_fallback(InputArrayOfArrays inputs, OutputArrayOfArrays outputs, OutputArrayOfArrays internals)
    {
        CV_Error(Error::StsNotImplemented, msg);
    }

    CV_DEPRECATED_EXTERNAL
    void finalize(const std::vector<Mat> &inputs, CV_OUT std::vector<Mat> &outputs)
    {
        CV_Error(Error::StsNotImplemented, msg);
    }

    CV_DEPRECATED std::vector<Mat> finalize(const std::vector<Mat> &inputs)
    {
        CV_Error(Error::StsNotImplemented, msg);
    }

    CV_DEPRECATED void run(const std::vector<Mat> &inputs,
                           CV_OUT std::vector<Mat> &outputs,
                           CV_IN_OUT std::vector<Mat> &internals)
    {
        CV_Error(Error::StsNotImplemented, msg);
    }

    virtual int inputNameToIndex(String inputName) CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented, msg);
    }

    virtual int outputNameToIndex(const String& outputName) CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented, msg);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented, msg);
    }

    virtual Ptr<BackendNode> initHalide(const std::vector<Ptr<BackendWrapper> > &inputs) CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented, msg);
    }

    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> > &inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented, msg);
    }

    virtual Ptr<BackendNode> initVkCom(const std::vector<Ptr<BackendWrapper> > &inputs) CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented, msg);
    }

    virtual Ptr<BackendNode> initCUDA(
            void *context,
            const std::vector<Ptr<BackendWrapper>>& inputs,
            const std::vector<Ptr<BackendWrapper>>& outputs
    ) CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented, msg);
    }

    virtual void applyHalideScheduler(Ptr<BackendNode>& node,
                                      const std::vector<Mat*> &inputs,
                                      const std::vector<Mat> &outputs,
                                      int targetId) const CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented, msg);
    }

    virtual Ptr<BackendNode> tryAttach(const Ptr<BackendNode>& node) CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented, msg);
    }

    virtual bool setActivation(const Ptr<ActivationLayer>& layer) CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented, msg);
    }

    virtual bool tryFuse(Ptr<Layer>& top) CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented, msg);
    }

    virtual void getScaleShift(Mat& scale, Mat& shift) const CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented, msg);
    }

    virtual void unsetAttached() CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented, msg);
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented, msg);
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented, msg);
    }

    virtual bool updateMemoryShapes(const std::vector<MatShape> &inputs) CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented, msg);
    }

private:
    std::string msg;
};

Ptr<Layer> NotImplemented::create(const LayerParams& params)
{
    return makePtr<NotImplementedImpl>(params);
}

Ptr<Layer> notImplementedRegisterer(LayerParams &params)
{
    return detail::NotImplemented::create(params);
}

void NotImplemented::Register()
{
    LayerFactory::registerLayer("NotImplemented", detail::notImplementedRegisterer);
}

void NotImplemented::unRegister()
{
    LayerFactory::unregisterLayer("NotImplemented");
}

} // namespace detail

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
