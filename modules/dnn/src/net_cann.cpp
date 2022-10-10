// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include <opencv2/core/utils/logger.hpp>

#include "net_impl.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

#ifdef HAVE_CANN

class NetImplCann CV_FINAL : public Net::Impl
{
public:
    typedef Net::Impl Base;
    std::shared_ptr<CannClient> cannClient = std::make_shared<CannClient>(); // TODO: supports setting different device?

    explicit NetImplCann(const Ptr<Net::Impl>& basePtr)
        : Net::Impl()
    {
        CV_LOG_INFO(NULL, "Initializing NetImplCann");
        basePtr_ = basePtr;

        Net::Impl& base = *basePtr_;

        preferableBackend = DNN_BACKEND_CANN;
        preferableTarget = DNN_TARGET_NPU;

        ge::Graph& graph = base.cannGraph->graph;
        cannClient->buildModelFromGraph(graph);
        cannClient->loadModel();

        CV_LOG_INFO(NULL, "Finished initializing NetImplCann");
    }

    bool empty() const override
    {
        return cannClient->empty();
    }

    void setPreferableBackend(Net& net, int backendId) override
    {
        if (backendId == preferableBackend)
            return;  // no-op
        else
            CV_Error(Error::StsError, "DNN: Can't switch backend from CANN to other");
    }

    void setPreferableTarget(int targetId) override
    {
        if (targetId != preferableTarget)
        {
            CV_Error(Error::StsError, "DNN: Can't switch target from NPU to other");
        }
    }

    void setInput(InputArray blob, const String& name, double scalefactor, const Scalar& mean) override
    {
        Mat blob_ = blob.getMat();
        Mat input_;
        blob_.convertTo(input_, CV_32F, scalefactor, -mean[0] * scalefactor);

        cannClient->setInput(input_, name);
    }

    Mat forward(const String& outputName) override
    {
        cannClient->forward();

        Mat output;
        cannClient->fetchOutput(output, outputName);
        return output;
    }

    void forward(OutputArrayOfArrays outputBlobs, const String& outputName) override
    {
        // cannGraph->forward();
        cannClient->forward();

        if (outputBlobs.isMat())
        {
            Mat output;
            cannClient->fetchOutput(output, outputName);
            outputBlobs.assign(output);
        }
        else if (outputBlobs.isMatVector())
        {
            int output_num = cannClient->getOutputNum();
            std::vector<Mat> matVec;
            for (int i = 0; i < output_num; i++)
            {
                Mat output_i;
                cannClient->fetchOutput(output_i, i);
                matVec.push_back(output_i);
            }
            outputBlobs.create(output_num, 1, CV_32F, -1);
            outputBlobs.assign(matVec);
        }
        else
            CV_Error(Error::StsNotImplemented, "Content of outputBlobs should be Mat or std::vector<Mat>");
    }

    void forward(OutputArrayOfArrays outputBlobs,
                 const std::vector<String>& outBlobNames) override
    {
        cannClient->forward();

        std::vector<Mat> matVec;
        for (size_t i = 0; i < outBlobNames.size(); i++)
        {
            Mat output_i;
            cannClient->fetchOutput(output_i, outBlobNames[i]);
            matVec.push_back(output_i);
        }
        outputBlobs.create((int)outBlobNames.size(), 1, CV_32F, -1);
        outputBlobs.assign(matVec);
    }

    void forward(std::vector<std::vector<Mat>>& outputBlobs,
                 const std::vector<String>& outBlobNames) override
    {
        // FIXIT: what does this API mean?
        CV_Error(Error::StsNotImplemented, "Not supported");
    }
};

void switchToCannBackend(Net& net)
{
    CV_TRACE_FUNCTION();
    Ptr<Net::Impl>& impl_ptr_ref = accessor::DnnNetAccessor::getImplPtrRef(net);
    CV_Assert(impl_ptr_ref);
    CV_LOG_INFO(NULL, "DNN: switching to CANN backend... (networkID=" << impl_ptr_ref->networkId << ")");
    Ptr<NetImplCann> impl_ptr_cann = makePtr<NetImplCann>(impl_ptr_ref);
    impl_ptr_ref = impl_ptr_cann;
}

#endif // HAVE_CANN

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn
