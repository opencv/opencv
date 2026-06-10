// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#ifdef HAVE_OPENCV_DNN
#include "opencv2/dnn.hpp"
#include "aliked_context.hpp"
#endif

namespace cv
{

ALIKED::ALIKED() {}
ALIKED::~ALIKED() {}

ALIKED::Params::Params()
{
    inputSize = Size(640, 640);
    normalizeDescriptors = true;
#ifdef HAVE_OPENCV_DNN
    engine = dnn::ENGINE_NEW;
    backend = dnn::DNN_BACKEND_DEFAULT;
    target = dnn::DNN_TARGET_CPU;
#else
    engine = -1;
    backend = -1;
    target = -1;
#endif
}

#ifdef HAVE_OPENCV_DNN

class ALIKEDImpl : public ALIKED
{
public:
    ALIKEDImpl(const ALIKED::Params& _params, const String& modelPath)
        : params(_params)
    {
        net = dnn::readNet(modelPath, "", "", static_cast<dnn::EngineType>(params.engine));
        CV_Assert(!net.empty());
        net.setPreferableBackend(params.backend);
        net.setPreferableTarget(params.target);
    }

    ALIKEDImpl(const std::vector<uchar>& modelData, const ALIKED::Params& _params)
        : params(_params)
    {
        net = dnn::readNetFromONNX(modelData);
        CV_Assert(!net.empty());
        net.setPreferableBackend(params.backend);
        net.setPreferableTarget(params.target);
    }

    void detectAndCompute(InputArray image, InputArray mask,
                          std::vector<KeyPoint>& keypoints,
                          OutputArray descriptors,
                          bool useProvidedKeypoints) CV_OVERRIDE;

    int descriptorSize() const CV_OVERRIDE;
    int descriptorType() const CV_OVERRIDE;
    int defaultNorm() const CV_OVERRIDE;
    bool empty() const CV_OVERRIDE;

    const ALIKEDContext& getLastContext() const { return lastContext; }

protected:
    dnn::Net net;
    ALIKED::Params params;
    ALIKEDContext lastContext;

    void runNetwork(InputArray image, std::vector<KeyPoint>& keypoints,
                    Mat& descriptors, Mat& scores);
};

void ALIKEDImpl::runNetwork(InputArray _image, std::vector<KeyPoint>& keypoints,
                             Mat& descriptors, Mat& scores)
{
    Mat image = _image.getMat();
    Size inputSz = params.inputSize;
    Size origSize = image.size();

    // BGR->RGB conversion via swapRB=true
    Mat blob = dnn::blobFromImage(image, 1.0/255.0, inputSz, Scalar(), /*swapRB=*/true, /*crop=*/false);

    net.setInput(blob, "image");

    std::vector<String> outNames = {"keypoints", "descriptors", "scores"};
    std::vector<Mat> outputs;
    net.forward(outputs, outNames);

    CV_Assert(outputs.size() == 3);

    // ORT engine drops the batch dimension, so outputs are:
    //   keypoints:   [N, 2]   (not [1, N, 2])
    //   descriptors: [N, 128] (not [1, N, 128])
    //   scores:      [N]      (not [1, N, 1])
    int N = outputs[0].rows;

    Mat normKpts = outputs[0].reshape(0, N);   // Nx2
    Mat desc = outputs[1].reshape(0, N);       // Nx128
    Mat scr = outputs[2].reshape(0, N);        // Nx1

    // Store normalized keypoints for LightGlue context
    lastContext.normalizedKeypoints = normKpts.clone();
    lastContext.imageSize = origSize;

    // Convert normalized [-1,1] coordinates to pixel coordinates
    keypoints.resize(N);
    for (int i = 0; i < N; i++)
    {
        float nx = normKpts.at<float>(i, 0);
        float ny = normKpts.at<float>(i, 1);
        float px = (nx + 1.0f) * 0.5f * (float)origSize.width;
        float py = (ny + 1.0f) * 0.5f * (float)origSize.height;
        float score = scr.at<float>(i, 0);
        keypoints[i] = KeyPoint(px, py, 1.0f, -1.0f, score, 0, -1);
    }

    // Optionally L2-normalize descriptors
    if (params.normalizeDescriptors)
    {
        for (int i = 0; i < N; i++)
        {
            Mat row = desc.row(i);
            normalize(row, row);
        }
    }

    descriptors = desc;
    scores = scr;
}

void ALIKEDImpl::detectAndCompute(InputArray image, InputArray mask,
                                   std::vector<KeyPoint>& keypoints,
                                   OutputArray descriptors,
                                   bool useProvidedKeypoints)
{
    CV_INSTRUMENT_REGION();
    CV_UNUSED(mask);
    CV_UNUSED(useProvidedKeypoints);

    if (image.empty())
    {
        keypoints.clear();
        descriptors.release();
        return;
    }

    Mat desc;
    Mat sc;
    runNetwork(image, keypoints, desc, sc);

    if (descriptors.needed())
        desc.copyTo(descriptors);
}

int ALIKEDImpl::descriptorSize() const { return 128; }
int ALIKEDImpl::descriptorType() const { return CV_32F; }
int ALIKEDImpl::defaultNorm() const { return NORM_L2; }
bool ALIKEDImpl::empty() const { return net.empty(); }

Ptr<ALIKED> ALIKED::create(const String& modelPath, const ALIKED::Params& params)
{
    return makePtr<ALIKEDImpl>(params, modelPath);
}

Ptr<ALIKED> ALIKED::create(const std::vector<uchar>& modelData, const ALIKED::Params& params)
{
    return makePtr<ALIKEDImpl>(modelData, params);
}

#else  // !HAVE_OPENCV_DNN

Ptr<ALIKED> ALIKED::create(const String& modelPath, const ALIKED::Params& params)
{
    CV_UNUSED(modelPath);
    CV_UNUSED(params);
    CV_Error(cv::Error::StsNotImplemented,
             "ALIKED requires OpenCV built with opencv_dnn module!");
}

#endif  // HAVE_OPENCV_DNN

}  // namespace cv
