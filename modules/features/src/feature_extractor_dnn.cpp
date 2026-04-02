// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "opencv2/features/feature_extractor.hpp"

#ifdef HAVE_OPENCV_DNN
#include "opencv2/dnn/dnn.hpp"
#endif

namespace cv
{
    namespace features
    {

        SuperPoint::Params::Params()
        {
            modelPath = String();
            dnnEngine = 4; // dnn::ENGINE_ORT
            dnnBackend = -1;
            dnnTarget = -1;
            inputSize = Size(640, 640);
            preferGrayInput = true;
        }

        namespace
        {

#ifdef HAVE_OPENCV_DNN

            static const char *const kInputName = "image";
            static const char *const kKeypointsName = "keypoints";
            static const char *const kDescriptorsName = "descriptors";
            static const char *const kScoresName = "scores";

            static Mat makeSuperPointInputBlob(const Mat &src,
                                               const Size &inputSize,
                                               bool useGray)
            {
                Mat resized;
                resize(src, resized, inputSize);

                if (useGray)
                {
                    Mat gray;
                    if (resized.channels() == 1)
                    {
                        gray = resized;
                    }
                    else
                    {
                        cvtColor(resized, gray, COLOR_BGR2GRAY);
                    }

                    // SuperPoint models are commonly exported with (1, 1, H, W).
                    return dnn::blobFromImage(gray, 1.0 / 255.0, inputSize, Scalar(), false, false, CV_32F);
                }

                return dnn::blobFromImage(resized, 1.0 / 255.0, inputSize, Scalar(), true, false, CV_32F);
            }

            static void parseSuperPointOutputs(const std::vector<Mat> &outs,
                                               Size imageSize,
                                               std::vector<KeyPoint> &keypoints,
                                               Mat &descriptors)
            {
                CV_Assert(outs.size() >= 2);

                Mat kpts = outs[0];
                Mat desc = outs[1];
                Mat scores = outs.size() > 2 ? outs[2] : Mat();

                if (!kpts.isContinuous())
                    kpts = kpts.clone();

                Mat kptsFlat;
                if (kpts.total() % 2 == 0)
                {
                    kptsFlat = kpts.reshape(1, static_cast<int>(kpts.total() / 2));
                }
                else
                {
                    CV_Error(Error::StsError, "SuperPoint keypoints output has invalid shape");
                }

                kptsFlat.convertTo(kptsFlat, CV_32F);

                Mat descFlat;
                if (desc.dims == 3)
                {
                    int rows = desc.size[1];
                    int cols = desc.size[2];
                    descFlat = desc.reshape(1, rows);
                    if (rows != kptsFlat.rows && cols == kptsFlat.rows)
                    {
                        Mat transposed;
                        transpose(descFlat, transposed);
                        descFlat = transposed;
                    }
                }
                else if (desc.dims == 2)
                {
                    descFlat = desc;
                }
                else
                {
                    CV_Error(Error::StsError, "SuperPoint descriptors output has invalid shape");
                }

                descFlat.convertTo(descFlat, CV_32F);

                if (descFlat.rows != kptsFlat.rows)
                {
                    CV_Error(Error::StsError, "SuperPoint outputs have inconsistent keypoint/descriptor count");
                }

                double minVal = 0.0;
                double maxVal = 0.0;
                minMaxIdx(kptsFlat, &minVal, &maxVal);
                const bool normalized = maxVal <= 1.0;
                const bool shifted = minVal < 0.0;

                Mat scoresFlat;
                if (!scores.empty())
                {
                    scoresFlat = scores.reshape(1, static_cast<int>(scores.total()));
                    scoresFlat.convertTo(scoresFlat, CV_32F);
                }

                keypoints.clear();
                keypoints.reserve(static_cast<size_t>(kptsFlat.rows));

                for (int i = 0; i < kptsFlat.rows; ++i)
                {
                    float x = kptsFlat.at<float>(i, 0);
                    float y = kptsFlat.at<float>(i, 1);
                    if (normalized)
                    {
                        if (shifted)
                        {
                            x = (x + 1.0f) * 0.5f;
                            y = (y + 1.0f) * 0.5f;
                        }
                        x *= static_cast<float>(imageSize.width);
                        y *= static_cast<float>(imageSize.height);
                    }

                    const float response = scoresFlat.empty() ? 1.0f : scoresFlat.at<float>(i, 0);
                    keypoints.push_back(KeyPoint(Point2f(x, y), 1.0f, -1.0f, response));
                }

                descriptors = descFlat;
            }

            class SuperPointImpl CV_FINAL : public SuperPoint
            {
            public:
                explicit SuperPointImpl(const SuperPoint::Params &params)
                    : params_(params)
                {
                    if (!params_.modelPath.empty())
                        setModel(params_.modelPath);
                }

                void setModel(const String &modelPath) CV_OVERRIDE
                {
                    params_.modelPath = modelPath;
                    net_ = dnn::readNetFromONNX(modelPath, params_.dnnEngine);
                    CV_Assert(!net_.empty());

                    if (params_.dnnBackend >= 0)
                        net_.setPreferableBackend(params_.dnnBackend);
                    if (params_.dnnTarget >= 0)
                        net_.setPreferableTarget(params_.dnnTarget);
                }

                String getModel() const CV_OVERRIDE
                {
                    return params_.modelPath;
                }

                void extract(InputArray image,
                             std::vector<KeyPoint> &keypoints,
                             OutputArray descriptors,
                             InputArray mask) const CV_OVERRIDE
                {
                    CV_Assert(!net_.empty());

                    Mat src = image.getMat();
                    CV_Assert(!src.empty());

                    Size inputSize = params_.inputSize;
                    if (inputSize.width <= 0 || inputSize.height <= 0)
                        inputSize = src.size();

                    std::vector<String> outNames;
                    outNames.push_back(kKeypointsName);
                    outNames.push_back(kDescriptorsName);
                    outNames.push_back(kScoresName);

                    std::vector<Mat> outs;
                    std::string inferenceError;
                    bool inferenceOk = false;

                    for (int attempt = 0; attempt < 2; ++attempt)
                    {
                        const bool useGray = params_.preferGrayInput ? (attempt == 0) : (attempt != 0);
                        Mat blob = makeSuperPointInputBlob(src, inputSize, useGray);

                        try
                        {
                            net_.setInput(blob, kInputName);
                            net_.forward(outs, outNames);
                            inferenceOk = true;
                            break;
                        }
                        catch (const std::exception &e)
                        {
                            if (inferenceError.empty())
                                inferenceError = e.what();
                        }
                    }

                    if (!inferenceOk)
                    {
                        CV_Error(Error::StsError,
                                 "SuperPoint inference failed for both grayscale and color inputs: " + inferenceError);
                    }

                    Mat desc;
                    parseSuperPointOutputs(outs, inputSize, keypoints, desc);

                    if (!mask.empty())
                    {
                        Mat maskMat = mask.getMat();
                        CV_Assert(maskMat.type() == CV_8UC1);
                        CV_Assert(maskMat.size() == inputSize);

                        std::vector<KeyPoint> filteredKeypoints;
                        Mat filteredDesc;
                        for (int i = 0; i < static_cast<int>(keypoints.size()); ++i)
                        {
                            const Point2f &pt = keypoints[i].pt;
                            const int x = cvRound(pt.x);
                            const int y = cvRound(pt.y);
                            if (x < 0 || y < 0 || x >= maskMat.cols || y >= maskMat.rows)
                                continue;
                            if (maskMat.at<uchar>(y, x) == 0)
                                continue;

                            filteredKeypoints.push_back(keypoints[i]);
                            filteredDesc.push_back(desc.row(i));
                        }
                        keypoints.swap(filteredKeypoints);
                        desc = filteredDesc;
                    }

                    desc.copyTo(descriptors);
                }

            private:
                SuperPoint::Params params_;
                mutable dnn::Net net_;
            };

#else

            class SuperPointImpl CV_FINAL : public SuperPoint
            {
            public:
                explicit SuperPointImpl(const SuperPoint::Params &params)
                    : params_(params)
                {
                }

                void setModel(const String &modelPath) CV_OVERRIDE
                {
                    params_.modelPath = modelPath;
                }

                String getModel() const CV_OVERRIDE
                {
                    return params_.modelPath;
                }

                void extract(InputArray,
                             std::vector<KeyPoint> &,
                             OutputArray,
                             InputArray) const CV_OVERRIDE
                {
                    CV_Error(Error::StsNotImplemented, "SuperPoint requires OpenCV built with DNN support");
                }

            private:
                SuperPoint::Params params_;
            };

#endif

        } // namespace

        Ptr<SuperPoint> SuperPoint::create()
        {
            return create(SuperPoint::Params());
        }

        Ptr<SuperPoint> SuperPoint::create(const SuperPoint::Params &params)
        {
            return makePtr<SuperPointImpl>(params);
        }

    } // namespace features
} // namespace cv
