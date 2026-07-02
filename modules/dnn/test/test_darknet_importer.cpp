/*M///////////////////////////////////////////////////////////////////////////////////////
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
//                        (3-clause BSD License)
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// * Neither the names of the copyright holders nor the names of the contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"
#include "npy_blob.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace opencv_test { namespace {

template<typename TString>
static std::string _tf(TString filename)
{
    return (getOpenCVExtraDir() + "/dnn/") + filename;
}

TEST(Test_YOLO, read_yolov4_onnx)
{
    Net net = readNet(findDataFile("dnn/yolov4.onnx", false));
    ASSERT_FALSE(net.empty());
}

class Test_YOLO_nets : public DNNTestLayer
{
public:
    // Test object detection network from ONNX model.
    void testYOLOModel(const std::string& model,
                       const std::vector<std::vector<int> >& refClassIds,
                       const std::vector<std::vector<float> >& refConfidences,
                       const std::vector<std::vector<Rect2d> >& refBoxes,
                       double scoreDiff, double iouDiff, float confThreshold = 0.24,
                       float nmsThreshold = 0.4, bool useWinograd = true,
                       int zeroPadW = 0, Size inputSize = Size())
    {
        checkBackend();

        Mat img1 = imread(_tf("dog416.png"));
        Mat img2 = imread(_tf("street.png"));
        cv::resize(img1, img1, inputSize);
        cv::resize(img2, img2, inputSize);

        // Pad images by black pixel at the right to test not equal width and height sizes
        if (zeroPadW) {
            cv::copyMakeBorder(img1, img1, 0, 0, 0, zeroPadW, BORDER_CONSTANT);
            cv::copyMakeBorder(img2, img2, 0, 0, 0, zeroPadW, BORDER_CONSTANT);
        }

        std::vector<Mat> samples(2);
        samples[0] = img1; samples[1] = img2;

        // determine test type, whether batch or single img
        int batch_size = refClassIds.size();
        CV_Assert(batch_size == 1 || batch_size == 2);
        samples.resize(batch_size);

        Mat inp = blobFromImages(samples, 1.0/255, Size(), Scalar(), true, false);

        Net net = readNet(findDataFile("dnn/" + model, false));
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);
        net.enableWinograd(useWinograd);
        net.setInput(inp);
        std::vector<Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());

        // Detect output format: pytorch-YOLOv4 exports "boxes" [batch, N, 1, 4] + "confs" [batch, N, classes]
        bool isBoxConfsFormat = (outs.size() == 2 && outs[0].dims == 4 && outs[0].size[outs[0].dims - 1] == 4);
        // Detect 3-output format: boxes [batch, N, 4] + scores [batch, N] + class_idx [batch, N]
        bool isBoxScoresIdxFormat = (outs.size() == 3 && outs[0].dims == 3 && outs[0].size[2] == 4);

        for (int b = 0; b < batch_size; ++b)
        {
            std::vector<int> classIds;
            std::vector<float> confidences;
            std::vector<Rect2d> boxes;
            if (isBoxScoresIdxFormat)
            {
                // yolov3-style format: boxes [batch, N, 4] + scores [batch, N] + class_idx [batch, N]
                // boxes are [x1, y1, x2, y2] in pixel coords (relative to model input size)
                int N = outs[0].size[1];
                float* boxesPtr = outs[0].ptr<float>(b);
                float* scoresPtr = outs[1].ptr<float>(b);
                float* classIdxPtr = outs[2].ptr<float>(b);

                float modelW = (float)inp.size[3];
                float modelH = (float)inp.size[2];

                for (int j = 0; j < N; ++j)
                {
                    float score = scoresPtr[j];
                    if (score > confThreshold)
                    {
                        float x1 = boxesPtr[j * 4 + 0] / modelW;
                        float y1 = boxesPtr[j * 4 + 1] / modelH;
                        float x2 = boxesPtr[j * 4 + 2] / modelW;
                        float y2 = boxesPtr[j * 4 + 3] / modelH;
                        boxes.push_back(Rect2d(x1, y1, x2 - x1, y2 - y1));
                        confidences.push_back(score);
                        classIds.push_back((int)classIdxPtr[j]);
                    }
                }
            }
            else if (isBoxConfsFormat)
            {
                // boxes [batch, N, 1, 4] (x1,y1,x2,y2), confs [batch, N, num_classes]
                Mat boxesMat = outs[0];
                Mat confsMat = outs[1];
                if (batch_size > 1)
                {
                    if (boxesMat.dims == 4) {
                        Range boxRanges[4] = {Range(b, b+1), Range::all(), Range::all(), Range::all()};
                        boxesMat = boxesMat(boxRanges);
                    } else {
                        Range boxRanges[3] = {Range(b, b+1), Range::all(), Range::all()};
                        boxesMat = boxesMat(boxRanges);
                    }
                    Range confRanges[3] = {Range(b, b+1), Range::all(), Range::all()};
                    confsMat = confsMat(confRanges);
                }

                int numBoxes = (int)(boxesMat.total() / 4);
                boxesMat = boxesMat.reshape(1, numBoxes);
                confsMat = confsMat.reshape(1, numBoxes);

                for (int j = 0; j < numBoxes; ++j)
                {
                    Mat scores = confsMat.row(j);
                    double confidence;
                    Point maxLoc;
                    minMaxLoc(scores, 0, &confidence, 0, &maxLoc);

                    if (confidence > confThreshold) {
                        float* box = boxesMat.ptr<float>(j);
                        double x1 = box[0];
                        double y1 = box[1];
                        double x2 = box[2];
                        double y2 = box[3];
                        boxes.push_back(Rect2d(x1, y1, x2 - x1, y2 - y1));
                        confidences.push_back(confidence);
                        classIds.push_back(maxLoc.x);
                    }
                }
            }
            else
            {
                for (int i = 0; i < (int)outs.size(); ++i)
                {
                    Mat out;
                    if (batch_size > 1){
                        Range ranges[3] = {Range(b, b+1), Range::all(), Range::all()};
                        out = outs[i](ranges).reshape(1, outs[i].size[1]);
                    }else{
                        out = outs[i];
                    }
                    for (int j = 0; j < out.rows; ++j)
                    {
                        float objConf = out.at<float>(j, 4);
                        Mat scores = out.row(j).colRange(5, out.cols);
                        double maxClsScore;
                        Point maxLoc;
                        minMaxLoc(scores, 0, &maxClsScore, 0, &maxLoc);
                        double confidence = objConf * maxClsScore;

                        if (confidence > confThreshold) {
                            float* detection = out.ptr<float>(j);
                            double centerX = detection[0];
                            double centerY = detection[1];
                            double width = detection[2];
                            double height = detection[3];
                            boxes.push_back(Rect2d(centerX - 0.5 * width, centerY - 0.5 * height,
                                                width, height));
                            confidences.push_back(confidence);
                            classIds.push_back(maxLoc.x);
                        }
                    }
                }
            }

            // here we need NMS of boxes
            std::vector<int> indices;
            NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

            std::vector<int> nms_classIds;
            std::vector<float> nms_confidences;
            std::vector<Rect2d> nms_boxes;

            for (size_t i = 0; i < indices.size(); ++i)
            {
                int idx = indices[i];
                Rect2d box = boxes[idx];
                float conf = confidences[idx];
                int class_id = classIds[idx];
                nms_boxes.push_back(box);
                nms_confidences.push_back(conf);
                nms_classIds.push_back(class_id);
                if (cvtest::debugLevel > 0)
                {
                    std::cout << b << ", " << class_id << ", " << conf << "f, "
                              << box.x << "f, " << box.y << "f, "
                              << box.x + box.width << "f, " << box.y + box.height << "f,"
                              << std::endl;
                }
            }

            if (cvIsNaN(iouDiff))
            {
                if (b == 0)
                    std::cout << "Skip accuracy checks" << std::endl;
                continue;
            }

            // Return predictions from padded image to the origin
            if (zeroPadW) {
                float scale = static_cast<float>(inp.size[3]) / (inp.size[3] - zeroPadW);
                for (auto& box : nms_boxes) {
                    box.x *= scale;
                    box.width *= scale;
                }
            }
            normAssertDetections(refClassIds[b], refConfidences[b], refBoxes[b], nms_classIds,
                             nms_confidences, nms_boxes, format("batch size %d, sample %d\n", batch_size, b).c_str(), confThreshold, scoreDiff, iouDiff);
        }
    }

    void testYOLOModel(const std::string& model,
                       const std::vector<int>& refClassIds,
                       const std::vector<float>& refConfidences,
                       const std::vector<Rect2d>& refBoxes,
                       double scoreDiff, double iouDiff, float confThreshold = 0.24,
                       float nmsThreshold = 0.4, bool useWinograd = true,
                       int zeroPadW = 0, Size inputSize = Size())
    {
        testYOLOModel(model,
                      std::vector<std::vector<int> >(1, refClassIds),
                      std::vector<std::vector<float> >(1, refConfidences),
                      std::vector<std::vector<Rect2d> >(1, refBoxes),
                      scoreDiff, iouDiff, confThreshold, nmsThreshold, useWinograd, zeroPadW, inputSize);
    }

    void testYOLOModel(const std::string& model,
                       const cv::Mat& ref, double scoreDiff, double iouDiff,
                       float confThreshold = 0.24, float nmsThreshold = 0.4, bool useWinograd = true,
                       int zeroPadW = 0, Size inputSize = Size())
    {
        CV_Assert(ref.cols == 7);
        std::vector<std::vector<int> > refClassIds;
        std::vector<std::vector<float> > refScores;
        std::vector<std::vector<Rect2d> > refBoxes;
        for (int i = 0; i < ref.rows; ++i)
        {
            int batchId = static_cast<int>(ref.at<float>(i, 0));
            int classId = static_cast<int>(ref.at<float>(i, 1));
            float score = ref.at<float>(i, 2);
            float left  = ref.at<float>(i, 3);
            float top   = ref.at<float>(i, 4);
            float right  = ref.at<float>(i, 5);
            float bottom = ref.at<float>(i, 6);
            Rect2d box(left, top, right - left, bottom - top);
            if (batchId >= (int)refClassIds.size())
            {
                refClassIds.resize(batchId + 1);
                refScores.resize(batchId + 1);
                refBoxes.resize(batchId + 1);
            }
            refClassIds[batchId].push_back(classId);
            refScores[batchId].push_back(score);
            refBoxes[batchId].push_back(box);
        }
        testYOLOModel(model, refClassIds, refScores, refBoxes,
                      scoreDiff, iouDiff, confThreshold, nmsThreshold, useWinograd, zeroPadW, inputSize);
    }
};

TEST_P(Test_YOLO_nets, YOLOv4)
{
    applyTestTag(
            CV_TEST_TAG_LONG,
            CV_TEST_TAG_MEMORY_2GB,
            CV_TEST_TAG_DEBUG_VERYLONG
            );

    // batchId, classId, confidence, left, top, right, bottom
    const int N0 = 3;
    const int N1 = 6;
    static const float ref_[/* (N0 + N1) * 7 */] = {
0, 16, 0.968371f, 0.167918f, 0.394843f, 0.40767f, 0.942042f,
0, 1, 0.963549f, 0.146538f, 0.227724f, 0.745242f, 0.736494f,
0, 7, 0.951405f, 0.606025f, 0.133886f, 0.895092f, 0.294835f,

1, 2, 0.99849f, 0.651516f, 0.456526f, 0.812706f, 0.66287f,
1, 0, 0.996791f, 0.200903f, 0.362404f, 0.264643f, 0.627633f,
1, 2, 0.987972f, 0.450125f, 0.464126f, 0.495712f, 0.519708f,
1, 9, 0.85872f, 0.375374f, 0.314192f, 0.399161f, 0.39453f,
1, 9, 0.841318f, 0.667602f, 0.377284f, 0.686024f, 0.440855f,
1, 9, 0.502608f, 0.656728f, 0.378153f, 0.668251f, 0.432035f,
    };
    Mat ref(N0 + N1, 7, CV_32FC1, (void*)ref_);

    double scoreDiff = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CPU_FP16) ? 0.006 : 8e-5;
    double iouDiff = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CPU_FP16) ? 0.042 : 3e-4;
    if (target == DNN_TARGET_CUDA_FP16)
    {
        scoreDiff = 0.008;
        iouDiff = 0.03;
    }

    std::string model_file = "yolov4.onnx";

    {
        SCOPED_TRACE("batch size 1");
        testYOLOModel(model_file, ref.rowRange(0, N0), scoreDiff, iouDiff, 0.3, 0.4, false, 0, Size(608, 608));
    }

    {
        SCOPED_TRACE("batch size 2");
        testYOLOModel(model_file, ref, scoreDiff, iouDiff, 0.3, 0.4, false, 0, Size(608, 608));
    }
}

TEST_P(Test_YOLO_nets, YOLOv3)
{
    applyTestTag(
            CV_TEST_TAG_LONG,
            CV_TEST_TAG_MEMORY_2GB,
            CV_TEST_TAG_DEBUG_VERYLONG
    );

    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);

    // batchId, classId, confidence, left, top, right, bottom
    const int N0 = 3;
    const int N1 = 0;
    static const float ref_[/* (N0 + N1) * 7 */] = {
0, 7,  0.606292f, 0.612037f, 0.149921f, 0.910763f, 0.300503f,
0, 16, 0.55195f,  0.17069f,  0.356024f, 0.471459f, 0.877178f,
0, 1,  0.433444f, 0.199235f, 0.301175f, 0.753253f, 0.744156f,
    };
    Mat ref(N0 + N1, 7, CV_32FC1, (void*)ref_);

    double scoreDiff = 8e-5, iouDiff = 3e-4;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CPU_FP16)
    {
        scoreDiff = 0.006;
        iouDiff = 0.042;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        scoreDiff = 0.04;
        iouDiff = 0.03;
    }
    std::string model_file = "yolov3.onnx";

    {
        SCOPED_TRACE("batch size 1");
        testYOLOModel(model_file, ref.rowRange(0, N0), scoreDiff, iouDiff, 0.24, 0.4, false, 0, Size(640, 640));
    }
}

TEST_P(Test_YOLO_nets, YOLOv4_tiny)
{
    applyTestTag(
        target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB
    );

    const double confThreshold = 0.5;
    // batchId, classId, confidence, left, top, right, bottom
    const int N0 = 3;
    const int N1 = 3;
    static const float ref_[/* (N0 + N1) * 7 */] = {
0, 16, 0.889883f, 0.177204f, 0.356279f, 0.417204f, 0.937517f,
0, 7, 0.816615f, 0.604293f, 0.137345f, 0.918016f, 0.295708f,
0, 1, 0.595912f, 0.0940107f, 0.178122f, 0.750619f, 0.829336f,

1, 2, 0.998224f, 0.652883f, 0.463477f, 0.813952f, 0.657163f,
1, 2, 0.967396f, 0.4539f, 0.466368f, 0.497716f, 0.520299f,
1, 0, 0.807866f, 0.205039f, 0.361842f, 0.260984f, 0.643621f,
    };
    Mat ref(N0 + N1, 7, CV_32FC1, (void*)ref_);

    double scoreDiff = 0.012f;
    double iouDiff = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CPU_FP16) ? 0.15 : 0.01f;
    if (target == DNN_TARGET_CUDA_FP16)
        iouDiff = 0.02;

    std::string model_file = "yolov4-tiny.onnx";

    {
        SCOPED_TRACE("batch size 1");
        testYOLOModel(model_file, ref.rowRange(0, N0), scoreDiff, iouDiff, confThreshold, 0.4, false, 0, Size(416, 416));
    }

    {
        SCOPED_TRACE("batch size 2");
        testYOLOModel(model_file, ref, scoreDiff, iouDiff, confThreshold, 0.4, false, 0, Size(416, 416));
    }
}

TEST_P(Test_YOLO_nets, YOLOv4x_mish)
{
    applyTestTag(
        CV_TEST_TAG_MEMORY_2GB,
        CV_TEST_TAG_LONG,
        CV_TEST_TAG_DEBUG_VERYLONG
    );

    // batchId, classId, confidence, left, top, right, bottom
    const int N0 = 3;
    const int N1 = 5;
    static const float ref_[/* (N0 + N1) * 7 */] = {
0, 1,  0.93241f,  0.161592f, 0.232638f, 0.738411f, 0.731285f,
0, 16, 0.929881f, 0.171312f, 0.385948f, 0.405568f, 0.940067f,
0, 7,  0.812158f, 0.60486f,  0.129621f, 0.895285f, 0.296402f,

1, 2, 0.929241f, 0.651517f, 0.457701f, 0.8147f,   0.660816f,
1, 0, 0.918966f, 0.200175f, 0.35915f,  0.265996f, 0.631935f,
1, 2, 0.881782f, 0.45082f,  0.461253f, 0.495884f, 0.522369f,
1, 9, 0.746081f, 0.661127f, 0.372649f, 0.686827f, 0.441998f,
1, 9, 0.730318f, 0.373671f, 0.314795f, 0.401108f, 0.397822f,
    };
    Mat ref(N0 + N1, 7, CV_32FC1, (void*)ref_);

    double scoreDiff = 8e-5;
    double iouDiff = 3e-4;

    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CUDA_FP16 || target == DNN_TARGET_CPU_FP16)
    {
        scoreDiff = 0.006;
        iouDiff = 0.042;
    }

    std::string model_file = "yolov4x-mish.onnx";

    {
        SCOPED_TRACE("batch size 1");
        testYOLOModel(model_file, ref.rowRange(0, N0), scoreDiff, iouDiff, 0.24, 0.4, false, 0, Size(640, 640));
    }

    {
        SCOPED_TRACE("batch size 2");
        testYOLOModel(model_file, ref, scoreDiff, iouDiff, 0.24, 0.4, false, 0, Size(640, 640));
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Test_YOLO_nets, dnnBackendsAndTargets());

}} // namespace
