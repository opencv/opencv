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
                       int zeroPadW = 0)
    {
        checkBackend();

        Mat img1 = imread(_tf("dog416.png"));
        Mat img2 = imread(_tf("street.png"));
        cv::resize(img2, img2, Size(416, 416));

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

        for (int b = 0; b < batch_size; ++b)
        {
            std::vector<int> classIds;
            std::vector<float> confidences;
            std::vector<Rect2d> boxes;
            for (int i = 0; i < outs.size(); ++i)
            {
                Mat out;
                if (batch_size > 1){
                    // get the sample slice from 3D matrix (batch, box, classes+5)
                    Range ranges[3] = {Range(b, b+1), Range::all(), Range::all()};
                    out = outs[i](ranges).reshape(1, outs[i].size[1]);
                }else{
                    out = outs[i];
                }
                for (int j = 0; j < out.rows; ++j)
                {
                    Mat scores = out.row(j).colRange(5, out.cols);
                    double confidence;
                    Point maxLoc;
                    minMaxLoc(scores, 0, &confidence, 0, &maxLoc);

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
                       int zeroPadW = 0)
    {
        testYOLOModel(model,
                      std::vector<std::vector<int> >(1, refClassIds),
                      std::vector<std::vector<float> >(1, refConfidences),
                      std::vector<std::vector<Rect2d> >(1, refBoxes),
                      scoreDiff, iouDiff, confThreshold, nmsThreshold, useWinograd, zeroPadW);
    }

    void testYOLOModel(const std::string& model,
                       const cv::Mat& ref, double scoreDiff, double iouDiff,
                       float confThreshold = 0.24, float nmsThreshold = 0.4, bool useWinograd = true,
                       int zeroPadW = 0)
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
            if (batchId >= refClassIds.size())
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
                      scoreDiff, iouDiff, confThreshold, nmsThreshold, useWinograd, zeroPadW);
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
    const int N1 = 7;
    static const float ref_[/* (N0 + N1) * 7 */] = {
0, 16, 0.992194f, 0.172375f, 0.402458f, 0.403918f, 0.932801f,
0, 1, 0.988326f, 0.166708f, 0.228236f, 0.737208f, 0.735803f,
0, 7, 0.94639f, 0.602523f, 0.130399f, 0.901623f, 0.298452f,

1, 2, 0.99761f, 0.646556f, 0.45985f, 0.816041f, 0.659067f,
1, 0, 0.988913f, 0.201726f, 0.360282f, 0.266181f, 0.631728f,
1, 2, 0.98233f, 0.452007f, 0.462217f, 0.495612f, 0.521687f,
1, 9, 0.919195f, 0.374642f, 0.316524f, 0.398126f, 0.393714f,
1, 9, 0.856303f, 0.666842f, 0.372215f, 0.685539f, 0.44141f,
1, 9, 0.313516f, 0.656791f, 0.374734f, 0.671959f, 0.438371f,
1, 9, 0.256625f, 0.940232f, 0.326931f, 0.967586f, 0.374002f,
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
        testYOLOModel(model_file, ref.rowRange(0, N0), scoreDiff, iouDiff, 0.24, 0.4, false);
    }

    {
        SCOPED_TRACE("batch size 2");
        testYOLOModel(model_file, ref, scoreDiff, iouDiff, 0.24, 0.4, false);
    }
}

TEST_P(Test_YOLO_nets, YoloVoc)
{
    applyTestTag(
#if defined(OPENCV_32BIT_CONFIGURATION) && defined(HAVE_OPENCL)
        CV_TEST_TAG_MEMORY_2GB,
#else
        CV_TEST_TAG_MEMORY_1GB,
#endif
        CV_TEST_TAG_LONG,
        CV_TEST_TAG_DEBUG_VERYLONG
    );

    // batchId, classId, confidence, left, top, right, bottom
    Mat ref = (Mat_<float>(6, 7) << 0, 6,  0.750469f, 0.577374f, 0.127391f, 0.902949f, 0.300809f,  // a car
                                    0, 1,  0.780879f, 0.270762f, 0.264102f, 0.732475f, 0.745412f,  // a bicycle
                                    0, 11, 0.901615f, 0.1386f,   0.338509f, 0.421337f, 0.938789f,  // a dog
                                    1, 14, 0.623813f, 0.183179f, 0.381921f, 0.247726f, 0.625847f,  // a person
                                    1, 6,  0.667770f, 0.446555f, 0.453578f, 0.499986f, 0.519167f,  // a car
                                    1, 6,  0.844947f, 0.637058f, 0.460398f, 0.828508f, 0.66427f);  // a car

    double nmsThreshold = (target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CPU_FP16) ? 0.397 : 0.4;
    double scoreDiff = 8e-5, iouDiff = 3e-4;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CPU_FP16)
    {
        scoreDiff = 1e-2;
        iouDiff = 0.018;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        scoreDiff = 0.03;
        iouDiff = 0.018;
    }

    std::string model_file = "yolo-voc.onnx";

    {
    SCOPED_TRACE("batch size 1");
    testYOLOModel(model_file, ref.rowRange(0, 3), scoreDiff, iouDiff, 0.24, 0.4, false);
    }

    {
    SCOPED_TRACE("batch size 2");
    testYOLOModel(model_file, ref, scoreDiff, iouDiff, 0.24, nmsThreshold, false);
    }
}

TEST_P(Test_YOLO_nets, TinyYoloVoc)
{
    applyTestTag(CV_TEST_TAG_MEMORY_512MB);

    // batchId, classId, confidence, left, top, right, bottom
    Mat ref = (Mat_<float>(4, 7) << 0, 6,  0.761967f, 0.579042f, 0.159161f, 0.894482f, 0.31994f,   // a car
                                    0, 11, 0.780595f, 0.129696f, 0.386467f, 0.445275f, 0.920994f,  // a dog
                                    1, 6,  0.651450f, 0.460526f, 0.458019f, 0.522527f, 0.5341f,    // a car
                                    1, 6,  0.928758f, 0.651024f, 0.463539f, 0.823784f, 0.654998f); // a car

    double scoreDiff = 8e-5, iouDiff = 3e-4;
    bool useWinograd = true;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD)
    {
        scoreDiff = 8e-3;
        iouDiff = 0.018;
    }
    else if(target == DNN_TARGET_CUDA_FP16)
    {
        scoreDiff = 0.008;
        iouDiff = 0.02;
    }
    else if (target == DNN_TARGET_CPU_FP16)
    {
        useWinograd = false;
        scoreDiff = 8e-3;
        iouDiff = 0.018;
    }

    std::string model_file = "tiny-yolo-voc.onnx";

    {
    SCOPED_TRACE("batch size 1");
    testYOLOModel(model_file, ref.rowRange(0, 2), scoreDiff, iouDiff, 0.24, 0.4, useWinograd);
    }

    {
    SCOPED_TRACE("batch size 2");
    testYOLOModel(model_file, ref, scoreDiff, iouDiff, 0.24, 0.4, useWinograd);
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
    const int N1 = 6;
    static const float ref_[/* (N0 + N1) * 7 */] = {
0, 16, 0.998836f, 0.160024f, 0.389964f, 0.417885f, 0.943716f,
0, 1, 0.987908f, 0.150913f, 0.221933f, 0.742255f, 0.746261f,
0, 7, 0.952983f, 0.614621f, 0.150257f, 0.901368f, 0.289251f,

1, 2, 0.997412f, 0.647584f, 0.459939f, 0.821037f, 0.663947f,
1, 2, 0.989633f, 0.450719f, 0.463353f, 0.496306f, 0.522258f,
1, 0, 0.980053f, 0.195856f, 0.378454f, 0.258626f, 0.629257f,
1, 9, 0.785341f, 0.665503f, 0.373543f, 0.688893f, 0.439244f,
1, 9, 0.733275f, 0.376029f, 0.315694f, 0.401776f, 0.395165f,
1, 9, 0.384815f, 0.659824f, 0.372389f, 0.673927f, 0.429412f,
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
        testYOLOModel(model_file, ref.rowRange(0, N0), scoreDiff, iouDiff, 0.24, 0.4, false);
    }

    {
        SCOPED_TRACE("batch size 2");
        testYOLOModel(model_file, ref, scoreDiff, iouDiff, 0.24, 0.4, false);
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

    std::string model_file = "yolov4-tiny-2020-12.onnx";

    {
        SCOPED_TRACE("batch size 1");
        testYOLOModel(model_file, ref.rowRange(0, N0), scoreDiff, iouDiff, confThreshold, 0.4, false);
    }

    {
        SCOPED_TRACE("batch size 2");
        testYOLOModel(model_file, ref, scoreDiff, iouDiff, confThreshold, 0.4, false);
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
0, 16, 0.925536f, 0.17188f,  0.386832f, 0.406138f, 0.941696f,
0, 1,  0.912028f, 0.162125f, 0.208863f, 0.741316f, 0.729332f,
0, 7,  0.841018f, 0.608953f, 0.128653f, 0.900692f, 0.295657f,

1, 2, 0.925697f, 0.650438f, 0.458118f, 0.813927f, 0.661775f,
1, 0, 0.882156f, 0.203644f, 0.365763f, 0.265473f, 0.632195f,
1, 2, 0.848857f, 0.451044f, 0.462997f, 0.496629f, 0.522719f,
1, 9, 0.736015f, 0.374503f, 0.316029f, 0.399358f, 0.392883f,
1, 9, 0.727129f, 0.662469f, 0.373687f, 0.687877f, 0.441335f,
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
        testYOLOModel(model_file, ref.rowRange(0, N0), scoreDiff, iouDiff, 0.24, 0.4, false);
    }

    {
        SCOPED_TRACE("batch size 2");
        testYOLOModel(model_file, ref, scoreDiff, iouDiff, 0.24, 0.4, false);
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Test_YOLO_nets, dnnBackendsAndTargets());

}} // namespace
