// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include "npy_blob.hpp"
namespace opencv_test { namespace {

template<typename TString>
static std::string _tf(TString filename, bool required = true)
{
    String rootFolder = "dnn/";
    return findDataFile(rootFolder + filename, required);
}


class Test_Model : public DNNTestLayer
{
public:
    void testDetectModel(const std::string& weights, const std::string& cfg,
                         const std::string& imgPath, const std::vector<int>& refClassIds,
                         const std::vector<float>& refConfidences,
                         const std::vector<Rect2d>& refBoxes,
                         double scoreDiff, double iouDiff,
                         double confThreshold = 0.24, double nmsThreshold = 0.0,
                         const Size& size = {-1, -1}, Scalar mean = Scalar(),
                         double scale = 1.0, bool swapRB = false, bool crop = false)
    {
        checkBackend();

        Mat frame = imread(imgPath);
        DetectionModel model(weights, cfg);

        model.setInputSize(size).setInputMean(mean).setInputScale(scale)
             .setInputSwapRB(swapRB).setInputCrop(crop);

        model.setPreferableBackend(backend);
        model.setPreferableTarget(target);

        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<Rect> boxes;

        model.detect(frame, classIds, confidences, boxes, confThreshold, nmsThreshold);

        std::vector<Rect2d> boxesDouble(boxes.size());
        for (int i = 0; i < boxes.size(); i++) {
            boxesDouble[i] = boxes[i];
        }
        normAssertDetections(refClassIds, refConfidences, refBoxes, classIds,
                             confidences, boxesDouble, "",
                             confThreshold, scoreDiff, iouDiff);
    }

    void testClassifyModel(const std::string& weights, const std::string& cfg,
                           const std::string& imgPath, std::pair<int, float> ref, float norm,
                           const Size& size = {-1, -1}, Scalar mean = Scalar(),
                           double scale = 1.0, bool swapRB = false, bool crop = false)
    {
        checkBackend();

        Mat frame = imread(imgPath);
        ClassificationModel model(weights, cfg);
        model.setInputSize(size).setInputMean(mean).setInputScale(scale)
             .setInputSwapRB(swapRB).setInputCrop(crop);

        std::pair<int, float> prediction = model.classify(frame);
        EXPECT_EQ(prediction.first, ref.first);
        ASSERT_NEAR(prediction.second, ref.second, norm);
    }

    void testSegmentationModel(const std::string& weights_file, const std::string& config_file,
                               const std::string& inImgPath, const std::string& outImgPath,
                               float norm, const Size& size = {-1, -1}, Scalar mean = Scalar(),
                               double scale = 1.0, bool swapRB = false, bool crop = false)
    {
        checkBackend();

        Mat frame = imread(inImgPath);
        Mat mask;
        Mat exp = imread(outImgPath, 0);

        SegmentationModel model(weights_file, config_file);
        model.setInputSize(size).setInputMean(mean).setInputScale(scale)
             .setInputSwapRB(swapRB).setInputCrop(crop);

        model.segment(frame, mask);
        normAssert(mask, exp, "", norm, norm);
    }
};

TEST_P(Test_Model, Classify)
{
    std::pair<int, float> ref(652, 0.641789);

    std::string img_path = _tf("grace_hopper_227.png");
    std::string config_file = _tf("bvlc_alexnet.prototxt");
    std::string weights_file = _tf("bvlc_alexnet.caffemodel", false);

    Size size{227, 227};
    float norm = 1e-4;

    testClassifyModel(weights_file, config_file, img_path, ref, norm, size);
}


TEST_P(Test_Model, DetectRegion)
{
    applyTestTag(CV_TEST_TAG_LONG, CV_TEST_TAG_MEMORY_1GB);

#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2019010000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16);
#endif

#if defined(INF_ENGINE_RELEASE)
    if (target == DNN_TARGET_MYRIAD
        && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X);
#endif

    std::vector<int> refClassIds = {6, 1, 11};
    std::vector<float> refConfidences = {0.750469f, 0.780879f, 0.901615f};
    std::vector<Rect2d> refBoxes = {Rect2d(240, 53, 135, 72),
                                    Rect2d(112, 109, 192, 200),
                                    Rect2d(58, 141, 117, 249)};

    std::string img_path = _tf("dog416.png");
    std::string weights_file = _tf("yolo-voc.weights", false);
    std::string config_file = _tf("yolo-voc.cfg");

    double scale = 1.0 / 255.0;
    Size size{416, 416};
    bool swapRB = true;

    double confThreshold = 0.24;
    double scoreDiff = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 1e-2 : 8e-5;
    double iouDiff = (target == DNN_TARGET_MYRIAD || target == DNN_TARGET_OPENCL_FP16) ? 1.6e-2 : 1e-5;
    double nmsThreshold = (target == DNN_TARGET_MYRIAD) ? 0.397 : 0.4;

    testDetectModel(weights_file, config_file, img_path, refClassIds, refConfidences,
                    refBoxes, scoreDiff, iouDiff, confThreshold, nmsThreshold, size,
                    Scalar(), scale, swapRB);
}

TEST_P(Test_Model, DetectionOutput)
{
#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16);

    if (target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD);
#endif

    std::vector<int> refClassIds = {7, 12};
    std::vector<float> refConfidences = {0.991359f, 0.94786f};
    std::vector<Rect2d> refBoxes = {Rect2d(491, 81, 212, 98),
                                    Rect2d(132, 223, 207, 344)};

    std::string img_path = _tf("dog416.png");
    std::string weights_file = _tf("resnet50_rfcn_final.caffemodel", false);
    std::string config_file = _tf("rfcn_pascal_voc_resnet50.prototxt");

    Scalar mean = Scalar(102.9801, 115.9465, 122.7717);
    Size size{800, 600};

    double scoreDiff = (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16) ?
                        4e-3 : default_l1;
    double iouDiff = (target == DNN_TARGET_OPENCL_FP16) ? 1.8e-1 : 1e-5;
    float confThreshold = 0.8;
    double nmsThreshold = 0.0;

    testDetectModel(weights_file, config_file, img_path, refClassIds, refConfidences, refBoxes,
                    scoreDiff, iouDiff, confThreshold, nmsThreshold, size, mean);
}


TEST_P(Test_Model, DetectionMobilenetSSD)
{
    Mat ref = blobFromNPY(_tf("mobilenet_ssd_caffe_out.npy"));
    ref = ref.reshape(1, ref.size[2]);

    std::string img_path = _tf("street.png");
    Mat frame = imread(img_path);
    int frameWidth  = frame.cols;
    int frameHeight = frame.rows;

    std::vector<int> refClassIds;
    std::vector<float> refConfidences;
    std::vector<Rect2d> refBoxes;
    for (int i = 0; i < ref.rows; i++)
    {
        refClassIds.emplace_back(ref.at<float>(i, 1));
        refConfidences.emplace_back(ref.at<float>(i, 2));
        int left   = ref.at<float>(i, 3) * frameWidth;
        int top    = ref.at<float>(i, 4) * frameHeight;
        int right  = ref.at<float>(i, 5) * frameWidth;
        int bottom = ref.at<float>(i, 6) * frameHeight;
        int width  = right  - left + 1;
        int height = bottom - top + 1;
        refBoxes.emplace_back(left, top, width, height);
    }

    std::string weights_file = _tf("MobileNetSSD_deploy.caffemodel", false);
    std::string config_file = _tf("MobileNetSSD_deploy.prototxt");

    Scalar mean = Scalar(127.5, 127.5, 127.5);
    double scale = 1.0 / 127.5;
    Size size{300, 300};

    double scoreDiff = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 1.7e-2 : 1e-5;
    double iouDiff = (target == DNN_TARGET_OPENCL_FP16 || (target == DNN_TARGET_MYRIAD &&
                      getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)) ? 6.91e-2 : 1e-5;

    float confThreshold = FLT_MIN;
    double nmsThreshold = 0.0;

    testDetectModel(weights_file, config_file, img_path, refClassIds, refConfidences, refBoxes,
                    scoreDiff, iouDiff, confThreshold, nmsThreshold, size, mean, scale);
}

TEST_P(Test_Model, Detection_normalized)
{
    std::string img_path = _tf("grace_hopper_227.png");
    std::vector<int> refClassIds = {15};
    std::vector<float> refConfidences = {0.999222f};
    std::vector<Rect2d> refBoxes = {Rect2d(0, 4, 227, 222)};

    std::string weights_file = _tf("MobileNetSSD_deploy.caffemodel", false);
    std::string config_file = _tf("MobileNetSSD_deploy.prototxt");

    Scalar mean = Scalar(127.5, 127.5, 127.5);
    double scale = 1.0 / 127.5;
    Size size{300, 300};

    double scoreDiff = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 5e-3 : 1e-5;
    double iouDiff = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 0.09 : 1e-5;
    float confThreshold = FLT_MIN;
    double nmsThreshold = 0.0;
    testDetectModel(weights_file, config_file, img_path, refClassIds, refConfidences, refBoxes,
                    scoreDiff, iouDiff, confThreshold, nmsThreshold, size, mean, scale);
}

TEST_P(Test_Model, Segmentation)
{
    std::string inp = _tf("dog416.png");
    std::string weights_file = _tf("fcn8s-heavy-pascal.prototxt");
    std::string config_file = _tf("fcn8s-heavy-pascal.caffemodel", false);
    std::string exp = _tf("segmentation_exp.png");

    Size size{128, 128};
    float norm = 0;
    double scale = 1.0;
    Scalar mean = Scalar();
    bool swapRB = false;

    testSegmentationModel(weights_file, config_file, inp, exp, norm, size, mean, scale, swapRB);
}

INSTANTIATE_TEST_CASE_P(/**/, Test_Model, dnnBackendsAndTargets());

}} // namespace
