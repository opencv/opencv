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
                         double scale = 1.0, bool swapRB = false, bool crop = false,
                         bool nmsAcrossClasses = false)
    {
        checkBackend();

        Mat frame = imread(imgPath);
        DetectionModel model(weights, cfg);

        model.setInputSize(size).setInputMean(mean).setInputScale(scale)
             .setInputSwapRB(swapRB).setInputCrop(crop);

        model.setPreferableBackend(backend);
        model.setPreferableTarget(target);

        model.setNmsAcrossClasses(nmsAcrossClasses);
        if (target == DNN_TARGET_CPU_FP16)
            model.enableWinograd(false);

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

    void testKeypointsModel(const std::string& weights, const std::string& cfg,
                            const Mat& frame, const Mat& exp, float norm,
                            const Size& size = {-1, -1}, Scalar mean = Scalar(),
                            double scale = 1.0, bool swapRB = false, bool crop = false)
    {
        checkBackend();

        std::vector<Point2f> points;

        KeypointsModel model(weights, cfg);
        model.setInputSize(size).setInputMean(mean).setInputScale(scale)
             .setInputSwapRB(swapRB).setInputCrop(crop);

        model.setPreferableBackend(backend);
        model.setPreferableTarget(target);

        points = model.estimate(frame, 0.5);

        Mat out = Mat(points).reshape(1, (int)points.size());
        normAssert(exp, out, "", norm, norm);
    }

    void testSegmentationModel(const std::string& weights_file, const std::string& config_file,
                               const std::string& inImgPath, const std::string& outImgPath,
                               float norm, const Size& size = {-1, -1}, Scalar mean = Scalar(),
                               double scale = 1.0, bool swapRB = false, bool crop = false,
                               const std::vector<std::string>& outnames=std::vector<std::string>())
    {
        checkBackend();

        Mat frame = imread(inImgPath);
        Mat mask;
        Mat exp = imread(outImgPath, 0);

        SegmentationModel model(weights_file, config_file);
        model.setInputSize(size).setInputMean(mean).setInputScale(scale)
             .setInputSwapRB(swapRB).setInputCrop(crop);

        model.setPreferableBackend(backend);
        model.setPreferableTarget(target);

        if(!outnames.empty())
            model.setOutputNames(outnames);

        model.segment(frame, mask);
        normAssert(mask, exp, "", norm, norm);
    }

    void testTextRecognitionModel(const std::string& weights, const std::string& cfg,
                                  const std::string& imgPath, const std::string& seq,
                                  const std::string& decodeType, const std::vector<std::string>& vocabulary,
                                  const Size& size = {-1, -1}, Scalar mean = Scalar(),
                                  double scale = 1.0, bool swapRB = false, bool crop = false)
    {
        checkBackend();

        Mat frame = imread(imgPath, IMREAD_GRAYSCALE);

        TextRecognitionModel model(weights, cfg);
        model.setDecodeType(decodeType)
             .setVocabulary(vocabulary)
             .setInputSize(size).setInputMean(mean).setInputScale(scale)
             .setInputSwapRB(swapRB).setInputCrop(crop);

        model.setPreferableBackend(backend);
        model.setPreferableTarget(target);

        std::string result = model.recognize(frame);
        EXPECT_EQ(result, seq) << "Full frame: " << imgPath;

        std::vector<Rect> rois;
        rois.push_back(Rect(0, 0, frame.cols, frame.rows));
        rois.push_back(Rect(0, 0, frame.cols, frame.rows));  // twice
        std::vector<std::string> results;
        model.recognize(frame, rois, results);
        EXPECT_EQ((size_t)2u, results.size()) << "ROI: " << imgPath;
        EXPECT_EQ(results[0], seq) << "ROI[0]: " << imgPath;
        EXPECT_EQ(results[1], seq) << "ROI[1]: " << imgPath;
    }

    void testTextDetectionModelByDB(const std::string& weights, const std::string& cfg,
                                    const std::string& imgPath, const std::vector<std::vector<Point>>& gt,
                                    float binThresh, float polyThresh,
                                    uint maxCandidates, double unclipRatio,
                                    const Size& size = {-1, -1}, Scalar mean = Scalar(), Scalar scale = Scalar::all(1.0),
                                    double boxes_iou_diff = 0.05, bool swapRB = false, bool crop = false)
    {
        checkBackend();

        Mat frame = imread(imgPath);

        TextDetectionModel_DB model(weights, cfg);
        model.setBinaryThreshold(binThresh)
             .setPolygonThreshold(polyThresh)
             .setUnclipRatio(unclipRatio)
             .setMaxCandidates(maxCandidates)
             .setInputSize(size).setInputMean(mean).setInputScale(scale)
             .setInputSwapRB(swapRB).setInputCrop(crop);

        model.setPreferableBackend(backend);
        model.setPreferableTarget(target);

        // 1. Check common TextDetectionModel API through RotatedRect
        std::vector<cv::RotatedRect> results;
        model.detectTextRectangles(frame, results);

        EXPECT_GT(results.size(), (size_t)0);

        std::vector< std::vector<Point> > contours;
        for (size_t i = 0; i < results.size(); i++)
        {
            const RotatedRect& box = results[i];
            Mat contour;
            boxPoints(box, contour);
            std::vector<Point> contour2i(4);
            for (int i = 0; i < 4; i++)
            {
                contour2i[i].x = cvRound(contour.at<float>(i, 0));
                contour2i[i].y = cvRound(contour.at<float>(i, 1));
            }
            contours.push_back(contour2i);
        }
#if 0 // test debug
        Mat result = frame.clone();
        drawContours(result, contours, -1, Scalar(0, 0, 255), 1);
        imshow("result", result); // imwrite("result.png", result);
        waitKey(0);
#endif
        normAssertTextDetections(gt, contours, "", boxes_iou_diff);

        // 2. Check quadrangle-based API
        // std::vector< std::vector<Point> > contours;
        model.detect(frame, contours);

#if 0 // test debug
        Mat result = frame.clone();
        drawContours(result, contours, -1, Scalar(0, 0, 255), 1);
        imshow("result_contours", result); // imwrite("result_contours.png", result);
        waitKey(0);
#endif
        normAssertTextDetections(gt, contours, "", boxes_iou_diff);
    }

    void testTextDetectionModelByEAST(
            const std::string& weights, const std::string& cfg,
            const std::string& imgPath, const std::vector<RotatedRect>& gt,
            float confThresh, float nmsThresh,
            const Size& size = {-1, -1}, Scalar mean = Scalar(),
            double scale = 1.0, bool swapRB = false, bool crop = false,
            double eps_center = 5/*pixels*/, double eps_size = 5/*pixels*/, double eps_angle = 1
    )
    {
        checkBackend();

        Mat frame = imread(imgPath);

        TextDetectionModel_EAST model(weights, cfg);
        model.setConfidenceThreshold(confThresh)
             .setNMSThreshold(nmsThresh)
             .setInputSize(size).setInputMean(mean).setInputScale(scale)
             .setInputSwapRB(swapRB).setInputCrop(crop);

        model.setPreferableBackend(backend);
        model.setPreferableTarget(target);

        std::vector<cv::RotatedRect> results;
        model.detectTextRectangles(frame, results);

        EXPECT_EQ(results.size(), (size_t)1);
        for (size_t i = 0; i < results.size(); i++)
        {
            const RotatedRect& box = results[i];
#if 0 // test debug
            Mat contour;
            boxPoints(box, contour);
            std::vector<Point> contour2i(4);
            for (int i = 0; i < 4; i++)
            {
                contour2i[i].x = cvRound(contour.at<float>(i, 0));
                contour2i[i].y = cvRound(contour.at<float>(i, 1));
            }
            std::vector< std::vector<Point> > contours;
            contours.push_back(contour2i);

            Mat result = frame.clone();
            drawContours(result, contours, -1, Scalar(0, 0, 255), 1);
            imshow("result", result); //imwrite("result.png", result);
            waitKey(0);
#endif
            const RotatedRect& gtBox = gt[i];
            EXPECT_NEAR(box.center.x, gtBox.center.x, eps_center);
            EXPECT_NEAR(box.center.y, gtBox.center.y, eps_center);
            EXPECT_NEAR(box.size.width, gtBox.size.width, eps_size);
            EXPECT_NEAR(box.size.height, gtBox.size.height, eps_size);
            EXPECT_NEAR(box.angle, gtBox.angle, eps_angle);
        }
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
    applyTestTag(
        CV_TEST_TAG_MEMORY_2GB,
        CV_TEST_TAG_LONG,
        CV_TEST_TAG_DEBUG_VERYLONG
    );

#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // accuracy
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // accuracy
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2020040000)  // nGraph compilation failure
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2019010000)
    // FIXIT DNN_BACKEND_INFERENCE_ENGINE is misused
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
    double nmsThreshold = (target == DNN_TARGET_MYRIAD) ? 0.397 : 0.4;
    double scoreDiff = 8e-5, iouDiff = 1e-5;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CUDA_FP16 || target == DNN_TARGET_CPU_FP16)
    {
        scoreDiff = 1e-2;
        iouDiff = 1.6e-2;
    }

    testDetectModel(weights_file, config_file, img_path, refClassIds, refConfidences,
                    refBoxes, scoreDiff, iouDiff, confThreshold, nmsThreshold, size,
                    Scalar(), scale, swapRB);
}

TEST_P(Test_Model, DetectRegionWithNmsAcrossClasses)
{
    applyTestTag(
        CV_TEST_TAG_MEMORY_2GB,
        CV_TEST_TAG_LONG,
        CV_TEST_TAG_DEBUG_VERYLONG
    );

#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // accuracy
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // accuracy
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2020040000)  // nGraph compilation failure
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2019010000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16);
#endif

#if defined(INF_ENGINE_RELEASE)
    if (target == DNN_TARGET_MYRIAD
        && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X);
#endif

    std::vector<int> refClassIds = { 6, 11 };
    std::vector<float> refConfidences = { 0.750469f, 0.901615f };
    std::vector<Rect2d> refBoxes = { Rect2d(240, 53, 135, 72),
                                    Rect2d(58, 141, 117, 249) };

    std::string img_path = _tf("dog416.png");
    std::string weights_file = _tf("yolo-voc.weights", false);
    std::string config_file = _tf("yolo-voc.cfg");

    double scale = 1.0 / 255.0;
    Size size{ 416, 416 };
    bool swapRB = true;
    bool crop = false;
    bool nmsAcrossClasses = true;

    double confThreshold = 0.24;
    double nmsThreshold = (target == DNN_TARGET_MYRIAD) ? 0.15: 0.15;
    double scoreDiff = 8e-5, iouDiff = 1e-5;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CUDA_FP16 || target == DNN_TARGET_CPU_FP16)
    {
        scoreDiff = 1e-2;
        iouDiff = 1.6e-2;
    }

    testDetectModel(weights_file, config_file, img_path, refClassIds, refConfidences,
        refBoxes, scoreDiff, iouDiff, confThreshold, nmsThreshold, size,
        Scalar(), scale, swapRB, crop,
        nmsAcrossClasses);
}

TEST_P(Test_Model, DetectionOutput)
{
    applyTestTag(CV_TEST_TAG_DEBUG_VERYLONG);

#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // Check 'backward_compatible_check || in_out_elements_equal' failed at core/src/op/reshape.cpp:427:
    // While validating node 'v1::Reshape bbox_pred_reshape (ave_bbox_pred_rois[0]:f32{1,8,1,1}, Constant_388[0]:i64{4}) -> (f32{?,?,?,?})' with friendly_name 'bbox_pred_reshape':
    // Requested output shape {1,300,8,1} is incompatible with input shape {1, 8, 1, 1}
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // Exception: Function contains several inputs and outputs with one friendly name! (HETERO bug?)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target != DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);

    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE)
    // FIXIT DNN_BACKEND_INFERENCE_ENGINE is misused
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

    double scoreDiff = default_l1, iouDiff = 1e-5;
    float confThreshold = 0.8;
    double nmsThreshold = 0.0;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_CUDA_FP16 || target == DNN_TARGET_CPU_FP16)
    {
        if (backend == DNN_BACKEND_OPENCV)
            scoreDiff = 4e-3;
        else
            scoreDiff = 2e-2;
        iouDiff = 1.8e-1;
    }
#if defined(INF_ENGINE_RELEASE)
        if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        {
            scoreDiff = 0.05;
            iouDiff = 0.08;
        }
#endif

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

    std::string weights_file = _tf("MobileNetSSD_deploy_19e3ec3.caffemodel", false);
    std::string config_file = _tf("MobileNetSSD_deploy_19e3ec3.prototxt");

    Scalar mean = Scalar(127.5, 127.5, 127.5);
    double scale = 1.0 / 127.5;
    Size size{300, 300};

    double scoreDiff = 1e-5, iouDiff = 1e-5;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_CPU_FP16)
    {
        scoreDiff = 1.7e-2;
        iouDiff = 6.91e-2;
    }
    else if (target == DNN_TARGET_MYRIAD)
    {
        scoreDiff = 0.017;
        if (getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
            iouDiff = 0.1;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        scoreDiff = 0.0028;
        iouDiff = 1e-2;
    }
    float confThreshold = FLT_MIN;
    double nmsThreshold = 0.0;

    testDetectModel(weights_file, config_file, img_path, refClassIds, refConfidences, refBoxes,
                    scoreDiff, iouDiff, confThreshold, nmsThreshold, size, mean, scale);
}

TEST_P(Test_Model, Keypoints_pose)
{
    if (target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_CPU_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CPU_FP16);
#ifdef HAVE_INF_ENGINE
    if (target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif

    Mat inp = imread(_tf("pose.png"));
    std::string weights = _tf("onnx/models/lightweight_pose_estimation_201912.onnx", false);
    float kpdata[] = {
        237.65625f, 78.25f, 237.65625f, 136.9375f,
        190.125f, 136.9375f, 142.59375f, 195.625f, 79.21875f, 176.0625f, 285.1875f, 117.375f,
        348.5625f, 195.625f, 396.09375f, 176.0625f, 205.96875f, 313.0f, 205.96875f, 430.375f,
        205.96875f, 528.1875f, 269.34375f, 293.4375f, 253.5f, 430.375f, 237.65625f, 528.1875f,
        221.8125f, 58.6875f, 253.5f, 58.6875f, 205.96875f, 78.25f, 253.5f, 58.6875f
    };
    Mat exp(18, 2, CV_32FC1, kpdata);

    Size size{256, 256};
    float norm = 1e-4;
    double scale = 1.0/255;
    Scalar mean = Scalar(128, 128, 128);
    bool swapRB = false;

    // Ref. Range: [58.6875, 508.625]
    if (target == DNN_TARGET_CUDA_FP16)
        norm = 20; // l1 = 1.5, lInf = 20

    testKeypointsModel(weights, "", inp, exp, norm, size, mean, scale, swapRB);
}

TEST_P(Test_Model, Keypoints_face)
{
#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif

    Mat inp = imread(_tf("gray_face.png"), 0);
    std::string weights = _tf("onnx/models/facial_keypoints.onnx", false);
    Mat exp = blobFromNPY(_tf("facial_keypoints_exp.npy"));

    Size size{224, 224};
    double scale = 1.0/255;
    Scalar mean = Scalar();
    bool swapRB = false;

    // Ref. Range: [-1.1784188, 1.7758257]
    float norm = 1e-4;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_CPU_FP16)
        norm = 5e-3;
    if (target == DNN_TARGET_MYRIAD)
    {
        // Myriad2: l1 = 0.0004, lInf = 0.002
        // MyriadX: l1 = 0.003, lInf = 0.009
        norm = 0.009;
    }
    if (target == DNN_TARGET_CUDA_FP16)
        norm = 0.004; // l1 = 0.0006, lInf = 0.004

    testKeypointsModel(weights, "", inp, exp, norm, size, mean, scale, swapRB);
}

TEST_P(Test_Model, Detection_normalized)
{
    std::string img_path = _tf("grace_hopper_227.png");
    std::vector<int> refClassIds = {15};
    std::vector<float> refConfidences = {0.999222f};
    std::vector<Rect2d> refBoxes = {Rect2d(0, 4, 227, 222)};

    std::string weights_file = _tf("MobileNetSSD_deploy_19e3ec3.caffemodel", false);
    std::string config_file = _tf("MobileNetSSD_deploy_19e3ec3.prototxt");

    Scalar mean = Scalar(127.5, 127.5, 127.5);
    double scale = 1.0 / 127.5;
    Size size{300, 300};

    double scoreDiff = 1e-5, iouDiff = 1e-5;
    float confThreshold = FLT_MIN;
    double nmsThreshold = 0.0;
    if (target == DNN_TARGET_CUDA)
    {
        scoreDiff = 3e-4;
        iouDiff = 0.018;
    }
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CUDA_FP16 || target == DNN_TARGET_CPU_FP16)
    {
        scoreDiff = 5e-3;
        iouDiff = 0.09;
    }
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2020040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
    {
        scoreDiff = 0.02;
        iouDiff = 0.1f;
    }
#endif
    testDetectModel(weights_file, config_file, img_path, refClassIds, refConfidences, refBoxes,
                    scoreDiff, iouDiff, confThreshold, nmsThreshold, size, mean, scale);
}

TEST_P(Test_Model, Segmentation)
{
    applyTestTag(
        CV_TEST_TAG_MEMORY_2GB,
        CV_TEST_TAG_DEBUG_VERYLONG
    );

    float norm = 0;

#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // Failed to allocate graph: NC_ERROR
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    // accuracy
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
    {
        norm = 25.0f;  // depends on OS/OpenCL version
    }
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // Failed to allocate graph: NC_ERROR
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    // cnn_network_ngraph_impl.cpp:104 Function contains several inputs and outputs with one friendly name: 'upscore2'!
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    // cnn_network_ngraph_impl.cpp:104 Function contains several inputs and outputs with one friendly name: 'upscore2'!
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE)
    // Failed to allocate graph: NC_ERROR
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif

    //if ((backend == DNN_BACKEND_OPENCV && (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_CPU_FP16))
    //    || (backend == DNN_BACKEND_CUDA && target == DNN_TARGET_CUDA_FP16))
    {
        // let's always set it to 7 for now
        norm = 7.0f;  // l1 = 0.01 lInf = 7
    }

    std::string inp = _tf("dog416.png");
    std::string weights_file = _tf("onnx/models/fcn-resnet50-12.onnx", false);
    std::string exp = _tf("segmentation_exp.png");

    Size size{128, 128};
    double scale = 0.019;
    Scalar mean = Scalar(0.485*255, 0.456*255, 0.406*255);
    bool swapRB = true;

    testSegmentationModel(weights_file, "", inp, exp, norm, size, mean, scale, swapRB, false);
}

TEST_P(Test_Model, TextRecognition)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // FIXIT: dnn/src/ie_ngraph.cpp:494: error: (-215:Assertion failed) !inps.empty() in function 'createNet'
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_CPU, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    // Node Transpose_79 was not assigned on any pointed device
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // IE Exception: Ngraph operation Reshape with name 71 has dynamic output shape on 0 port, but CPU plug-in supports only static shape
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
#endif

    std::string imgPath = _tf("text_rec_test.png");
    std::string weightPath = _tf("onnx/models/crnn.onnx", false);
    std::string seq = "welcome";

    Size size{100, 32};
    double scale = 1.0 / 127.5;
    Scalar mean = Scalar(127.5);
    std::string decodeType = "CTC-greedy";
    std::vector<std::string> vocabulary = {"0","1","2","3","4","5","6","7","8","9",
                                           "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"};

    testTextRecognitionModel(weightPath, "", imgPath, seq, decodeType, vocabulary, size, mean, scale);
}

TEST_P(Test_Model, TextRecognitionWithCTCPrefixBeamSearch)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // Node Transpose_79 was not assigned on any pointed device
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // IE Exception: Ngraph operation Reshape with name 71 has dynamic output shape on 0 port, but CPU plug-in supports only static shape
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
#endif


    std::string imgPath = _tf("text_rec_test.png");
    std::string weightPath = _tf("onnx/models/crnn.onnx", false);
    std::string seq = "welcome";

    Size size{100, 32};
    double scale = 1.0 / 127.5;
    Scalar mean = Scalar(127.5);
    std::string decodeType = "CTC-prefix-beam-search";
    std::vector<std::string> vocabulary = {"0","1","2","3","4","5","6","7","8","9",
                                           "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"};

    testTextRecognitionModel(weightPath, "", imgPath, seq, decodeType, vocabulary, size, mean, scale);
}

TEST_P(Test_Model, TextDetectionByDB)
{
    applyTestTag(CV_TEST_TAG_DEBUG_VERYLONG);

    if (target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_CPU_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CPU_FP16);

    std::string imgPath = _tf("text_det_test1.png");
    std::string weightPathDB = _tf("onnx/models/DB_TD500_resnet50.onnx", false);
    std::string weightPathPPDB = _tf("onnx/models/PP_OCRv3_DB_text_det.onnx", false);

    // GroundTruth
    std::vector<std::vector<Point>> gt = {
        { Point(142, 193), Point(136, 164), Point(213, 150), Point(219, 178) },
        { Point(136, 165), Point(122, 114), Point(319, 71), Point(330, 122) }
    };

    Size size{736, 736};
    Scalar scaleDB = Scalar::all(1.0 / 255.0);
    Scalar meanDB = Scalar(122.67891434, 116.66876762, 104.00698793);

    // new mean and stddev
    Scalar meanPPDB = Scalar(123.675, 116.28, 103.53);
    Scalar stddevPPDB = Scalar(0.229, 0.224, 0.225);
    Scalar scalePPDB = scaleDB / stddevPPDB;

    float binThresh = 0.3;
    float polyThresh = 0.5;
    uint maxCandidates = 200;
    double unclipRatio = 2.0;

    {
    SCOPED_TRACE("Original DB");
    testTextDetectionModelByDB(weightPathDB, "", imgPath, gt, binThresh, polyThresh, maxCandidates, unclipRatio, size, meanDB, scaleDB, 0.05f);
    }

    {
    SCOPED_TRACE("PP-OCRDBv3");
    testTextDetectionModelByDB(weightPathPPDB, "", imgPath, gt, binThresh, polyThresh, maxCandidates, unclipRatio, size, meanPPDB, scalePPDB, 0.21f);
    }
}

TEST_P(Test_Model, TextDetectionByEAST)
{
    applyTestTag(CV_TEST_TAG_DEBUG_VERYLONG);

    std::string imgPath = _tf("text_det_test2.jpg");
    std::string weightPath = _tf("frozen_east_text_detection.pb", false);

    // GroundTruth
    std::vector<RotatedRect> gt = {
        RotatedRect(Point2f(657.55f, 409.5f), Size2f(316.84f, 62.45f), -4.79)
    };

    // Model parameters
    Size size{320, 320};
    double scale = 1.0;
    Scalar mean = Scalar(123.68, 116.78, 103.94);
    bool swapRB = true;

    // Detection algorithm parameters
    float confThresh = 0.5;
    float nmsThresh = 0.4;

    double eps_center = 5/*pixels*/;
    double eps_size = 5/*pixels*/;
    double eps_angle = 1;

    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_CUDA_FP16 || target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CPU_FP16)
    {
        eps_center = 10;
        eps_size = 25;
        eps_angle = 3;
    }

    testTextDetectionModelByEAST(weightPath, "", imgPath, gt, confThresh, nmsThresh, size, mean, scale, swapRB, false/*crop*/,
        eps_center, eps_size, eps_angle
    );
}

INSTANTIATE_TEST_CASE_P(/**/, Test_Model, dnnBackendsAndTargets());

static void topK(const Mat& probs, std::vector<std::pair<int, float> >& result, int K)
{
    CV_Assert(probs.type() == CV_32F);
    CV_Assert(probs.dims == 2 && probs.rows == 1);
    int N = int(probs.total());
    K = std::min(K, N);
    std::vector<std::pair<float, int> > pairs(N);
    for (int i = 0; i < N; i++) {
        pairs[i] = {-probs.at<float>(i), i};
    }
    std::partial_sort(pairs.begin(), pairs.begin() + K, pairs.end());
    result.resize(K);
    for (int i = 0; i < K; i++) {
        result[i] = {pairs[i].second, -pairs[i].first};
    }
}

typedef testing::TestWithParam<Target> Reproducibility_ResNet50_ONNX;
TEST_P(Reproducibility_ResNet50_ONNX, Accuracy)
{
    Target targetId = GetParam();
    applyTestTag(targetId == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB);
    ASSERT_TRUE(ocl::useOpenCL() || targetId == DNN_TARGET_CPU || targetId == DNN_TARGET_CPU_FP16);

    std::string modelname = _tf("onnx/models/resnet50v1.onnx", false);
    Net net = readNetFromONNX(modelname);

    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(targetId);

    if (targetId == DNN_TARGET_CPU_FP16)
        net.enableWinograd(false);

    //net.dumpToStream(std::cout);
    //net.setTracingMode(DNN_TRACE_ALL);

    std::string imgname = _tf("sqcat.png");
    Mat image = imread(imgname);
    Mat input = blobFromImage(image, 0.017, Size(224,224),
                              Scalar(103.939, 116.779, 123.68),
                              false, true, CV_32F);
    ASSERT_TRUE(!input.empty());

    net.setInput(input);
    Mat out = net.forward();

    std::vector<std::pair<int, float> > ref = {{285, 10.13}, {287, 9.68}, {283, 8.83}, {278, 8.56}, {279, 8.34}};
    std::vector<std::pair<int, float> > res;
    const int K = 5;

    topK(out, res, K);
    const float eps = 0.15f;

    ASSERT_EQ(int(res.size()), K);

    std::vector<int> reflabels(K), reslabels(K);
    for (int i = 0; i < K; i++) {
        reflabels[i] = ref[i].first;
        reslabels[i] = res[i].first;
    }
    ASSERT_EQ(reflabels, reslabels);

    for (int i = 0; i < K; i++) {
        EXPECT_NEAR(ref[i].second, res[i].second, eps);
    }
}
INSTANTIATE_TEST_CASE_P(/**/, Reproducibility_ResNet50_ONNX,
                        testing::ValuesIn(getAvailableTargets(DNN_BACKEND_OPENCV)));

typedef testing::TestWithParam<Target> Reproducibility_ResNet50_QDQ_ONNX;
TEST_P(Reproducibility_ResNet50_QDQ_ONNX, Accuracy)
{
    Target targetId = GetParam();
    applyTestTag(targetId == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB);
    ASSERT_TRUE(ocl::useOpenCL() || targetId == DNN_TARGET_CPU || targetId == DNN_TARGET_CPU_FP16);

    std::string modelname = _tf("onnx/models/resnet50-v1-12-qdq.onnx", false);
    Net net = readNetFromONNX(modelname);

    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(targetId);

    if (targetId == DNN_TARGET_CPU_FP16)
        net.enableWinograd(false);

    std::string imgname = _tf("sqcat.png");
    Mat image = imread(imgname);
    Mat input = blobFromImage(image, 0.017, Size(224,224),
                              Scalar(103.939, 116.779, 123.68),
                              false, true, CV_32F);
    ASSERT_TRUE(!input.empty());

    net.setInput(input);
    Mat out = net.forward();

    const int K = 5;
    std::vector<std::pair<int, float> > res;
    topK(out, res, K);
    ASSERT_EQ(int(res.size()), K);

    std::vector<std::pair<int, float> > ref = {{285, 10.44}, {287, 10.13}, {283, 8.89}, {278, 8.43}};
    const float eps = 0.5f;

    for (int i = 0; i < (int)ref.size(); i++) {
        EXPECT_EQ(ref[i].first, res[i].first);
        EXPECT_NEAR(ref[i].second, res[i].second, eps);
    }
}
INSTANTIATE_TEST_CASE_P(/**/, Reproducibility_ResNet50_QDQ_ONNX,
                        testing::ValuesIn(getAvailableTargets(DNN_BACKEND_OPENCV)));

typedef testing::TestWithParam<Target> Reproducibility_MobileNetSSD_ONNX;
TEST_P(Reproducibility_MobileNetSSD_ONNX, Accuracy)
{
    Target targetId = GetParam();
    auto engine_forced = static_cast<EngineType>(
        cv::utils::getConfigurationParameterSizeT("OPENCV_FORCE_DNN_ENGINE", ENGINE_AUTO));
    if (engine_forced == ENGINE_CLASSIC)
    {
        applyTestTag(CV_TEST_TAG_DNN_SKIP_PARSER);
        return;
    }

    applyTestTag(targetId == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB);
    ASSERT_TRUE(ocl::useOpenCL() || targetId == DNN_TARGET_CPU || targetId == DNN_TARGET_CPU_FP16);

    std::string modelname = _tf("onnx/models/ssd_mobilenet_v1_12.onnx", false);
    Net net = readNetFromONNX(modelname);

    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(targetId);

    if (targetId == DNN_TARGET_CPU_FP16)
        net.enableWinograd(false);

    std::string imgname = _tf("dog_orig_size.png");
    Mat image = imread(imgname);
    ASSERT_TRUE(!image.empty());
    Mat input;
    resize(image, input, Size(300, 300));
    int imsize[] = {1, input.rows, input.cols, 3};
    Mat input8dim4(4, imsize, CV_8U, input.data);

    std::vector<String> outNames = net.getUnconnectedOutLayersNames();
    std::vector<Mat> outs;
    net.setInput(input8dim4);
    net.forward(outs, outNames);

    // Model outputs: detection_boxes [1,N,4], detection_classes [1,N],
    //                detection_scores [1,N], num_detections [1]
    ASSERT_EQ(outs.size(), (size_t)4);

    Mat boxes, classes, scores, numDet;
    for (size_t i = 0; i < outs.size(); i++) {
        if (outs[i].dims == 3 && outs[i].size[2] == 4)
            boxes = outs[i];
        else if (outs[i].total() == 1)
            numDet = outs[i];
        else if (outs[i].dims == 2) {
            float first = outs[i].at<float>(0, 0);
            if (first == std::round(first) && first > 0)
                classes = outs[i];
            else
                scores = outs[i];
        }
    }
    ASSERT_FALSE(boxes.empty());
    ASSERT_FALSE(scores.empty());
    ASSERT_FALSE(classes.empty());
    ASSERT_FALSE(numDet.empty());

    int ndet = (int)numDet.at<float>(0);
    printf("num_detections = %d\n", ndet);
    ASSERT_GT(ndet, 0);

    // Build test detection vectors from model outputs
    // Model boxes are normalized (y1, x1, y2, x2) — convert to Rect2d(x, y, w, h)
    std::vector<int> testClassIds;
    std::vector<float> testScores;
    std::vector<Rect2d> testBoxes;
    for (int j = 0; j < ndet; j++) {
        testClassIds.push_back((int)classes.at<float>(0, j));
        testScores.push_back(scores.at<float>(0, j));
        float y1 = boxes.at<float>(0, j, 0);
        float x1 = boxes.at<float>(0, j, 1);
        float y2 = boxes.at<float>(0, j, 2);
        float x2 = boxes.at<float>(0, j, 3);
        testBoxes.push_back(Rect2d(x1, y1, x2 - x1, y2 - y1));
    }

    // Reference detections for dog_orig_size.png
    // COCO 1-indexed: 2=bicycle, 3=car, 18=dog
    std::vector<int> refClassIds = {2, 18, 3};
    std::vector<float> refScores = {0.944377f, 0.877805f, 0.787824f};
    std::vector<Rect2d> refBoxes = {
        Rect2d(0.157917, 0.219984, 0.742909 - 0.157917, 0.739280 - 0.219984),  // bicycle
        Rect2d(0.168082, 0.360803, 0.426304 - 0.168082, 0.919625 - 0.360803),  // dog
        Rect2d(0.600506, 0.114612, 0.899101 - 0.600506, 0.298757 - 0.114612),  // car
    };

    float confThreshold = 0.5f;
    double scoreDiff = 0.1;
    double iouDiff = 0.05;
    normAssertDetections(refClassIds, refScores, refBoxes,
                         testClassIds, testScores, testBoxes,
                         "", confThreshold, scoreDiff, iouDiff);
}
INSTANTIATE_TEST_CASE_P(/**/, Reproducibility_MobileNetSSD_ONNX,
                        testing::ValuesIn(getAvailableTargets(DNN_BACKEND_OPENCV)));


namespace {

enum YoloFormat { YOLO_V5, YOLO_V8, YOLO_X };
static void YoloPostprocess(const Mat& out, YoloFormat fmt, int inputSize,
                       float confTh, float nmsTh,
                       std::vector<int>& classIds,
                       std::vector<float>& confidences,
                       std::vector<Rect2d>& boxes)
{
    CV_Assert(out.dims == 3);
    bool hasObj  = (fmt != YOLO_V8);
    int nclasses = 80;
    int stride   = 4 + (hasObj ? 1 : 0) + nclasses;

    const float* data = nullptr;
    std::vector<float> buf;
    int N;
    if (fmt == YOLO_V8) {
        int C = out.size[1];
        N = out.size[2];
        CV_Assert(C == stride);
        const float* src = out.ptr<float>();
        buf.resize((size_t)N * C);
        for (int i = 0; i < C; i++)
            for (int j = 0; j < N; j++)
                buf[j * C + i] = src[i * N + j];
        data = buf.data();
    } else {
        N = out.size[1];
        CV_Assert(out.size[2] == stride);
        data = out.ptr<float>();
    }

    // YOLOX grid decode tables
    std::vector<float> gridX, gridY, strideVec;
    if (fmt == YOLO_X) {
        const int strides[] = {8, 16, 32};
        gridX.resize(N); gridY.resize(N); strideVec.resize(N);
        int idx = 0;
        for (int si = 0; si < 3; si++) {
            int gs = inputSize / strides[si];
            for (int y = 0; y < gs; y++)
                for (int x = 0; x < gs; x++) {
                    gridX[idx] = (float)x;
                    gridY[idx] = (float)y;
                    strideVec[idx] = (float)strides[si];
                    idx++;
                }
        }
        CV_Assert(idx == N);
    }

    int classOff = hasObj ? 5 : 4;
    double scale = 1.0 / inputSize;
    std::vector<Rect> intBoxes;
    std::vector<int> allCls;
    std::vector<float> allConf;
    std::vector<Rect2d> allBoxes;

    for (int i = 0; i < N; i++) {
        const float* r = data + (size_t)i * stride;
        float obj = hasObj ? r[4] : 1.0f;
        if (obj < confTh) continue;

        int bestCls = 0; float bestScore = 0;
        for (int c = 0; c < nclasses; c++) {
            float s = r[classOff + c] * obj;
            if (s > bestScore) { bestScore = s; bestCls = c; }
        }
        if (bestScore < confTh) continue;

        float cx, cy, w, h;
        if (fmt == YOLO_X) {
            cx = (r[0] + gridX[i]) * strideVec[i];
            cy = (r[1] + gridY[i]) * strideVec[i];
            w  = std::exp(r[2]) * strideVec[i];
            h  = std::exp(r[3]) * strideVec[i];
        } else {
            cx = r[0]; cy = r[1]; w = r[2]; h = r[3];
        }

        intBoxes.push_back(Rect((int)(cx-w/2), (int)(cy-h/2), (int)w, (int)h));
        allConf.push_back(bestScore);
        allCls.push_back(bestCls);
        allBoxes.push_back(Rect2d((cx-w/2)*scale, (cy-h/2)*scale, w*scale, h*scale));
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(intBoxes, allConf, confTh, nmsTh, indices);
    for (int idx : indices) {
        classIds.push_back(allCls[idx]);
        confidences.push_back(allConf[idx]);
        boxes.push_back(allBoxes[idx]);
    }
}

} // local namespace

typedef testing::TestWithParam<Target> Reproducibility_YOLOv5n_ONNX;
TEST_P(Reproducibility_YOLOv5n_ONNX, Accuracy)
{
    Target targetId = GetParam();
    applyTestTag(targetId == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB);
    ASSERT_TRUE(ocl::useOpenCL() || targetId == DNN_TARGET_CPU || targetId == DNN_TARGET_CPU_FP16);

    std::string modelname = _tf("yolov5n.onnx", false);
    Net net = readNetFromONNX(modelname);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(targetId);
    if (targetId == DNN_TARGET_CPU_FP16)
        net.enableWinograd(false);

    std::string imgname = _tf("dog416.png");
    Mat image = imread(imgname);
    ASSERT_TRUE(!image.empty());
    Mat input = blobFromImage(image, 1.0/255.0, Size(640, 640), Scalar(), true, false, CV_32F);

    net.setInput(input);
    Mat out = net.forward();

    if (out.type() != CV_32F) out.convertTo(out, CV_32F);

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect2d> testBoxes;
    YoloPostprocess(out, YOLO_V5, 640, 0.25f, 0.45f, classIds, confidences, testBoxes);

    std::vector<int>    refClassIds  = {16, 2, 1, 1};
    std::vector<float>  refScores    = {0.711f, 0.581f, 0.344f, 0.275f};
    std::vector<Rect2d> refBoxes     = {
        Rect2d(0.168262, 0.374023, 0.247852, 0.577734),  // dog
        Rect2d(0.605469, 0.134375, 0.286719, 0.156250),  // car
        Rect2d(0.186279, 0.248828, 0.148926, 0.129688),  // bicycle (small)
        Rect2d(0.231836, 0.277930, 0.519141, 0.483203),  // bicycle (large)
    };

    normAssertDetections(refClassIds, refScores, refBoxes,
                         classIds, confidences, testBoxes,
                         "", 0.25f, /*scoreDiff=*/0.2, /*iouDiff=*/0.2);
}
INSTANTIATE_TEST_CASE_P(/**/, Reproducibility_YOLOv5n_ONNX,
                        testing::ValuesIn(getAvailableTargets(DNN_BACKEND_OPENCV)));


typedef testing::TestWithParam<Target> Reproducibility_YOLOv8n_ONNX;
TEST_P(Reproducibility_YOLOv8n_ONNX, Accuracy)
{
    Target targetId = GetParam();
    applyTestTag(targetId == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB);
    ASSERT_TRUE(ocl::useOpenCL() || targetId == DNN_TARGET_CPU || targetId == DNN_TARGET_CPU_FP16);

    std::string modelname = _tf("yolov8n.onnx", false);
    Net net = readNetFromONNX(modelname);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(targetId);
    if (targetId == DNN_TARGET_CPU_FP16)
        net.enableWinograd(false);

    std::string imgname = _tf("dog416.png");
    Mat image = imread(imgname);
    ASSERT_TRUE(!image.empty());
    Mat input = blobFromImage(image, 1.0/255.0, Size(640, 640), Scalar(), true, false, CV_32F);

    net.setInput(input);
    Mat out = net.forward();

    if (out.type() != CV_32F) out.convertTo(out, CV_32F);

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect2d> testBoxes;
    YoloPostprocess(out, YOLO_V8, 640, 0.25f, 0.45f, classIds, confidences, testBoxes);

    std::vector<int>    refClassIds  = {16, 1, 7};
    std::vector<float>  refScores    = {0.827f, 0.809f, 0.544f};
    std::vector<Rect2d> refBoxes     = {
        Rect2d(0.171157, 0.386951, 0.231909, 0.551873),  // dog
        Rect2d(0.160967, 0.234788, 0.577899, 0.495077),  // bicycle
        Rect2d(0.608337, 0.130141, 0.291832, 0.167390),  // truck
    };

    normAssertDetections(refClassIds, refScores, refBoxes,
                         classIds, confidences, testBoxes,
                         "", 0.25f, /*scoreDiff=*/0.1, /*iouDiff=*/0.1);
}
INSTANTIATE_TEST_CASE_P(/**/, Reproducibility_YOLOv8n_ONNX,
                        testing::ValuesIn(getAvailableTargets(DNN_BACKEND_OPENCV)));


typedef testing::TestWithParam<Target> Reproducibility_YOLOXS_ONNX;
TEST_P(Reproducibility_YOLOXS_ONNX, Accuracy)
{
    Target targetId = GetParam();
    applyTestTag(targetId == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB);
    ASSERT_TRUE(ocl::useOpenCL() || targetId == DNN_TARGET_CPU || targetId == DNN_TARGET_CPU_FP16);

    std::string modelname = _tf("yolox_s.onnx", false);
    Net net = readNetFromONNX(modelname);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(targetId);
    if (targetId == DNN_TARGET_CPU_FP16)
        net.enableWinograd(false);

    std::string imgname = _tf("dog416.png");
    Mat image = imread(imgname);
    ASSERT_TRUE(!image.empty());
    Mat input = blobFromImage(image, 1.0, Size(640, 640), Scalar(), false, false, CV_32F);

    net.setInput(input);
    Mat out = net.forward();

    if (out.type() != CV_32F) out.convertTo(out, CV_32F);

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect2d> testBoxes;
    YoloPostprocess(out, YOLO_X, 640, 0.25f, 0.45f, classIds, confidences, testBoxes);

    std::vector<int>    refClassIds  = {1, 16, 7, 1};
    std::vector<float>  refScores    = {0.962f, 0.920f, 0.833f, 0.266f};
    std::vector<Rect2d> refBoxes     = {
        Rect2d(0.160787, 0.225276, 0.577830, 0.503752),  // bicycle (large)
        Rect2d(0.172622, 0.386773, 0.230225, 0.554768),  // dog
        Rect2d(0.601869, 0.128871, 0.302539, 0.168476),  // truck
        Rect2d(0.166281, 0.251719, 0.339791, 0.385267),  // bicycle (small)
    };

    normAssertDetections(refClassIds, refScores, refBoxes,
                         classIds, confidences, testBoxes,
                         "", 0.25f, /*scoreDiff=*/0.2, /*iouDiff=*/0.2);
}
INSTANTIATE_TEST_CASE_P(/**/, Reproducibility_YOLOXS_ONNX,
                        testing::ValuesIn(getAvailableTargets(DNN_BACKEND_OPENCV)));


}} // namespace
