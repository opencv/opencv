
#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/features.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/dnn.hpp"
#include <cfloat>

#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#endif

#define ENABLE_LOG 1
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

using namespace std;
using namespace cv;
using namespace cv::detail;

namespace
{
static Mat toGrayForXFeat(const Mat& image)
{
    if (image.channels() == 1)
        return image;
    Mat gray;
    if (image.channels() == 3)
        cvtColor(image, gray, COLOR_BGR2GRAY);
    else if (image.channels() == 4)
        cvtColor(image, gray, COLOR_BGRA2GRAY);
    else
        CV_Error(Error::StsBadArg, "XFeat expects grayscale, BGR, or BGRA image");
    return gray;
}

static Mat toNCHWForXFeat(const Mat& blob, int channelsHint = -1)
{
    CV_Assert(blob.dims == 4);
    if (channelsHint > 0 && blob.size[1] == channelsHint)
        return blob;

    const bool isNHWC = (channelsHint > 0) ? (blob.size[3] == channelsHint) : (blob.size[3] < blob.size[1]);
    if (!isNHWC)
        return blob;

    const int H = blob.size[1], W = blob.size[2], C = blob.size[3];
    int outSize[] = {1, C, H, W};
    Mat out(4, outSize, CV_32F);
    const float* src = blob.ptr<float>();
    float* dst = out.ptr<float>();
    for (int c = 0; c < C; ++c)
    {
        for (int y = 0; y < H; ++y)
        {
            for (int x = 0; x < W; ++x)
            {
                const int srcIdx = (y * W + x) * C + c;
                const int dstIdx = c * H * W + y * W + x;
                dst[dstIdx] = src[srcIdx];
            }
        }
    }
    return out;
}

static float sampleNearestForXFeat(const Mat& map, float x, float y, int normW, int normH)
{
    CV_Assert(map.type() == CV_32F);
    if (normW <= 1 || normH <= 1 || map.empty())
        return 0.0f;
    const float fx = x * static_cast<float>(map.cols - 1) / static_cast<float>(normW - 1);
    const float fy = y * static_cast<float>(map.rows - 1) / static_cast<float>(normH - 1);
    const int ix = std::max(0, std::min(map.cols - 1, cvRound(fx)));
    const int iy = std::max(0, std::min(map.rows - 1, cvRound(fy)));
    return map.at<float>(iy, ix);
}

static float sampleBilinearForXFeat(const Mat& map, float x, float y, int normW, int normH)
{
    CV_Assert(map.type() == CV_32F);
    if (map.empty() || normW <= 1 || normH <= 1)
        return 0.0f;

    float fx = x * static_cast<float>(map.cols - 1) / static_cast<float>(normW - 1);
    float fy = y * static_cast<float>(map.rows - 1) / static_cast<float>(normH - 1);
    fx = std::max(0.0f, std::min(fx, static_cast<float>(map.cols - 1)));
    fy = std::max(0.0f, std::min(fy, static_cast<float>(map.rows - 1)));

    const int x0 = cvFloor(fx), y0 = cvFloor(fy);
    const int x1 = std::min(x0 + 1, map.cols - 1);
    const int y1 = std::min(y0 + 1, map.rows - 1);
    const float dx = fx - x0, dy = fy - y0;

    const float v00 = map.at<float>(y0, x0);
    const float v01 = map.at<float>(y0, x1);
    const float v10 = map.at<float>(y1, x0);
    const float v11 = map.at<float>(y1, x1);
    return (1.f - dx) * (1.f - dy) * v00 +
           dx * (1.f - dy) * v01 +
           (1.f - dx) * dy * v10 +
           dx * dy * v11;
}

static void computeImageFeaturesXFeat(const Mat& image, ImageFeatures& features, dnn::Net& net,
                                      int maxKeypoints = 4096, float detectionThreshold = 0.05f,
                                      int inputSize = 640)
{
    features.keypoints.clear();
    features.descriptors.release();
    features.img_size = image.size();
    if (image.empty())
        return;

    Mat gray = toGrayForXFeat(image);
    const float scale = static_cast<float>(inputSize) / static_cast<float>(std::max(gray.cols, gray.rows));
    const int resizedW = std::max(1, cvRound(gray.cols * scale));
    const int resizedH = std::max(1, cvRound(gray.rows * scale));
    Mat resized;
    resize(gray, resized, Size(resizedW, resizedH), 0, 0, INTER_LINEAR_EXACT);
    Mat padded = Mat::zeros(inputSize, inputSize, CV_8UC1);
    resized.copyTo(padded(Rect(0, 0, resizedW, resizedH)));

    Mat blob;
    dnn::blobFromImage(padded, blob, 1.0 / 255.0, Size(inputSize, inputSize), Scalar(), false, false, CV_32F);
    net.setInput(blob);
    vector<Mat> outs;
    net.forward(outs, {"output_feats", "output_keypoints", "output_heatmap"});
    CV_Assert(outs.size() == 3);

    Mat featBlob = toNCHWForXFeat(outs[0], 64);
    Mat kptBlob = toNCHWForXFeat(outs[1], 65);
    Mat relBlob = toNCHWForXFeat(outs[2], 1);
    CV_Assert(featBlob.dims == 4 && kptBlob.dims == 4 && relBlob.dims == 4);

    const int featH = featBlob.size[2], featW = featBlob.size[3];
    const int kptC = kptBlob.size[1], kptH = kptBlob.size[2], kptW = kptBlob.size[3];
    CV_Assert(featBlob.size[1] == 64 && kptC >= 64);

    Mat reliability(relBlob.size[2], relBlob.size[3], CV_32F, relBlob.ptr<float>());
    Mat heatmap = Mat::zeros(kptH * 8, kptW * 8, CV_32F);
    const float* kptPtr = kptBlob.ptr<float>();
    const int kptHW = kptH * kptW;

    for (int y = 0; y < kptH; ++y)
    {
        for (int x = 0; x < kptW; ++x)
        {
            const int offset = y * kptW + x;
            float maxLogit = -FLT_MAX;
            for (int c = 0; c < kptC; ++c)
                maxLogit = std::max(maxLogit, kptPtr[c * kptHW + offset]);

            float sumExp = 0.f;
            float probs[64];
            for (int c = 0; c < 64; ++c)
            {
                probs[c] = std::exp(kptPtr[c * kptHW + offset] - maxLogit);
                sumExp += probs[c];
            }
            if (kptC > 64)
                sumExp += std::exp(kptPtr[64 * kptHW + offset] - maxLogit);
            if (sumExp <= 0.f)
                continue;

            for (int c = 0; c < 64; ++c)
            {
                const int dy = c / 8;
                const int dx = c % 8;
                heatmap.at<float>(y * 8 + dy, x * 8 + dx) = probs[c] / sumExp;
            }
        }
    }

    Mat localMax;
    dilate(heatmap, localMax, getStructuringElement(MORPH_RECT, Size(5, 5)));

    struct Candidate { Point2f ptPadded; float score; };
    vector<Candidate> candidates;
    candidates.reserve(4096);

    for (int y = 0; y < heatmap.rows; ++y)
    {
        const float* hm = heatmap.ptr<float>(y);
        const float* mx = localMax.ptr<float>(y);
        for (int x = 0; x < heatmap.cols; ++x)
        {
            const float h = hm[x];
            if (h <= detectionThreshold || h != mx[x])
                continue;
            const float xf = static_cast<float>(x), yf = static_cast<float>(y);
            const float score = sampleNearestForXFeat(heatmap, xf, yf, inputSize, inputSize) *
                                sampleBilinearForXFeat(reliability, xf, yf, inputSize, inputSize);
            if (score > 0.f)
                candidates.push_back({Point2f(xf, yf), score});
        }
    }

    if (candidates.empty())
        return;

    const int keep = maxKeypoints > 0 ? std::min(maxKeypoints, static_cast<int>(candidates.size()))
                                      : static_cast<int>(candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + keep, candidates.end(),
                      [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
    candidates.resize(keep);

    Mat descriptors(keep, 64, CV_32F);
    const float* featPtr = featBlob.ptr<float>();
    const int featHW = featH * featW;

    for (int i = 0; i < keep; ++i)
    {
        const float xp = candidates[i].ptPadded.x;
        const float yp = candidates[i].ptPadded.y;
        const float xOrig = xp / scale;
        const float yOrig = yp / scale;
        const int ix = cvFloor(xOrig), iy = cvFloor(yOrig);
        if (ix < 0 || iy < 0 || ix >= image.cols || iy >= image.rows)
            continue;

        float* dst = descriptors.ptr<float>(static_cast<int>(features.keypoints.size()));
        for (int c = 0; c < 64; ++c)
        {
            Mat channel(featH, featW, CV_32F, const_cast<float*>(featPtr + c * featHW));
            dst[c] = sampleBilinearForXFeat(channel, xp, yp, inputSize, inputSize);
        }
        Mat row(1, 64, CV_32F, dst);
        normalize(row, row, 1.0, 0.0, NORM_L2);
        features.keypoints.emplace_back(Point2f(xOrig, yOrig), 1.0f, -1.0f, candidates[i].score);
    }

    if (features.keypoints.empty())
        return;

    Mat validDesc = descriptors.rowRange(0, static_cast<int>(features.keypoints.size())).clone();
    validDesc.copyTo(features.descriptors);
}
} // namespace

static void printUsage(char** argv)
{
    cout <<
        "Rotation model images stitcher.\n\n"
         << argv[0] << " img1 img2 [...imgN] [flags]\n\n"
        "Flags:\n"
        "  --preview\n"
        "      Run stitching in the preview mode. Works faster than usual mode,\n"
        "      but output image will have lower resolution.\n"
        "  --try_cuda (yes|no)\n"
        "      Try to use CUDA. The default value is 'no'. All default values\n"
        "      are for CPU mode.\n"
        "\nMotion Estimation Flags:\n"
        "  --work_megapix <float>\n"
        "      Resolution for image registration step. The default is 0.6 Mpx.\n"
        "  --features (surf|orb|sift|akaze|aliked|xfeat)\n"
        "      Type of features used for images matching.\n"
        "      The default is surf if available, orb otherwise.\n"
        "      When using 'aliked', requires --matcher lightglue and DNN model paths.\n"
        "      When using 'xfeat', requires --xfeat_model and uses the standard matcher.\n"
        "  --matcher (homography|affine|lightglue)\n"
        "      Matcher used for pairwise image matching.\n"
        "  --estimator (homography|affine)\n"
        "      Type of estimator used for transformation estimation.\n"
        "  --match_conf <float>\n"
        "      Confidence for feature matching step. The default is 0.65 for surf and 0.3 for orb.\n"
        "  --conf_thresh <float>\n"
        "      Threshold for two images are from the same panorama confidence.\n"
        "      The default is 1.0.\n"
        "  --ba (no|reproj|ray|affine)\n"
        "      Bundle adjustment cost function. The default is ray.\n"
        "  --ba_refine_mask (mask)\n"
        "      Set refinement mask for bundle adjustment. It looks like 'x_xxx',\n"
        "      where 'x' means refine respective parameter and '_' means don't\n"
        "      refine one, and has the following format:\n"
        "      <fx><skew><ppx><aspect><ppy>. The default mask is 'xxxxx'. If bundle\n"
        "      adjustment doesn't support estimation of selected parameter then\n"
        "      the respective flag is ignored.\n"
        "  --wave_correct (no|horiz|vert)\n"
        "      Perform wave effect correction. The default is 'horiz'.\n"
        "  --save_graph <file_name>\n"
        "      Save matches graph represented in DOT language to <file_name> file.\n"
        "      Labels description: Nm is number of matches, Ni is number of inliers,\n"
        "      C is confidence.\n"
        "\nCompositing Flags:\n"
        "  --warp (affine|plane|cylindrical|spherical|fisheye|stereographic|compressedPlaneA2B1|compressedPlaneA1.5B1|compressedPlanePortraitA2B1|compressedPlanePortraitA1.5B1|paniniA2B1|paniniA1.5B1|paniniPortraitA2B1|paniniPortraitA1.5B1|mercator|transverseMercator)\n"
        "      Warp surface type. The default is 'spherical'.\n"
        "  --seam_megapix <float>\n"
        "      Resolution for seam estimation step. The default is 0.1 Mpx.\n"
        "  --seam (no|voronoi|gc_color|gc_colorgrad)\n"
        "      Seam estimation method. The default is 'gc_color'.\n"
        "  --compose_megapix <float>\n"
        "      Resolution for compositing step. Use -1 for original resolution.\n"
        "      The default is -1.\n"
        "  --expos_comp (no|gain|gain_blocks|channels|channels_blocks)\n"
        "      Exposure compensation method. The default is 'gain_blocks'.\n"
        "  --expos_comp_nr_feeds <int>\n"
        "      Number of exposure compensation feed. The default is 1.\n"
        "  --expos_comp_nr_filtering <int>\n"
        "      Number of filtering iterations of the exposure compensation gains.\n"
        "      Only used when using a block exposure compensation method.\n"
        "      The default is 2.\n"
        "  --expos_comp_block_size <int>\n"
        "      BLock size in pixels used by the exposure compensator.\n"
        "      Only used when using a block exposure compensation method.\n"
        "      The default is 32.\n"
        "  --blend (no|feather|multiband)\n"
        "      Blending method. The default is 'multiband'.\n"
        "  --blend_strength <float>\n"
        "      Blending strength from [0,100] range. The default is 5.\n"
        "  --output <result_img>\n"
        "      The default is 'result.jpg'.\n"
        "  --timelapse (as_is|crop) \n"
        "      Output warped images separately as frames of a time lapse movie, with 'fixed_' prepended to input file names.\n"
        "  --rangewidth <int>\n"
        "      uses range_width to limit number of images to match with.\n"
        "\nDNN Feature Options:\n"
        "  --aliked_model <path>\n"
        "      Path to ALIKED ONNX model file.\n"
        "  --lightglue_model <path>\n"
        "      Path to LightGlue ONNX model file (for ALIKED descriptors).\n"
        "  --lg_score_thresh <float>\n"
        "      LightGlue confidence threshold. The default is 0.0 (accept all).\n"
        "  --xfeat_model <path>\n"
        "      Path to XFeat ONNX model file.\n";
}


// Default command line args
vector<String> img_names;
bool preview = false;
bool try_cuda = false;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 1.f;
#ifdef HAVE_OPENCV_XFEATURES2D
string features_type = "surf";
float match_conf = 0.65f;
#else
string features_type = "orb";
float match_conf = 0.3f;
#endif
string matcher_type = "homography";
string estimator_type = "homography";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string warp_type = "spherical";
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
int expos_comp_nr_feeds = 1;
int expos_comp_nr_filtering = 2;
int expos_comp_block_size = 32;
string seam_find_type = "gc_color";
int blend_type = Blender::MULTI_BAND;
int timelapse_type = Timelapser::AS_IS;
float blend_strength = 5;
string result_name = "result.jpg";
bool timelapse = false;
int range_width = -1;
String aliked_model_path;
String lightglue_model_path;
String xfeat_model_path;
float lg_score_thresh = 0.0f;


static int parseCmdArgs(int argc, char** argv)
{
    if (argc == 1)
    {
        printUsage(argv);
        return -1;
    }
    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
        {
            printUsage(argv);
            return -1;
        }
        else if (string(argv[i]) == "--preview")
        {
            preview = true;
        }
        else if (string(argv[i]) == "--try_cuda")
        {
            if (string(argv[i + 1]) == "no")
                try_cuda = false;
            else if (string(argv[i + 1]) == "yes")
                try_cuda = true;
            else
            {
                cout << "Bad --try_cuda flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--work_megapix")
        {
            work_megapix = atof(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--seam_megapix")
        {
            seam_megapix = atof(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--compose_megapix")
        {
            compose_megapix = atof(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--result")
        {
            result_name = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--features")
        {
            features_type = argv[i + 1];
            if (string(features_type) == "orb")
                match_conf = 0.3f;
            i++;
        }
        else if (string(argv[i]) == "--matcher")
        {
            if (string(argv[i + 1]) == "homography" || string(argv[i + 1]) == "affine" || string(argv[i + 1]) == "lightglue")
                matcher_type = argv[i + 1];
            else
            {
                cout << "Bad --matcher flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--estimator")
        {
            if (string(argv[i + 1]) == "homography" || string(argv[i + 1]) == "affine")
                estimator_type = argv[i + 1];
            else
            {
                cout << "Bad --estimator flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--match_conf")
        {
            match_conf = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--conf_thresh")
        {
            conf_thresh = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--ba")
        {
            ba_cost_func = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--ba_refine_mask")
        {
            ba_refine_mask = argv[i + 1];
            if (ba_refine_mask.size() != 5)
            {
                cout << "Incorrect refinement mask length.\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--wave_correct")
        {
            if (string(argv[i + 1]) == "no")
                do_wave_correct = false;
            else if (string(argv[i + 1]) == "horiz")
            {
                do_wave_correct = true;
                wave_correct = detail::WAVE_CORRECT_HORIZ;
            }
            else if (string(argv[i + 1]) == "vert")
            {
                do_wave_correct = true;
                wave_correct = detail::WAVE_CORRECT_VERT;
            }
            else
            {
                cout << "Bad --wave_correct flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--save_graph")
        {
            save_graph = true;
            save_graph_to = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--warp")
        {
            warp_type = string(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--expos_comp")
        {
            if (string(argv[i + 1]) == "no")
                expos_comp_type = ExposureCompensator::NO;
            else if (string(argv[i + 1]) == "gain")
                expos_comp_type = ExposureCompensator::GAIN;
            else if (string(argv[i + 1]) == "gain_blocks")
                expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
            else if (string(argv[i + 1]) == "channels")
                expos_comp_type = ExposureCompensator::CHANNELS;
            else if (string(argv[i + 1]) == "channels_blocks")
                expos_comp_type = ExposureCompensator::CHANNELS_BLOCKS;
            else
            {
                cout << "Bad exposure compensation method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--expos_comp_nr_feeds")
        {
            expos_comp_nr_feeds = atoi(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--expos_comp_nr_filtering")
        {
            expos_comp_nr_filtering = atoi(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--expos_comp_block_size")
        {
            expos_comp_block_size = atoi(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--seam")
        {
            if (string(argv[i + 1]) == "no" ||
                string(argv[i + 1]) == "voronoi" ||
                string(argv[i + 1]) == "gc_color" ||
                string(argv[i + 1]) == "gc_colorgrad" ||
                string(argv[i + 1]) == "dp_color" ||
                string(argv[i + 1]) == "dp_colorgrad")
                seam_find_type = argv[i + 1];
            else
            {
                cout << "Bad seam finding method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--blend")
        {
            if (string(argv[i + 1]) == "no")
                blend_type = Blender::NO;
            else if (string(argv[i + 1]) == "feather")
                blend_type = Blender::FEATHER;
            else if (string(argv[i + 1]) == "multiband")
                blend_type = Blender::MULTI_BAND;
            else
            {
                cout << "Bad blending method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--timelapse")
        {
            timelapse = true;

            if (string(argv[i + 1]) == "as_is")
                timelapse_type = Timelapser::AS_IS;
            else if (string(argv[i + 1]) == "crop")
                timelapse_type = Timelapser::CROP;
            else
            {
                cout << "Bad timelapse method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--rangewidth")
        {
            range_width = atoi(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--blend_strength")
        {
            blend_strength = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--output")
        {
            result_name = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--aliked_model")
        {
            aliked_model_path = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--lightglue_model")
        {
            lightglue_model_path = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--xfeat_model")
        {
            xfeat_model_path = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--lg_score_thresh")
        {
            lg_score_thresh = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else
            img_names.push_back(argv[i]);
    }
    if (preview)
    {
        compose_megapix = 0.6;
    }

    // Validate DNN options
    if (features_type == "aliked" && matcher_type != "lightglue")
    {
        cout << "Error: --features aliked requires --matcher lightglue\n";
        return -1;
    }
    if (features_type == "aliked" && (aliked_model_path.empty() || lightglue_model_path.empty()))
    {
        cout << "Error: --features aliked requires --aliked_model and --lightglue_model\n";
        return -1;
    }
    if (features_type == "xfeat" && xfeat_model_path.empty())
    {
        cout << "Error: --features xfeat requires --xfeat_model\n";
        return -1;
    }
    if (features_type == "xfeat" && matcher_type == "lightglue")
    {
        cout << "Error: --features xfeat does not support --matcher lightglue; use homography or affine\n";
        return -1;
    }

    return 0;
}


int main(int argc, char* argv[])
{
#if ENABLE_LOG
    int64 app_start_time = getTickCount();
#endif

#if 0
    cv::setBreakOnError(true);
#endif

    int retval = parseCmdArgs(argc, argv);
    if (retval)
        return retval;

    // Disable OpenCL for DNN-based features to avoid backend sync issues
    bool use_aliked = (features_type == "aliked");
    bool use_xfeat = (features_type == "xfeat");
    if (use_aliked || use_xfeat)
        cv::ocl::setUseOpenCL(false);

    // Check if have enough images
    int num_images = static_cast<int>(img_names.size());
    if (num_images < 2)
    {
        LOGLN("Need more images");
        return -1;
    }

    double work_scale = 1, seam_scale = 1, compose_scale = 1;
    bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;

    LOGLN("Finding features...");
#if ENABLE_LOG
    int64 t = getTickCount();
#endif

    Ptr<Feature2D> finder;
    dnn::Net xfeatNet;
    if (use_aliked)
    {
        // ALIKED will be created per-image in the loop below
    }
    else if (features_type == "orb")
    {
        finder = ORB::create();
    }
    else if (features_type == "akaze")
    {
#ifdef HAVE_OPENCV_XFEATURES2D
        finder = xfeatures2d::AKAZE::create();
#else
        cout << "OpenCV is built without opencv_contrib modules. AKAZE algorithm is not available!" << std::endl;
        return -1;
#endif
    }
    else if (features_type == "surf")
    {
#if defined(HAVE_OPENCV_XFEATURES2D) && defined(HAVE_OPENCV_NONFREE)
        finder = xfeatures2d::SURF::create();
#else
        cout << "OpenCV is built without NONFREE modules. SURF algorithm is not available!" << std::endl;
        return -1;
#endif
    }
    else if (features_type == "sift")
    {
        finder = SIFT::create();
    }
    else if (features_type == "xfeat")
    {
#ifdef HAVE_OPENCV_DNN
        xfeatNet = dnn::readNetFromONNX(xfeat_model_path);
#else
        cout << "OpenCV is built without opencv_dnn module. XFeat algorithm is not available!" << std::endl;
        return -1;
#endif
    }
    else
    {
        cout << "Unknown 2D features type: '" << features_type << "'.\n";
        return -1;
    }

    Mat full_img, img;
    vector<ImageFeatures> features(num_images);
    vector<Mat> images(num_images);
    vector<Size> full_img_sizes(num_images);
    double seam_work_aspect = 1;

    for (int i = 0; i < num_images; ++i)
    {
        full_img = imread(samples::findFile(img_names[i]));
        full_img_sizes[i] = full_img.size();

        if (full_img.empty())
        {
            LOGLN("Can't open image " << img_names[i]);
            return -1;
        }
        if (work_megapix < 0)
        {
            img = full_img;
            work_scale = 1;
            is_work_scale_set = true;
        }
        else
        {
            if (!is_work_scale_set)
            {
                work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
                is_work_scale_set = true;
            }
            resize(full_img, img, Size(), work_scale, work_scale, INTER_LINEAR_EXACT);
        }
        if (!is_seam_scale_set)
        {
            seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }

        if (use_aliked)
        {
            Ptr<ALIKED> aliked = ALIKED::create(aliked_model_path);
            computeImageFeatures(aliked, img, features[i]);
        }
        else if (use_xfeat)
        {
            computeImageFeaturesXFeat(img, features[i], xfeatNet);
        }
        else
        {
            computeImageFeatures(finder, img, features[i]);
        }
        features[i].img_idx = i;
        LOGLN("Features in image #" << i+1 << ": " << features[i].keypoints.size());

        resize(full_img, img, Size(), seam_scale, seam_scale, INTER_LINEAR_EXACT);
        images[i] = img.clone();
    }

    full_img.release();
    img.release();

    LOGLN("Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    LOG("Pairwise matching");
#if ENABLE_LOG
    t = getTickCount();
#endif
    vector<MatchesInfo> pairwise_matches;
    Ptr<FeaturesMatcher> matcher;
    if (use_aliked && matcher_type == "lightglue")
    {
        Ptr<LightGlueMatcher> lg = LightGlueMatcher::create(lightglue_model_path);
        Ptr<LightGlueFeaturesMatcher> lgMatcher = makePtr<LightGlueFeaturesMatcher>(lg);
        lgMatcher->setScoreThreshold(lg_score_thresh);
        matcher = lgMatcher;
    }
    else if (matcher_type == "affine")
        matcher = makePtr<AffineBestOf2NearestMatcher>(false, try_cuda, match_conf);
    else if (range_width==-1)
        matcher = makePtr<BestOf2NearestMatcher>(try_cuda, match_conf);
    else
        matcher = makePtr<BestOf2NearestRangeMatcher>(range_width, try_cuda, match_conf);

    (*matcher)(features, pairwise_matches);
    matcher->collectGarbage();

    LOGLN("Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    // Check if we should save matches graph
    if (save_graph)
    {
        LOGLN("Saving matches graph...");
        ofstream f(save_graph_to.c_str());
        f << matchesGraphAsString(img_names, pairwise_matches, conf_thresh);
    }

    // Leave only images we are sure are from the same panorama
    vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
    vector<Mat> img_subset;
    vector<String> img_names_subset;
    vector<Size> full_img_sizes_subset;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        img_names_subset.push_back(img_names[indices[i]]);
        img_subset.push_back(images[indices[i]]);
        full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
    }

    images = img_subset;
    img_names = img_names_subset;
    full_img_sizes = full_img_sizes_subset;

    // Check if we still have enough images
    num_images = static_cast<int>(img_names.size());
    if (num_images < 2)
    {
        LOGLN("Need more images");
        return -1;
    }

    Ptr<Estimator> estimator;
    if (estimator_type == "affine")
        estimator = makePtr<AffineBasedEstimator>();
    else
        estimator = makePtr<HomographyBasedEstimator>();

    vector<CameraParams> cameras;
    if (!(*estimator)(features, pairwise_matches, cameras))
    {
        cout << "Homography estimation failed.\n";
        return -1;
    }

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
        LOGLN("Initial camera intrinsics #" << indices[i]+1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
    }

    Ptr<detail::BundleAdjusterBase> adjuster;
    if (ba_cost_func == "reproj") adjuster = makePtr<detail::BundleAdjusterReproj>();
    else if (ba_cost_func == "ray") adjuster = makePtr<detail::BundleAdjusterRay>();
    else if (ba_cost_func == "affine") adjuster = makePtr<detail::BundleAdjusterAffinePartial>();
    else if (ba_cost_func == "no") adjuster = makePtr<NoBundleAdjuster>();
    else
    {
        cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
        return -1;
    }
    adjuster->setConfThresh(conf_thresh);
    Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
    if (ba_refine_mask[0] == 'x') refine_mask(0,0) = 1;
    if (ba_refine_mask[1] == 'x') refine_mask(0,1) = 1;
    if (ba_refine_mask[2] == 'x') refine_mask(0,2) = 1;
    if (ba_refine_mask[3] == 'x') refine_mask(1,1) = 1;
    if (ba_refine_mask[4] == 'x') refine_mask(1,2) = 1;
    adjuster->setRefinementMask(refine_mask);
    if (!(*adjuster)(features, pairwise_matches, cameras))
    {
        cout << "Camera parameters adjusting failed.\n";
        return -1;
    }

    // Find median focal length

    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        LOGLN("Camera #" << indices[i]+1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
        focals.push_back(cameras[i].focal);
    }

    sort(focals.begin(), focals.end());
    float warped_image_scale;
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

    if (do_wave_correct)
    {
        vector<Mat> rmats;
        for (size_t i = 0; i < cameras.size(); ++i)
            rmats.push_back(cameras[i].R.clone());
        waveCorrect(rmats, wave_correct);
        for (size_t i = 0; i < cameras.size(); ++i)
            cameras[i].R = rmats[i];
    }

    LOGLN("Warping images (auxiliary)... ");
#if ENABLE_LOG
    t = getTickCount();
#endif

    vector<Point> corners(num_images);
    vector<UMat> masks_warped(num_images);
    vector<UMat> images_warped(num_images);
    vector<Size> sizes(num_images);
    vector<UMat> masks(num_images);

    // Prepare images masks
    for (int i = 0; i < num_images; ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    // Warp images and their masks

    Ptr<WarperCreator> warper_creator;
#ifdef HAVE_OPENCV_CUDAWARPING
    if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
    {
        if (warp_type == "plane")
            warper_creator = makePtr<cv::PlaneWarperGpu>();
        else if (warp_type == "cylindrical")
            warper_creator = makePtr<cv::CylindricalWarperGpu>();
        else if (warp_type == "spherical")
            warper_creator = makePtr<cv::SphericalWarperGpu>();
    }
    else
#endif
    {
        if (warp_type == "plane")
            warper_creator = makePtr<cv::PlaneWarper>();
        else if (warp_type == "affine")
            warper_creator = makePtr<cv::AffineWarper>();
        else if (warp_type == "cylindrical")
            warper_creator = makePtr<cv::CylindricalWarper>();
        else if (warp_type == "spherical")
            warper_creator = makePtr<cv::SphericalWarper>();
        else if (warp_type == "fisheye")
            warper_creator = makePtr<cv::FisheyeWarper>();
        else if (warp_type == "stereographic")
            warper_creator = makePtr<cv::StereographicWarper>();
        else if (warp_type == "compressedPlaneA2B1")
            warper_creator = makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
        else if (warp_type == "compressedPlaneA1.5B1")
            warper_creator = makePtr<cv::CompressedRectilinearWarper>(1.5f, 1.0f);
        else if (warp_type == "compressedPlanePortraitA2B1")
            warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
        else if (warp_type == "compressedPlanePortraitA1.5B1")
            warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
        else if (warp_type == "paniniA2B1")
            warper_creator = makePtr<cv::PaniniWarper>(2.0f, 1.0f);
        else if (warp_type == "paniniA1.5B1")
            warper_creator = makePtr<cv::PaniniWarper>(1.5f, 1.0f);
        else if (warp_type == "paniniPortraitA2B1")
            warper_creator = makePtr<cv::PaniniPortraitWarper>(2.0f, 1.0f);
        else if (warp_type == "paniniPortraitA1.5B1")
            warper_creator = makePtr<cv::PaniniPortraitWarper>(1.5f, 1.0f);
        else if (warp_type == "mercator")
            warper_creator = makePtr<cv::MercatorWarper>();
        else if (warp_type == "transverseMercator")
            warper_creator = makePtr<cv::TransverseMercatorWarper>();
    }

    if (!warper_creator)
    {
        cout << "Can't create the following warper '" << warp_type << "'\n";
        return 1;
    }

    Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

    for (int i = 0; i < num_images; ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect;
        K(0,0) *= swa; K(0,2) *= swa;
        K(1,1) *= swa; K(1,2) *= swa;

        corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();

        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }

    vector<UMat> images_warped_f(num_images);
    for (int i = 0; i < num_images; ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);

    LOGLN("Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    LOGLN("Compensating exposure...");
#if ENABLE_LOG
    t = getTickCount();
#endif

    Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
    if (dynamic_cast<GainCompensator*>(compensator.get()))
    {
        GainCompensator* gcompensator = dynamic_cast<GainCompensator*>(compensator.get());
        gcompensator->setNrFeeds(expos_comp_nr_feeds);
    }

    if (dynamic_cast<ChannelsCompensator*>(compensator.get()))
    {
        ChannelsCompensator* ccompensator = dynamic_cast<ChannelsCompensator*>(compensator.get());
        ccompensator->setNrFeeds(expos_comp_nr_feeds);
    }

    if (dynamic_cast<BlocksCompensator*>(compensator.get()))
    {
        BlocksCompensator* bcompensator = dynamic_cast<BlocksCompensator*>(compensator.get());
        bcompensator->setNrFeeds(expos_comp_nr_feeds);
        bcompensator->setNrGainsFilteringIterations(expos_comp_nr_filtering);
        bcompensator->setBlockSize(expos_comp_block_size, expos_comp_block_size);
    }

    compensator->feed(corners, images_warped, masks_warped);

    LOGLN("Compensating exposure, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    LOGLN("Finding seams...");
#if ENABLE_LOG
    t = getTickCount();
#endif

    Ptr<SeamFinder> seam_finder;
    if (seam_find_type == "no")
        seam_finder = makePtr<detail::NoSeamFinder>();
    else if (seam_find_type == "voronoi")
        seam_finder = makePtr<detail::VoronoiSeamFinder>();
    else if (seam_find_type == "gc_color")
    {
#ifdef HAVE_OPENCV_CUDALEGACY
        if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
            seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR);
        else
#endif
            seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
    }
    else if (seam_find_type == "gc_colorgrad")
    {
#ifdef HAVE_OPENCV_CUDALEGACY
        if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
            seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
        else
#endif
            seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
    }
    else if (seam_find_type == "dp_color")
        seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR);
    else if (seam_find_type == "dp_colorgrad")
        seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR_GRAD);
    if (!seam_finder)
    {
        cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
        return 1;
    }

    seam_finder->find(images_warped_f, corners, masks_warped);

    LOGLN("Finding seams, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    // Release unused memory
    images.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();

    LOGLN("Compositing...");
#if ENABLE_LOG
    t = getTickCount();
#endif

    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;
    Ptr<Blender> blender;
    Ptr<Timelapser> timelapser;
    //double compose_seam_aspect = 1;
    double compose_work_aspect = 1;

    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        LOGLN("Compositing image #" << indices[img_idx]+1);

        // Read image and resize it if necessary
        full_img = imread(samples::findFile(img_names[img_idx]));
        if (!is_compose_scale_set)
        {
            if (compose_megapix > 0)
                compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;

            // Compute relative scales
            //compose_seam_aspect = compose_scale / seam_scale;
            compose_work_aspect = compose_scale / work_scale;

            // Update warped image scale
            warped_image_scale *= static_cast<float>(compose_work_aspect);
            warper = warper_creator->create(warped_image_scale);

            // Update corners and sizes
            for (int i = 0; i < num_images; ++i)
            {
                // Update intrinsics
                cameras[i].focal *= compose_work_aspect;
                cameras[i].ppx *= compose_work_aspect;
                cameras[i].ppy *= compose_work_aspect;

                // Update corner and size
                Size sz = full_img_sizes[i];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                    sz.height = cvRound(full_img_sizes[i].height * compose_scale);
                }

                Mat K;
                cameras[i].K().convertTo(K, CV_32F);
                Rect roi = warper->warpRoi(sz, K, cameras[i].R);
                corners[i] = roi.tl();
                sizes[i] = roi.size();
            }
        }
        if (abs(compose_scale - 1) > 1e-1)
            resize(full_img, img, Size(), compose_scale, compose_scale, INTER_LINEAR_EXACT);
        else
            img = full_img;
        full_img.release();
        Size img_size = img.size();

        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);

        // Warp the current image
        warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

        // Warp the current image mask
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

        // Compensate exposure
        compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();

        dilate(masks_warped[img_idx], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);
        mask_warped = seam_mask & mask_warped;

        if (!blender && !timelapse)
        {
            blender = Blender::createDefault(blend_type, try_cuda);
            Size dst_sz = resultRoi(corners, sizes).size();
            float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
            if (blend_width < 1.f)
                blender = Blender::createDefault(Blender::NO, try_cuda);
            else if (blend_type == Blender::MULTI_BAND)
            {
                MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
                mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
                LOGLN("Multi-band blender, number of bands: " << mb->numBands());
            }
            else if (blend_type == Blender::FEATHER)
            {
                FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
                fb->setSharpness(1.f/blend_width);
                LOGLN("Feather blender, sharpness: " << fb->sharpness());
            }
            blender->prepare(corners, sizes);
        }
        else if (!timelapser && timelapse)
        {
            timelapser = Timelapser::createDefault(timelapse_type);
            timelapser->initialize(corners, sizes);
        }

        // Blend the current image
        if (timelapse)
        {
            timelapser->process(img_warped_s, Mat::ones(img_warped_s.size(), CV_8UC1), corners[img_idx]);
            String fixedFileName;
            size_t pos_s = String(img_names[img_idx]).find_last_of("/\\");
            if (pos_s == String::npos)
            {
                fixedFileName = "fixed_" + img_names[img_idx];
            }
            else
            {
                fixedFileName = "fixed_" + String(img_names[img_idx]).substr(pos_s + 1, String(img_names[img_idx]).length() - pos_s);
            }
            imwrite(fixedFileName, timelapser->getDst());
        }
        else
        {
            blender->feed(img_warped_s, mask_warped, corners[img_idx]);
        }
    }

    if (!timelapse)
    {
        Mat result, result_mask;
        blender->blend(result, result_mask);

        LOGLN("Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

        imwrite(result_name, result);
    }

    LOGLN("Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");
    return 0;
}
