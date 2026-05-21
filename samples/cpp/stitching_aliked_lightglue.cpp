// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Image stitching sample using ALIKED feature extractor and LightGlue matcher.
// Replaces the traditional SIFT/ORB + FLANN pipeline with DNN-based features.

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
#include "opencv2/calib3d.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/features.hpp"

#define ENABLE_LOG 1
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

using namespace std;
using namespace cv;
using namespace cv::detail;

// ============================================================================
//  LightGlueFeaturesMatcher — adapts LightGlueMatcher (DescriptorMatcher)
//  to the stitching pipeline's FeaturesMatcher interface.
// ============================================================================
class LightGlueFeaturesMatcher : public FeaturesMatcher
{
public:
    LightGlueFeaturesMatcher(Ptr<LightGlueMatcher> lgMatcher,
                             int num_matches_thresh1 = 6,
                             int num_matches_thresh2 = 6,
                             double matches_confidence_thresh = 3.0)
        : FeaturesMatcher(false),  // not thread-safe: setPairInfo() mutates state
          lgMatcher_(lgMatcher),
          num_matches_thresh1_(num_matches_thresh1),
          num_matches_thresh2_(num_matches_thresh2),
          matches_confidence_thresh_(matches_confidence_thresh),
          lg_score_thresh_(0.0f)
    {}

    void setScoreThreshold(float thresh) { lg_score_thresh_ = thresh; }

    void match(const ImageFeatures &features1, const ImageFeatures &features2,
               MatchesInfo &matches_info) CV_OVERRIDE
    {
        matches_info.src_img_idx = features1.img_idx;
        matches_info.dst_img_idx = features2.img_idx;
        matches_info.confidence = 0;

        // Step 1: Guard empty keypoints
        if (features1.keypoints.empty() || features2.keypoints.empty())
            return;

        const int N = static_cast<int>(features1.keypoints.size());
        const int M = static_cast<int>(features2.keypoints.size());

        // Step 2: Build Nx2 keypoint coordinate matrices
        Mat kpts1Mat(N, 2, CV_32F);
        Mat kpts2Mat(M, 2, CV_32F);
        for (int i = 0; i < N; i++)
        {
            kpts1Mat.at<float>(i, 0) = features1.keypoints[i].pt.x;
            kpts1Mat.at<float>(i, 1) = features1.keypoints[i].pt.y;
        }
        for (int i = 0; i < M; i++)
        {
            kpts2Mat.at<float>(i, 0) = features2.keypoints[i].pt.x;
            kpts2Mat.at<float>(i, 1) = features2.keypoints[i].pt.y;
        }

        // Step 3: Set pair context for LightGlue spatial reasoning
        lgMatcher_->setPairInfo(kpts1Mat, kpts2Mat, features1.img_size, features2.img_size);

        // Step 4: Run LightGlue matching
        Mat desc1 = features1.descriptors.getMat(ACCESS_READ);
        Mat desc2 = features2.descriptors.getMat(ACCESS_READ);
        vector<DMatch> matches;
        lgMatcher_->match(desc1, desc2, matches);

        // Step 5: Filter by score threshold
        if (lg_score_thresh_ > 0.0f)
        {
            float maxDist = 1.0f - lg_score_thresh_;
            vector<DMatch> filtered;
            filtered.reserve(matches.size());
            for (const auto& m : matches)
            {
                if (m.distance <= maxDist)
                    filtered.push_back(m);
            }
            matches.swap(filtered);
        }

        // Store matches before homography estimation
        matches_info.matches = matches;

        // Guard: need at least 4 matches for findHomography
        if (matches.size() < 4)
            return;

        // Step 6: Estimate homography with centered coordinates
        Mat src_points(1, static_cast<int>(matches.size()), CV_32FC2);
        Mat dst_points(1, static_cast<int>(matches.size()), CV_32FC2);
        for (size_t i = 0; i < matches.size(); ++i)
        {
            const DMatch& m = matches[i];

            Point2f p = features1.keypoints[m.queryIdx].pt;
            p.x -= features1.img_size.width * 0.5f;
            p.y -= features1.img_size.height * 0.5f;
            src_points.at<Point2f>(0, static_cast<int>(i)) = p;

            p = features2.keypoints[m.trainIdx].pt;
            p.x -= features2.img_size.width * 0.5f;
            p.y -= features2.img_size.height * 0.5f;
            dst_points.at<Point2f>(0, static_cast<int>(i)) = p;
        }

        matches_info.H = findHomography(src_points, dst_points, matches_info.inliers_mask, RANSAC);
        if (matches_info.H.empty() ||
            std::abs(determinant(matches_info.H)) < std::numeric_limits<double>::epsilon())
            return;

        // Step 7: Compute inliers and confidence (Brown & Lowe formula)
        matches_info.num_inliers = 0;
        for (size_t i = 0; i < matches_info.inliers_mask.size(); ++i)
            if (matches_info.inliers_mask[i])
                matches_info.num_inliers++;

        matches_info.confidence = matches_info.num_inliers /
                                  (8 + 0.3 * matches_info.matches.size());

        // Zero confidence for too-close image pairs
        if (matches_info.confidence > matches_confidence_thresh_)
            matches_info.confidence = 0.;

        // Step 8: Refine homography on inliers if enough
        if (matches_info.num_inliers >= num_matches_thresh2_)
        {
            src_points.create(1, matches_info.num_inliers, CV_32FC2);
            dst_points.create(1, matches_info.num_inliers, CV_32FC2);
            int inlier_idx = 0;
            for (size_t i = 0; i < matches_info.matches.size(); ++i)
            {
                if (!matches_info.inliers_mask[i])
                    continue;

                const DMatch& m = matches_info.matches[i];

                Point2f p = features1.keypoints[m.queryIdx].pt;
                p.x -= features1.img_size.width * 0.5f;
                p.y -= features1.img_size.height * 0.5f;
                src_points.at<Point2f>(0, inlier_idx) = p;

                p = features2.keypoints[m.trainIdx].pt;
                p.x -= features2.img_size.width * 0.5f;
                p.y -= features2.img_size.height * 0.5f;
                dst_points.at<Point2f>(0, inlier_idx) = p;

                inlier_idx++;
            }

            matches_info.H = findHomography(src_points, dst_points, RANSAC);
        }

        LOGLN("  Pair " << features1.img_idx << "->" << features2.img_idx
              << ": matches=" << matches_info.matches.size()
              << ", inliers=" << matches_info.num_inliers
              << ", conf=" << matches_info.confidence);
    }

private:
    Ptr<LightGlueMatcher> lgMatcher_;
    int num_matches_thresh1_;
    int num_matches_thresh2_;
    double matches_confidence_thresh_;
    float lg_score_thresh_;
};


// ============================================================================
//  Command-line arguments
// ============================================================================
static void printUsage(char** argv)
{
    cout <<
        "Image stitcher using ALIKED + LightGlue (DNN-based features).\n\n"
         << argv[0] << " img1 img2 [...imgN] --aliked_model <path> --lightglue_model <path> [flags]\n\n"
        "Required:\n"
        "  --aliked_model <path>\n"
        "      Path to ALIKED ONNX model file.\n"
        "  --lightglue_model <path>\n"
        "      Path to LightGlue ONNX model file (for ALIKED descriptors).\n"
        "\nOptional:\n"
        "  --lg_score_thresh <float>\n"
        "      LightGlue confidence threshold. Matches with confidence below this\n"
        "      are discarded. The default is 0.0 (accept all).\n"
        "  --preview\n"
        "      Run stitching in the preview mode (lower resolution, faster).\n"
        "\nMotion Estimation Flags:\n"
        "  --work_megapix <float>\n"
        "      Resolution for image registration step. The default is 0.6 Mpx.\n"
        "  --conf_thresh <float>\n"
        "      Threshold for two images are from the same panorama confidence.\n"
        "      The default is 1.0.\n"
        "  --ba (no|reproj|ray|affine)\n"
        "      Bundle adjustment cost function. The default is ray.\n"
        "  --ba_refine_mask (mask)\n"
        "      Set refinement mask for bundle adjustment. The default is 'xxxxx'.\n"
        "  --wave_correct (no|horiz|vert)\n"
        "      Perform wave effect correction. The default is 'horiz'.\n"
        "  --save_graph <file_name>\n"
        "      Save matches graph to <file_name> file.\n"
        "\nCompositing Flags:\n"
        "  --warp (affine|plane|cylindrical|spherical|fisheye|stereographic|...)\n"
        "      Warp surface type. The default is 'spherical'.\n"
        "  --seam_megapix <float>\n"
        "      Resolution for seam estimation step. The default is 0.1 Mpx.\n"
        "  --seam (no|voronoi|gc_color|gc_colorgrad)\n"
        "      Seam estimation method. The default is 'gc_color'.\n"
        "  --compose_megapix <float>\n"
        "      Resolution for compositing step. Use -1 for original resolution.\n"
        "  --expos_comp (no|gain|gain_blocks|channels|channels_blocks)\n"
        "      Exposure compensation method. The default is 'gain_blocks'.\n"
        "  --blend (no|feather|multiband)\n"
        "      Blending method. The default is 'multiband'.\n"
        "  --blend_strength <float>\n"
        "      Blending strength from [0,100] range. The default is 5.\n"
        "  --output <result_img>\n"
        "      The default is 'result.jpg'.\n"
        "  --timelapse (as_is|crop)\n"
        "      Output warped images separately as frames of a time lapse movie.\n"
        "  --rangewidth <int>\n"
        "      Uses range_width to limit number of images to match with.\n";
}


// Default command line args
vector<String> img_names;
bool preview = false;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 1.f;
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
        else if (string(argv[i]) == "--lg_score_thresh")
        {
            lg_score_thresh = static_cast<float>(atof(argv[i + 1]));
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
        else
            img_names.push_back(argv[i]);
    }

    // Validate required model paths
    if (aliked_model_path.empty())
    {
        cout << "Error: --aliked_model is required.\n";
        printUsage(argv);
        return -1;
    }
    if (lightglue_model_path.empty())
    {
        cout << "Error: --lightglue_model is required.\n";
        printUsage(argv);
        return -1;
    }

    if (preview)
    {
        compose_megapix = 0.6;
    }
    return 0;
}


int main(int argc, char* argv[])
{
#if ENABLE_LOG
    int64 app_start_time = getTickCount();
#endif

    // Disable OpenCL to avoid UMat/DNN backend sync issues
    cv::ocl::setUseOpenCL(false);

    int retval = parseCmdArgs(argc, argv);
    if (retval)
        return retval;

    // Check if have enough images
    int num_images = static_cast<int>(img_names.size());
    if (num_images < 2)
    {
        LOGLN("Need more images");
        return -1;
    }

    double work_scale = 1, seam_scale = 1, compose_scale = 1;
    bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;

    // ================================================================
    //  Feature extraction with ALIKED
    // ================================================================
    LOGLN("Creating ALIKED feature extractor...");
    Ptr<ALIKED> aliked = ALIKED::create(aliked_model_path);

    LOGLN("Finding features...");
#if ENABLE_LOG
    int64 t = getTickCount();
#endif

    vector<ImageFeatures> features(num_images);
    vector<Mat> images(num_images);
    vector<Size> full_img_sizes(num_images);
    double seam_work_aspect = 1;

    Mat full_img, img;
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

        computeImageFeatures(aliked, img, features[i]);
        features[i].img_idx = i;
        LOGLN("Features in image #" << i+1 << ": " << features[i].keypoints.size());

        resize(full_img, img, Size(), seam_scale, seam_scale, INTER_LINEAR_EXACT);
        images[i] = img.clone();
    }

    full_img.release();
    img.release();

    LOGLN("Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    // ================================================================
    //  Pairwise matching with LightGlue
    // ================================================================
    LOGLN("Creating LightGlue matcher...");
    Ptr<LightGlueMatcher> lg = LightGlueMatcher::create(lightglue_model_path);
    Ptr<LightGlueFeaturesMatcher> lgMatcher = makePtr<LightGlueFeaturesMatcher>(lg);
    lgMatcher->setScoreThreshold(lg_score_thresh);

    LOG("Pairwise matching");
#if ENABLE_LOG
    t = getTickCount();
#endif

    vector<MatchesInfo> pairwise_matches;
    (*lgMatcher)(features, pairwise_matches);
    lgMatcher->collectGarbage();

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
        seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
    else if (seam_find_type == "gc_colorgrad")
        seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
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
            blender = Blender::createDefault(blend_type, false);
            Size dst_sz = resultRoi(corners, sizes).size();
            float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
            if (blend_width < 1.f)
                blender = Blender::createDefault(Blender::NO, false);
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
