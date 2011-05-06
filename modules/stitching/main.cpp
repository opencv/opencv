#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "util.hpp"
#include "warpers.hpp"
#include "blenders.hpp"
#include "seam_finders.hpp"
#include "motion_estimators.hpp"

using namespace std;
using namespace cv;

void printUsage()
{
    cout << "Rotation model images stitcher.\n\n";
    cout << "Usage: opencv_stitching img1 img2 [...imgN]\n" 
        << "\t[--matchconf <float>]\n"
        << "\t[--ba (ray|focal_ray)]\n"
        << "\t[--ba_thresh <float>]\n"
        << "\t[--wavecorrect (no|yes)]\n"
        << "\t[--warp (plane|cylindrical|spherical)]\n" 
        << "\t[--seam (no|voronoi|graphcut)]\n" 
        << "\t[--blend (no|feather|multiband)]\n"
        << "\t[--output <result_img>]\n\n";
    cout << "--matchconf\n"
        << "\tGood values are in [0.2, 0.8] range usually.\n\n";
    cout << "--ba_thresh\n"
        << "\tGood values are in [0.3, 1.0] range usually.\n";
}

int main(int argc, char* argv[])
{
    cv::setBreakOnError(true);

    vector<Mat> images;
    string result_name = "result.png";
    int ba_space = BundleAdjuster::FOCAL_RAY_SPACE;
    float ba_thresh = 1.f;
    bool wave_correct = false;
    int warp_type = Warper::SPHERICAL;
    bool user_match_conf = false;
    float match_conf = 0.55f;
    int seam_find_type = SeamFinder::GRAPH_CUT;
    int blend_type = Blender::MULTI_BAND;

    if (argc == 1)
    {
        printUsage();
        return 0;
    }

    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--result")
        {
            result_name = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--matchconf")
        {
            user_match_conf = true;
            match_conf = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--ba")
        {
            if (string(argv[i + 1]) == "ray")
                ba_space = BundleAdjuster::RAY_SPACE;
            else if (string(argv[i + 1]) == "focal_ray")
                ba_space = BundleAdjuster::FOCAL_RAY_SPACE;
            else
            {
                cout << "Bad bundle adjustment space\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--ba_thresh")
        {
            ba_thresh = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--wavecorrect")
        {
            if (string(argv[i + 1]) == "no")
                wave_correct = false;
            else if (string(argv[i + 1]) == "yes")
                wave_correct = true;
            else
            {
                cout << "Bad wave correct flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--warp")
        {
            if (string(argv[i + 1]) == "plane")
                warp_type = Warper::PLANE;
            else if (string(argv[i + 1]) == "cylindrical")
                warp_type = Warper::CYLINDRICAL;
            else if (string(argv[i + 1]) == "spherical")
                warp_type = Warper::SPHERICAL;
            else
            {
                cout << "Bad warping method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--seam")
        {
            if (string(argv[i + 1]) == "no")
                seam_find_type = SeamFinder::NO;
            else if (string(argv[i + 1]) == "voronoi")
                seam_find_type = SeamFinder::VORONOI;
            else if (string(argv[i + 1]) == "graphcut")
                seam_find_type = SeamFinder::GRAPH_CUT;
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
        else if (string(argv[i]) == "--output")
        {
            result_name = argv[i + 1];
            i++;
        }
        else
        {
            Mat img = imread(argv[i]);
            if (img.empty())
            {
                cout << "Can't open image " << argv[i] << endl;
                return -1;
            }
            images.push_back(img);
        }
    }

    const int num_images = static_cast<int>(images.size());
    if (num_images < 2)
    {
        cout << "Need more images\n";
        return -1;
    }

    LOGLN("Finding features...");
    vector<ImageFeatures> features;
    SurfFeaturesFinder finder;
    finder(images, features);

    LOGLN("Pairwise matching...");
    vector<MatchesInfo> pairwise_matches;
    BestOf2NearestMatcher matcher;
    if (user_match_conf)
        matcher = BestOf2NearestMatcher(true, match_conf);
    matcher(images, features, pairwise_matches);

    LOGLN("Estimating rotations...");
    HomographyBasedEstimator estimator;
    vector<CameraParams> cameras;
    estimator(images, features, pairwise_matches, cameras);

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].M.convertTo(R, CV_32F);
        cameras[i].M = R;
        LOGLN("Initial focal length " << i << ": " << cameras[i].focal);
    }

    LOGLN("Bundle adjustment...");
    BundleAdjuster adjuster(ba_space, ba_thresh);
    adjuster(images, features, pairwise_matches, cameras);

    if (wave_correct)
    {
        LOGLN("Wave correcting...");
        vector<Mat> rmats;
        for (size_t i = 0; i < cameras.size(); ++i)
            rmats.push_back(cameras[i].M);
        waveCorrect(rmats);
        for (size_t i = 0; i < cameras.size(); ++i)
            cameras[i].M = rmats[i];
    }

    // Find median focal length
    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        LOGLN("Camera focal length " << i << ": " << cameras[i].focal);
        focals.push_back(cameras[i].focal);
    }
    nth_element(focals.begin(), focals.end(), focals.begin() + focals.size() / 2);
    float camera_focal = static_cast<float>(focals[focals.size() / 2]);

    vector<Mat> masks(num_images);
    for (int i = 0; i < num_images; ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    vector<Point> corners(num_images);
    vector<Mat> masks_warped(num_images);
    vector<Mat> images_warped(num_images);

    LOGLN("Warping images...");
    Ptr<Warper> warper = Warper::createByCameraFocal(camera_focal, warp_type);
    for (int i = 0; i < num_images; ++i)
    {
        corners[i] = (*warper)(images[i], static_cast<float>(cameras[i].focal), cameras[i].M, images_warped[i]);
        (*warper)(masks[i], static_cast<float>(cameras[i].focal), cameras[i].M, masks_warped[i], INTER_NEAREST, BORDER_CONSTANT);
    }
    vector<Mat> images_f(num_images);
    for (int i = 0; i < num_images; ++i)
        images_warped[i].convertTo(images_f[i], CV_32F);

    LOGLN("Finding seams...");
    Ptr<SeamFinder> seam_finder = SeamFinder::createDefault(seam_find_type);
    (*seam_finder)(images_f, corners, masks_warped);

    LOGLN("Blending images...");
    Mat result, result_mask;
    Ptr<Blender> blender = Blender::createDefault(blend_type);
    (*blender)(images_f, corners, masks_warped, result, result_mask);

    imwrite(result_name, result);

    LOGLN("Finished");
    return 0;
}
