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
        << "\t[--trygpu (yes|no)]\n"
        << "\t[--work_megapix <float>]\n"
        << "\t[--compose_megapix <float>]\n"
        << "\t[--matchconf <float>]\n"
        << "\t[--ba (ray|focal_ray)]\n"
        << "\t[--conf_thresh <float>]\n"
        << "\t[--wavecorrect (no|yes)]\n"
        << "\t[--warp (plane|cylindrical|spherical)]\n" 
        << "\t[--seam (no|voronoi|graphcut)]\n" 
        << "\t[--blend (no|feather|multiband)]\n"
        << "\t[--output <result_img>]\n\n";
    cout << "--matchconf\n"
        << "\tGood values are in [0.2, 0.8] range usually.\n\n";
    cout << "HINT:\n"  
        << "\tDefault parameters are for '--trygpu no' configuration.\n"
        << "\tTry bigger values for --work_megapix if something is wrong.\n\n";
}

int main(int argc, char* argv[])
{
    int64 app_start_time = getTickCount();
    cv::setBreakOnError(true);

    vector<string> img_names;

    // Default parameters
    bool trygpu = false;
    double work_megapix = 0.2;
    double compose_megapix = 1;
    int ba_space = BundleAdjuster::FOCAL_RAY_SPACE;
    float conf_thresh = 1.f;
    bool wave_correct = true;
    int warp_type = Warper::SPHERICAL;
    bool user_match_conf = false;
    float match_conf = 0.6f;
    int seam_find_type = SeamFinder::VORONOI;
    int blend_type = Blender::MULTI_BAND;
    string result_name = "result.png";

    double work_scale = -1, compose_scale = -1;
    bool is_work_scale_set = false, is_compose_scale_set = false;

    if (argc == 1)
    {
        printUsage();
        return 0;
    }

    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--work_megapix")
        {
            work_megapix = atof(argv[i + 1]);
            break;
        }
    }

    int64 t = getTickCount();
    LOGLN("Parsing params and reading images...");
    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--trygpu")
        {
            if (string(argv[i + 1]) == "no")
                trygpu = false;
            else if (string(argv[i + 1]) == "yes")
                trygpu = true;
            else
            {
                cout << "Bad --trygpu flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--work_megapix") 
            i++; 
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
        else if (string(argv[i]) == "--conf_thresh")
        {
            conf_thresh = static_cast<float>(atof(argv[i + 1]));
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
                cout << "Bad --wavecorrect flag value\n";
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
            img_names.push_back(argv[i]);
    }
    LOGLN("Parsing params and reading images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    int num_images = static_cast<int>(img_names.size());
    if (num_images < 2)
    {
        cout << "Need more images\n";
        return -1;
    }

    t = getTickCount();
    LOGLN("Finding features...");
    vector<ImageFeatures> features(num_images);
    SurfFeaturesFinder finder(trygpu);
    Mat full_img, img;
    for (int i = 0; i < num_images; ++i)
    {
        full_img = imread(img_names[i]);
        if (full_img.empty())
        {
            cout << "Can't open image " << img_names[i] << endl;
            return -1;
        }
        if (work_megapix < 0)
            img = full_img;
        else
        {
            if (!is_work_scale_set)
            {
                work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));                    
                is_work_scale_set = true;
            }
            resize(full_img, img, Size(), work_scale, work_scale);
        }
        finder(img, features[i]);
    }
    LOGLN("Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    t = getTickCount();
    LOGLN("Pairwise matching... ");
    vector<MatchesInfo> pairwise_matches;
    BestOf2NearestMatcher matcher(trygpu);
    if (user_match_conf)
        matcher = BestOf2NearestMatcher(trygpu, match_conf);
    matcher(features, pairwise_matches);
    LOGLN("Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
    vector<string> img_names_subset;
    for (size_t i = 0; i < indices.size(); ++i)
        img_names_subset.push_back(img_names[indices[i]]);
    img_names = img_names_subset;

    num_images = static_cast<int>(img_names.size());
    if (num_images < 2)
    {
        cout << "Need more images\n";
        return -1;
    }

    t = getTickCount();
    LOGLN("Estimating rotations...");
    HomographyBasedEstimator estimator;
    vector<CameraParams> cameras;
    estimator(features, pairwise_matches, cameras);
    LOGLN("Estimating rotations, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
        LOGLN("Initial focal length " << i << ": " << cameras[i].focal);
    }

    t = getTickCount();
    LOGLN("Bundle adjustment... ");
    BundleAdjuster adjuster(ba_space, conf_thresh);
    adjuster(features, pairwise_matches, cameras);
    LOGLN("Bundle adjustment, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    if (wave_correct)
    {
        t = getTickCount();
        LOGLN("Wave correcting...");
        vector<Mat> rmats;
        for (size_t i = 0; i < cameras.size(); ++i)
            rmats.push_back(cameras[i].R);
        waveCorrect(rmats);
        for (size_t i = 0; i < cameras.size(); ++i)
            cameras[i].R = rmats[i];
        LOGLN("Wave correcting, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
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

    t = getTickCount();
    vector<Mat> images(num_images);
    LOGLN("Compose scaling...");
    for (int i = 0; i < num_images; ++i)
    {
        Mat full_img = imread(img_names[i]);
        if (!is_compose_scale_set)
        {
            compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));                    
            is_compose_scale_set = true;
        }
        Mat img;
        resize(full_img, img, Size(), compose_scale, compose_scale);
        images[i] = img;
        cameras[i].focal *= compose_scale / work_scale;
    }
    camera_focal *= static_cast<float>(compose_scale / work_scale);
    LOGLN("Compose scaling, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    vector<Mat> masks(num_images);
    for (int i = 0; i < num_images; ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    vector<Point> corners(num_images);
    vector<Mat> masks_warped(num_images);
    vector<Mat> images_warped(num_images);

    t = getTickCount();
    LOGLN("Warping images... ");
    Ptr<Warper> warper = Warper::createByCameraFocal(camera_focal, warp_type);
    for (int i = 0; i < num_images; ++i)
    {
        corners[i] = (*warper)(images[i], static_cast<float>(cameras[i].focal), cameras[i].R, images_warped[i]);
        (*warper)(masks[i], static_cast<float>(cameras[i].focal), cameras[i].R, masks_warped[i], INTER_NEAREST, BORDER_CONSTANT);
    }
    vector<Mat> images_f(num_images);
    for (int i = 0; i < num_images; ++i)
        images_warped[i].convertTo(images_f[i], CV_32F);
    LOGLN("Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    t = getTickCount();
    LOGLN("Finding seams...");
    Ptr<SeamFinder> seam_finder = SeamFinder::createDefault(seam_find_type);
    (*seam_finder)(images_f, corners, masks_warped);
    LOGLN("Finding seams, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    t = getTickCount();
    LOGLN("Blending images...");
    Ptr<Blender> blender = Blender::createDefault(blend_type);
    if (blend_type == Blender::MULTI_BAND)
        // Ensure last pyramid layer area is about 1 pix 
        dynamic_cast<MultiBandBlender*>((Blender*)(blender))
            ->setNumBands(static_cast<int>(ceil(log(static_cast<double>(images_f[0].size().area())) 
                                                / log(4.0))));
    Mat result, result_mask;
    (*blender)(images_f, corners, masks_warped, result, result_mask);
    LOGLN("Blending images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    imwrite(result_name, result);

    LOGLN("Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");
    return 0;
}

