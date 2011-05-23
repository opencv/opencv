/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include "precomp.hpp"
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
        << "\t[--seam_megapix <float>]\n"
        << "\t[--compose_megapix <float>]\n"
        << "\t[--match_conf <float>]\n"
        << "\t[--ba (ray|focal_ray)]\n"
        << "\t[--conf_thresh <float>]\n"
        << "\t[--wavecorrect (no|yes)]\n"
        << "\t[--warp (plane|cylindrical|spherical)]\n" 
        << "\t[--seam (no|voronoi|graphcut)]\n" 
        << "\t[--blend (no|feather|multiband)]\n"
        << "\t[--numbands <int>]\n"
        << "\t[--output <result_img>]\n\n";
    cout << "--match_conf\n"
        << "\tGood values are in [0.2, 0.8] range usually.\n\n";
    cout << "HINT:\n"  
        << "\tTry bigger values for --work_megapix if something is wrong.\n\n";
}


// Default command line args
vector<string> img_names;
bool trygpu = false;
double work_megapix = 0.3;
double seam_megapix = 0.1;
double compose_megapix = 1;
int ba_space = BundleAdjuster::FOCAL_RAY_SPACE;
float conf_thresh = 1.f;
bool wave_correct = true;
int warp_type = Warper::SPHERICAL;
bool user_match_conf = false;
float match_conf = 0.6f;
int seam_find_type = SeamFinder::GRAPH_CUT;
int blend_type = Blender::MULTI_BAND;
int numbands = 5;
string result_name = "result.png";

int parseCmdArgs(int argc, char** argv)
{
    if (argc == 1)
    {
        printUsage();
        return -1;
    }

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
        else if (string(argv[i]) == "--match_conf")
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
        else if (string(argv[i]) == "--numbands")
        {
            numbands = atoi(argv[i + 1]);
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
    return 0;
}


int main(int argc, char* argv[])
{
    int64 app_start_time = getTickCount();
    cv::setBreakOnError(true);

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

    LOGLN("Finding features...");
    int64 t = getTickCount();

    vector<ImageFeatures> features(num_images);
    SurfFeaturesFinder finder(trygpu);
    Mat full_img, img;

    vector<Mat> images(num_images);
    double seam_work_aspect = 1;

    for (int i = 0; i < num_images; ++i)
    {
        full_img = imread(img_names[i]);
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
            resize(full_img, img, Size(), work_scale, work_scale);
        }
        if (!is_seam_scale_set)
        {
            seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));                    
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }

        finder(img, features[i]);
        LOGLN("Features in image #" << i << ": " << features[i].keypoints.size());

        resize(full_img, img, Size(), seam_scale, seam_scale);
        images[i] = img.clone();
    }

    full_img.release();
    img.release();

    LOGLN("Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    LOGLN("Pairwise matching... ");
    t = getTickCount();
    vector<MatchesInfo> pairwise_matches;
    BestOf2NearestMatcher matcher(trygpu);
    if (user_match_conf)
        matcher = BestOf2NearestMatcher(trygpu, match_conf);
    matcher(features, pairwise_matches);
    LOGLN("Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    // Leave only images we are sure are from the same panorama
    vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
    vector<Mat> img_subset;
    vector<string> img_names_subset;
    for (size_t i = 0; i < indices.size(); ++i)
        img_names_subset.push_back(img_names[indices[i]]);
    img_names = img_names_subset;

    // Check if we still have enough images
    num_images = static_cast<int>(img_names.size());
    if (num_images < 2)
    {
        LOGLN("Need more images");
        return -1;
    }

    LOGLN("Estimating rotations...");
    t = getTickCount();
    HomographyBasedEstimator estimator;
    vector<CameraParams> cameras;
    estimator(features, pairwise_matches, cameras);
    LOGLN("Estimating rotations, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
        LOGLN("Initial focal length #" << i << ": " << cameras[i].focal);
    }

    LOGLN("Bundle adjustment... ");
    t = getTickCount();
    BundleAdjuster adjuster(ba_space, conf_thresh);
    adjuster(features, pairwise_matches, cameras);
    LOGLN("Bundle adjustment, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    if (wave_correct)
    {
        LOGLN("Wave correcting...");
        t = getTickCount();
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
        LOGLN("Camera #" << i << " focal length: " << cameras[i].focal);
        focals.push_back(cameras[i].focal);
    }
    nth_element(focals.begin(), focals.end(), focals.begin() + focals.size() / 2);
    float camera_focal = static_cast<float>(focals[focals.size() / 2]);

    LOGLN("Warping images (auxiliary)... ");
    t = getTickCount();

    vector<Point> corners(num_images);
    vector<Mat> masks_warped(num_images);
    vector<Mat> images_warped(num_images);
    vector<Size> sizes(num_images);
    vector<Mat> masks(num_images);

    // Preapre images masks
    for (int i = 0; i < num_images; ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    // Warp images and their masks
    Ptr<Warper> warper = Warper::createByCameraFocal(static_cast<float>(camera_focal * seam_work_aspect), 
                                                     warp_type);
    for (int i = 0; i < num_images; ++i)
    {
        corners[i] = warper->warp(images[i], static_cast<float>(cameras[i].focal * seam_work_aspect), 
                                  cameras[i].R, images_warped[i]);
        sizes[i] = images_warped[i].size();
        warper->warp(masks[i], static_cast<float>(cameras[i].focal * seam_work_aspect), 
                     cameras[i].R, masks_warped[i], INTER_NEAREST, BORDER_CONSTANT);
    }

    // Convert to float for blending
    vector<Mat> images_warped_f(num_images);
    for (int i = 0; i < num_images; ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);

    LOGLN("Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    LOGLN("Finding seams...");
    t = getTickCount();

    // Find seams
    Ptr<SeamFinder> seam_finder = SeamFinder::createDefault(seam_find_type);
    seam_finder->find(images_warped_f, corners, masks_warped);

    LOGLN("Finding seams, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    // Release unused memory
    images.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();

    LOGLN("Compositing...");
    t = getTickCount();

    Mat img_warped, img_warped_f;
    Mat dilated_mask, seam_mask, mask, mask_warped;
    Ptr<Blender> blender;
    double compose_seam_aspect = 1;
    double compose_work_aspect = 1;

    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        LOGLN("Compositing image #" << img_idx);

        // Read image and resize it if necessary
        full_img = imread(img_names[img_idx]);
        if (!is_compose_scale_set)
        {
            if (compose_megapix > 0)
                compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;
            compose_seam_aspect = compose_scale / seam_scale;
            compose_work_aspect = compose_scale / work_scale;
            camera_focal *= static_cast<float>(compose_work_aspect);
            warper = Warper::createByCameraFocal(camera_focal, warp_type);
        }
        if (abs(compose_scale - 1) > 1e-1)
            resize(full_img, img, Size(), compose_scale, compose_scale);
        else
            img = full_img;
        full_img.release();

        // Update cameras paramters
        cameras[img_idx].focal *= compose_work_aspect;

        // Warp the current image
        warper->warp(img, static_cast<float>(cameras[img_idx].focal), cameras[img_idx].R, 
                     img_warped);
        img_warped.convertTo(img_warped_f, CV_32F);
        img_warped.release();

        // Warp current image mask
        mask.create(img.size(), CV_8U);
        mask.setTo(Scalar::all(255));    
        warper->warp(mask, static_cast<float>(cameras[img_idx].focal), cameras[img_idx].R, mask_warped,
                     INTER_NEAREST, BORDER_CONSTANT);
        mask.release();
        dilate(masks_warped[img_idx], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size());
        mask_warped = seam_mask & mask_warped;

        if (static_cast<Blender*>(blender) == 0)
        {
            blender = Blender::createDefault(blend_type);

            if (blend_type == Blender::MULTI_BAND)
            {
                MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
                mb->setNumBands(numbands);
            }

            // Determine the final image size
            Rect dst_roi = resultRoi(corners, sizes);
            for (int i = 0; i < num_images; ++i)
            {
                corners[i] = dst_roi.tl() + (corners[i] - dst_roi.tl()) * compose_seam_aspect;
                sizes[i] = Size(static_cast<int>((sizes[i].width + 1) * compose_seam_aspect), 
                                static_cast<int>((sizes[i].height + 1) * compose_seam_aspect));
            }
            blender->prepare(corners, sizes);
        }

        // Blend the current image
        blender->feed(img_warped_f, mask_warped, corners[img_idx]);
    }
   
    Mat result, result_mask;
    blender->blend(result, result_mask);

    LOGLN("Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    imwrite(result_name, result);

    LOGLN("Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");
    return 0;
}


