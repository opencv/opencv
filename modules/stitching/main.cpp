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

// We follow to methods described in these two papers:
// 1) Construction of panoramic mosaics with global and local alignment. 
//    Heung-Yeung Shum and Richard Szeliski. 2000.
// 2) Eliminating Ghosting and Exposure Artifacts in Image Mosaics. 
//    Matthew Uyttendaele, Ashley Eden and Richard Szeliski. 2001.
// 3) Automatic Panoramic Image Stitching using Invariant Features. 
//    Matthew Brown and David G. Lowe. 2007.

#include "precomp.hpp"
#include "util.hpp"
#include "warpers.hpp"
#include "blenders.hpp"
#include "seam_finders.hpp"
#include "motion_estimators.hpp"
#include "exposure_compensate.hpp"

using namespace std;
using namespace cv;

void printUsage()
{
    cout << 
        "Rotation model images stitcher.\n\n"
        "opencv_stitching img1 img2 [...imgN] [flags]\n\n" 
        "Flags:\n"
        "  --preview\n"
        "      Run stitching in the preview mode. Works faster than usual mode,\n"
        "      but output image will have lower resolution.\n"
        "  --try_gpu (yes|no)\n"
        "      Try to use GPU. The default value is 'no'. All default values\n"
        "      are for CPU mode.\n"
        "\nMotion Estimation Flags:\n"
        "  --work_megapix <float>\n"
        "      Resolution for image registration step. The default is 0.6 Mpx.\n"
        "  --match_conf <float>\n"
        "      Confidence for feature matching step. The default is 0.65.\n"
        "  --conf_thresh <float>\n"
        "      Threshold for two images are from the same panorama confidence.\n"
        "      The default is 1.0.\n"
        "  --ba (ray|focal_ray)\n"
        "      Bundle adjustment cost function. The default is 'focal_ray'.\n"
        "  --wave_correct (no|yes)\n"
        "      Perform wave effect correction. The default is 'yes'.\n"
        "\nCompositing Flags:\n"
        "  --warp (plane|cylindrical|spherical)\n" 
        "      Warp surface type. The default is 'spherical'.\n"
        "  --seam_megapix <float>\n"
        "      Resolution for seam estimation step. The default is 0.1 Mpx.\n"
        "  --seam (no|voronoi|gc_color|gc_colorgrad)\n" 
        "      Seam estimation method. The default is 'gc_color'.\n"
        "  --compose_megapix <float>\n"
        "      Resolution for compositing step. Use -1 for original resolution.\n"
        "      The default is -1.\n"
        "  --expos_comp (no|gain|gain_blocks)\n"
        "      Exposure compensation method. The default is 'gain_blocks'.\n"
        "  --blend (no|feather|multiband)\n"
        "      Blending method. The default is 'multiband'.\n"
        "  --blend_strength <float>\n"
        "      Blending strength from [0,100] range. The default is 5.\n"
        "  --output <result_img>\n"
        "      The default is 'result.png'.\n";
}


// Default command line args
vector<string> img_names;
bool preview = false;
bool try_gpu = false;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = -1;
int ba_space = BundleAdjuster::FOCAL_RAY_SPACE;
float conf_thresh = 1.f;
bool wave_correct = true;
int warp_type = Warper::SPHERICAL;
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
float match_conf = 0.65f;
int seam_find_type = SeamFinder::GC_COLOR;
int blend_type = Blender::MULTI_BAND;
float blend_strength = 5;
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
        if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
        {
            printUsage();
            return -1;
        }
        else if (string(argv[i]) == "--preview")
        {
            preview = true;
        }
        else if (string(argv[i]) == "--try_gpu")
        {
            if (string(argv[i + 1]) == "no")
                try_gpu = false;
            else if (string(argv[i + 1]) == "yes")
                try_gpu = true;
            else
            {
                cout << "Bad --try_gpu flag value\n";
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
        else if (string(argv[i]) == "--wave_correct")
        {
            if (string(argv[i + 1]) == "no")
                wave_correct = false;
            else if (string(argv[i + 1]) == "yes")
                wave_correct = true;
            else
            {
                cout << "Bad --wave_correct flag value\n";
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
        else if (string(argv[i]) == "--expos_comp")
        {
            if (string(argv[i + 1]) == "no")
                expos_comp_type = ExposureCompensator::NO;
            else if (string(argv[i + 1]) == "gain")
                expos_comp_type = ExposureCompensator::GAIN;
            else if (string(argv[i + 1]) == "gain_blocks")
                expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
            else
            {
                cout << "Bad exposure compensation method\n";
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
            else if (string(argv[i + 1]) == "gc_color")
                seam_find_type = SeamFinder::GC_COLOR;
            else if (string(argv[i + 1]) == "gc_colorgrad")
                seam_find_type = SeamFinder::GC_COLOR_GRAD;
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
    if (preview)
    {
        compose_megapix = 0.6;
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
    SurfFeaturesFinder finder(try_gpu);
    Mat full_img, img;

    vector<Mat> images(num_images);
    vector<Size> full_img_sizes(num_images);
    double seam_work_aspect = 1;

    for (int i = 0; i < num_images; ++i)
    {
        full_img = imread(img_names[i]);
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
            resize(full_img, img, Size(), work_scale, work_scale);
        }
        if (!is_seam_scale_set)
        {
            seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));                    
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }

        finder(img, features[i]);
        features[i].img_idx = i;
        LOGLN("Features in image #" << i+1 << ": " << features[i].keypoints.size());

        resize(full_img, img, Size(), seam_scale, seam_scale);
        images[i] = img.clone();
    }

    full_img.release();
    img.release();

    LOGLN("Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    LOG("Pairwise matching");
    t = getTickCount();
    vector<MatchesInfo> pairwise_matches;
    BestOf2NearestMatcher matcher(try_gpu, match_conf);
    matcher(features, pairwise_matches);
    LOGLN("Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    // Leave only images we are sure are from the same panorama
    vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
    vector<Mat> img_subset;
    vector<string> img_names_subset;
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
        LOGLN("Initial focal length #" << indices[i]+1 << ": " << cameras[i].focal);
    }

    LOG("Bundle adjustment");
    t = getTickCount();
    BundleAdjuster adjuster(ba_space, conf_thresh);
    adjuster(features, pairwise_matches, cameras);
    LOGLN("Bundle adjustment, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    // Find median focal length
    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        LOGLN("Camera #" << indices[i]+1 << " focal length: " << cameras[i].focal);
        focals.push_back(cameras[i].focal);
    }
    nth_element(focals.begin(), focals.begin() + focals.size()/2, focals.end());
    float warped_image_scale = static_cast<float>(focals[focals.size() / 2]);

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
    Ptr<Warper> warper = Warper::createByCameraFocal(static_cast<float>(warped_image_scale * seam_work_aspect), 
                                                     warp_type);
    for (int i = 0; i < num_images; ++i)
    {
        corners[i] = warper->warp(images[i], static_cast<float>(cameras[i].focal * seam_work_aspect), 
                                  cameras[i].R, images_warped[i]);
        sizes[i] = images_warped[i].size();
        warper->warp(masks[i], static_cast<float>(cameras[i].focal * seam_work_aspect), 
                     cameras[i].R, masks_warped[i], INTER_NEAREST, BORDER_CONSTANT);
    }

    vector<Mat> images_warped_f(num_images);
    for (int i = 0; i < num_images; ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);

    LOGLN("Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    LOGLN("Exposure compensation (feed)...");
    t = getTickCount();
    Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
    compensator->feed(corners, images_warped, masks_warped);
    LOGLN("Exposure compensation (feed), time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    LOGLN("Finding seams...");
    t = getTickCount();
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

    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;
    Ptr<Blender> blender;
    double compose_seam_aspect = 1;
    double compose_work_aspect = 1;

    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        LOGLN("Compositing image #" << indices[img_idx]+1);

        // Read image and resize it if necessary
        full_img = imread(img_names[img_idx]);
        if (!is_compose_scale_set)
        {
            if (compose_megapix > 0)
                compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;

            // Compute relative scales
            compose_seam_aspect = compose_scale / seam_scale;
            compose_work_aspect = compose_scale / work_scale;

            // Update warped image scale
            warped_image_scale *= static_cast<float>(compose_work_aspect);
            warper = Warper::createByCameraFocal(warped_image_scale, warp_type);

            // Update corners and sizes
            for (int i = 0; i < num_images; ++i)
            {
                // Update camera focal
                cameras[i].focal *= compose_work_aspect;

                // Update corner and size
                Size sz = full_img_sizes[i];
                if (abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                    sz.height = cvRound(full_img_sizes[i].height * compose_scale);
                }
                Rect roi = warper->warpRoi(sz, static_cast<float>(cameras[i].focal), cameras[i].R);
                corners[i] = roi.tl();
                sizes[i] = roi.size();
            }
        }
        if (abs(compose_scale - 1) > 1e-1)
            resize(full_img, img, Size(), compose_scale, compose_scale);
        else
            img = full_img;
        full_img.release();
        Size img_size = img.size();

        // Warp the current image
        warper->warp(img, static_cast<float>(cameras[img_idx].focal), cameras[img_idx].R,
                     img_warped);

        // Warp the current image mask
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));    
        warper->warp(mask, static_cast<float>(cameras[img_idx].focal), cameras[img_idx].R, mask_warped,
                     INTER_NEAREST, BORDER_CONSTANT);

        // Compensate exposure
        compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();

        dilate(masks_warped[img_idx], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size());
        mask_warped = seam_mask & mask_warped;

        if (static_cast<Blender*>(blender) == 0)
        {
            blender = Blender::createDefault(blend_type);
            Size dst_sz = resultRoi(corners, sizes).size();
            float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
            if (blend_width < 1.f)
                blender = Blender::createDefault(Blender::NO);
            else if (blend_type == Blender::MULTI_BAND)
            {
                MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
                mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
                LOGLN("Multi-band blender, number of bands: " << mb->numBands());
            }
            else if (blend_type == Blender::FEATHER)
            {
                FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
                fb->setSharpness(1.f/blend_width);
                LOGLN("Feather blender, number of bands: " << fb->sharpness());
            }
            blender->prepare(corners, sizes);
        }

        // Blend the current image
        blender->feed(img_warped_s, mask_warped, corners[img_idx]);
    }
   
    Mat result, result_mask;
    blender->blend(result, result_mask);    

    LOGLN("Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    imwrite(result_name, result);

    LOGLN("Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");
    return 0;
}


