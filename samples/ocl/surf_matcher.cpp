#include <iostream>
#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ocl/ocl.hpp"
#include "opencv2/nonfree/ocl.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace cv;
using namespace cv::ocl;

const int LOOP_NUM = 10;
const int GOOD_PTS_MAX = 50;
const float GOOD_PORTION = 0.15f;

namespace
{

int64 work_begin = 0;
int64 work_end = 0;

void workBegin()
{
    work_begin = getTickCount();
}
void workEnd()
{
    work_end = getTickCount() - work_begin;
}
double getTime()
{
    return work_end /((double)getTickFrequency() * 1000.);
}

template<class KPDetector>
struct SURFDetector
{
    KPDetector surf;
    SURFDetector(double hessian = 800.0)
        :surf(hessian)
    {
    }
    template<class T>
    void operator()(const T& in, const T& mask, std::vector<cv::KeyPoint>& pts, T& descriptors, bool useProvided = false)
    {
        surf(in, mask, pts, descriptors, useProvided);
    }
};

template<class KPMatcher>
struct SURFMatcher
{
    KPMatcher matcher;
    template<class T>
    void match(const T& in1, const T& in2, std::vector<cv::DMatch>& matches)
    {
        matcher.match(in1, in2, matches);
    }
};

Mat drawGoodMatches(
    const Mat& cpu_img1,
    const Mat& cpu_img2,
    const std::vector<KeyPoint>& keypoints1,
    const std::vector<KeyPoint>& keypoints2,
    std::vector<DMatch>& matches,
    std::vector<Point2f>& scene_corners_
    )
{
    //-- Sort matches and preserve top 10% matches
    std::sort(matches.begin(), matches.end());
    std::vector< DMatch > good_matches;
    double minDist = matches.front().distance,
           maxDist = matches.back().distance;

    const int ptsPairs = std::min(GOOD_PTS_MAX, (int)(matches.size() * GOOD_PORTION));
    for( int i = 0; i < ptsPairs; i++ )
    {
        good_matches.push_back( matches[i] );
    }
    std::cout << "\nMax distance: " << maxDist << std::endl;
    std::cout << "Min distance: " << minDist << std::endl;

    std::cout << "Calculating homography using " << ptsPairs << " point pairs." << std::endl;

    // drawing the results
    Mat img_matches;
    drawMatches( cpu_img1, keypoints1, cpu_img2, keypoints2,
                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS  );

    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
    }
    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point(0,0);
    obj_corners[1] = Point( cpu_img1.cols, 0 );
    obj_corners[2] = Point( cpu_img1.cols, cpu_img1.rows );
    obj_corners[3] = Point( 0, cpu_img1.rows );
    std::vector<Point2f> scene_corners(4);

    Mat H = findHomography( obj, scene, RANSAC );
    perspectiveTransform( obj_corners, scene_corners, H);

    scene_corners_ = scene_corners;

    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( img_matches,
          scene_corners[0] + Point2f( (float)cpu_img1.cols, 0), scene_corners[1] + Point2f( (float)cpu_img1.cols, 0),
          Scalar( 0, 255, 0), 2, LINE_AA );
    line( img_matches,
          scene_corners[1] + Point2f( (float)cpu_img1.cols, 0), scene_corners[2] + Point2f( (float)cpu_img1.cols, 0),
          Scalar( 0, 255, 0), 2, LINE_AA );
    line( img_matches,
          scene_corners[2] + Point2f( (float)cpu_img1.cols, 0), scene_corners[3] + Point2f( (float)cpu_img1.cols, 0),
          Scalar( 0, 255, 0), 2, LINE_AA );
    line( img_matches,
          scene_corners[3] + Point2f( (float)cpu_img1.cols, 0), scene_corners[0] + Point2f( (float)cpu_img1.cols, 0),
          Scalar( 0, 255, 0), 2, LINE_AA );
    return img_matches;
}

}
////////////////////////////////////////////////////
// This program demonstrates the usage of SURF_OCL.
// use cpu findHomography interface to calculate the transformation matrix
int main(int argc, char* argv[])
{
    const char* keys =
        "{ help h    | false           | print help message  }"
        "{ left l    |                 | specify left image  }"
        "{ right r   |                 | specify right image }"
        "{ output o  | SURF_output.jpg | specify output save path (only works in CPU or GPU only mode) }"
        "{ use_cpu c | false           | use CPU algorithms  }"
        "{ use_all a | false           | use both CPU and GPU algorithms}";
    CommandLineParser cmd(argc, argv, keys);
    if (cmd.get<bool>("help"))
    {
        std::cout << "Available options:" << std::endl;
        cmd.printMessage();
        return 0;
    }

    Mat cpu_img1, cpu_img2, cpu_img1_grey, cpu_img2_grey;
    oclMat img1, img2;
    bool useCPU = cmd.get<bool>("c");
    bool useGPU = false;
    bool useALL = cmd.get<bool>("a");

    std::string outpath = cmd.get<std::string>("o");

    cpu_img1 = imread(cmd.get<std::string>("l"));
    CV_Assert(!cpu_img1.empty());
    cvtColor(cpu_img1, cpu_img1_grey, COLOR_BGR2GRAY);
    img1 = cpu_img1_grey;

    cpu_img2 = imread(cmd.get<std::string>("r"));
    CV_Assert(!cpu_img2.empty());
    cvtColor(cpu_img2, cpu_img2_grey, COLOR_BGR2GRAY);
    img2 = cpu_img2_grey;

    if(useALL)
    {
        useCPU = false;
        useGPU = false;
    }
    else if(useCPU==false && useALL==false)
    {
        useGPU = true;
    }

    if(!useCPU)
    {
        std::cout
                << "Device name:"
                << cv::ocl::Context::getContext()->getDeviceInfo().deviceName
                << std::endl;
    }
    double surf_time = 0.;

    //declare input/output
    std::vector<KeyPoint> keypoints1, keypoints2;
    std::vector<DMatch> matches;

    std::vector<KeyPoint> gpu_keypoints1;
    std::vector<KeyPoint> gpu_keypoints2;
    std::vector<DMatch> gpu_matches;

    Mat descriptors1CPU, descriptors2CPU;

    oclMat keypoints1GPU, keypoints2GPU;
    oclMat descriptors1GPU, descriptors2GPU;

    //instantiate detectors/matchers
    SURFDetector<SURF>     cpp_surf;
    SURFDetector<SURF_OCL> ocl_surf;

    SURFMatcher<BFMatcher>      cpp_matcher;
    SURFMatcher<BFMatcher_OCL>  ocl_matcher;

    //-- start of timing section
    if (useCPU)
    {
        for (int i = 0; i <= LOOP_NUM; i++)
        {
            if(i == 1) workBegin();
            cpp_surf(cpu_img1_grey, Mat(), keypoints1, descriptors1CPU);
            cpp_surf(cpu_img2_grey, Mat(), keypoints2, descriptors2CPU);
            cpp_matcher.match(descriptors1CPU, descriptors2CPU, matches);
        }
        workEnd();
        std::cout << "CPP: FOUND " << keypoints1.size() << " keypoints on first image" << std::endl;
        std::cout << "CPP: FOUND " << keypoints2.size() << " keypoints on second image" << std::endl;

        surf_time = getTime();
        std::cout << "SURF run time: " << surf_time / LOOP_NUM << " ms" << std::endl<<"\n";
    }
    else if(useGPU)
    {
        for (int i = 0; i <= LOOP_NUM; i++)
        {
            if(i == 1) workBegin();
            ocl_surf(img1, oclMat(), keypoints1, descriptors1GPU);
            ocl_surf(img2, oclMat(), keypoints2, descriptors2GPU);
            ocl_matcher.match(descriptors1GPU, descriptors2GPU, matches);
        }
        workEnd();
        std::cout << "OCL: FOUND " << keypoints1.size() << " keypoints on first image" << std::endl;
        std::cout << "OCL: FOUND " << keypoints2.size() << " keypoints on second image" << std::endl;

        surf_time = getTime();
        std::cout << "SURF run time: " << surf_time / LOOP_NUM << " ms" << std::endl<<"\n";
    }
    else
    {
        //cpu runs
        for (int i = 0; i <= LOOP_NUM; i++)
        {
            if(i == 1) workBegin();
            cpp_surf(cpu_img1_grey, Mat(), keypoints1, descriptors1CPU);
            cpp_surf(cpu_img2_grey, Mat(), keypoints2, descriptors2CPU);
            cpp_matcher.match(descriptors1CPU, descriptors2CPU, matches);
        }
        workEnd();
        std::cout << "\nCPP: FOUND " << keypoints1.size() << " keypoints on first image" << std::endl;
        std::cout << "CPP: FOUND " << keypoints2.size() << " keypoints on second image" << std::endl;

        surf_time = getTime();
        std::cout << "(CPP)SURF run time: " << surf_time / LOOP_NUM << " ms" << std::endl;

        //gpu runs
        for (int i = 0; i <= LOOP_NUM; i++)
        {
            if(i == 1) workBegin();
            ocl_surf(img1, oclMat(), gpu_keypoints1, descriptors1GPU);
            ocl_surf(img2, oclMat(), gpu_keypoints2, descriptors2GPU);
            ocl_matcher.match(descriptors1GPU, descriptors2GPU, gpu_matches);
        }
        workEnd();
        std::cout << "\nOCL: FOUND " << keypoints1.size() << " keypoints on first image" << std::endl;
        std::cout << "OCL: FOUND " << keypoints2.size() << " keypoints on second image" << std::endl;

        surf_time = getTime();
        std::cout << "(OCL)SURF run time: " << surf_time / LOOP_NUM << " ms" << std::endl<<"\n";

    }

    //--------------------------------------------------------------------------
    std::vector<Point2f> cpu_corner;
    Mat img_matches = drawGoodMatches(cpu_img1, cpu_img2, keypoints1, keypoints2, matches, cpu_corner);

    std::vector<Point2f> gpu_corner;
    Mat ocl_img_matches;
    if(useALL || (!useCPU&&!useGPU))
    {
        ocl_img_matches = drawGoodMatches(cpu_img1, cpu_img2, gpu_keypoints1, gpu_keypoints2, gpu_matches, gpu_corner);

        //check accuracy
        std::cout<<"\nCheck accuracy:\n";

        if(cpu_corner.size()!=gpu_corner.size())
            std::cout<<"Failed\n";
        else
        {
            bool result = false;
            for(size_t i = 0; i < cpu_corner.size(); i++)
            {
                if((std::abs(cpu_corner[i].x - gpu_corner[i].x) > 10)
                        ||(std::abs(cpu_corner[i].y - gpu_corner[i].y) > 10))
                {
                    std::cout<<"Failed\n";
                    result = false;
                    break;
                }
                result = true;
            }
            if(result)
                std::cout<<"Passed\n";
        }
    }

    //-- Show detected matches
    if (useCPU)
    {
        namedWindow("cpu surf matches", 0);
        imshow("cpu surf matches", img_matches);
        imwrite(outpath, img_matches);
    }
    else if(useGPU)
    {
        namedWindow("ocl surf matches", 0);
        imshow("ocl surf matches", img_matches);
        imwrite(outpath, img_matches);
    }
    else
    {
        namedWindow("cpu surf matches", 0);
        imshow("cpu surf matches", img_matches);

        namedWindow("ocl surf matches", 0);
        imshow("ocl surf matches", ocl_img_matches);
    }
    waitKey(0);
    return 0;
}
