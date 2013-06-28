//  demo.cpp
//
//	Here is an example on how to use the descriptor presented in the following paper:
//	A. Alahi, R. Ortiz, and P. Vandergheynst. FREAK: Fast Retina Keypoint. In IEEE Conference on Computer Vision and Pattern Recognition, 2012.
//  CVPR 2012 Open Source Award winner
//
//	Copyright (C) 2011-2012  Signal processing laboratory 2, EPFL,
//	Kirell Benzi (kirell.benzi@epfl.ch),
//	Raphael Ortiz (raphael.ortiz@a3.epfl.ch),
//	Alexandre Alahi (alexandre.alahi@epfl.ch)
//	and Pierre Vandergheynst (pierre.vandergheynst@epfl.ch)
//
//  Redistribution and use in source and binary forms, with or without modification,
//  are permitted provided that the following conditions are met:
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
//  This software is provided by the copyright holders and contributors "as is" and
//  any express or implied warranties, including, but not limited to, the implied
//  warranties of merchantability and fitness for a particular purpose are disclaimed.
//  In no event shall the Intel Corporation or contributors be liable for any direct,
//  indirect, incidental, special, exemplary, or consequential damages
//  (including, but not limited to, procurement of substitute goods or services;
//  loss of use, data, or profits; or business interruption) however caused
//  and on any theory of liability, whether in contract, strict liability,
//  or tort (including negligence or otherwise) arising in any way out of
//  the use of this software, even if advised of the possibility of such damage.

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>

using namespace cv;

static void help( char** argv )
{
    std::cout << "\nUsage: " << argv[0] << " [path/to/image1] [path/to/image2] \n"
              << "This is an example on how to use the keypoint descriptor presented in the following paper: \n"
              << "A. Alahi, R. Ortiz, and P. Vandergheynst. FREAK: Fast Retina Keypoint. \n"
              << "In IEEE Conference on Computer Vision and Pattern Recognition, 2012. CVPR 2012 Open Source Award winner \n"
              << std::endl;
}

int main( int argc, char** argv ) {
    // check http://docs.opencv.org/doc/tutorials/features2d/table_of_content_features2d/table_of_content_features2d.html
    // for OpenCV general detection/matching framework details

    if( argc != 3 ) {
        help(argv);
        return -1;
    }

    // Load images
    Mat imgA = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE );
    if( !imgA.data ) {
        std::cout<< " --(!) Error reading image " << argv[1] << std::endl;
        return -1;
    }

    Mat imgB = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE );
    if( !imgB.data ) {
        std::cout << " --(!) Error reading image " << argv[2] << std::endl;
        return -1;
    }

    std::vector<KeyPoint> keypointsA, keypointsB;
    Mat descriptorsA, descriptorsB;
    std::vector<DMatch> matches;

    // DETECTION
    // Any openCV detector such as
    SurfFeatureDetector detector(2000,4);

    // DESCRIPTOR
    // Our proposed FREAK descriptor
    // (roation invariance, scale invariance, pattern radius corresponding to SMALLEST_KP_SIZE,
    // number of octaves, optional vector containing the selected pairs)
    // FREAK extractor(true, true, 22, 4, std::vector<int>());
    FREAK extractor;

    // MATCHER
    // The standard Hamming distance can be used such as
    // BruteForceMatcher<Hamming> matcher;
    // or the proposed cascade of hamming distance using SSSE3
    BruteForceMatcher<Hamming> matcher;

    // detect
    double t = (double)getTickCount();
    detector.detect( imgA, keypointsA );
    detector.detect( imgB, keypointsB );
    t = ((double)getTickCount() - t)/getTickFrequency();
    std::cout << "detection time [s]: " << t/1.0 << std::endl;

    // extract
    t = (double)getTickCount();
    extractor.compute( imgA, keypointsA, descriptorsA );
    extractor.compute( imgB, keypointsB, descriptorsB );
    t = ((double)getTickCount() - t)/getTickFrequency();
    std::cout << "extraction time [s]: " << t << std::endl;

    // match
    t = (double)getTickCount();
    matcher.match(descriptorsA, descriptorsB, matches);
    t = ((double)getTickCount() - t)/getTickFrequency();
    std::cout << "matching time [s]: " << t << std::endl;

    // Draw matches
    Mat imgMatch;
    drawMatches(imgA, keypointsA, imgB, keypointsB, matches, imgMatch);

    namedWindow("matches", CV_WINDOW_KEEPRATIO);
    imshow("matches", imgMatch);
    waitKey(0);
}
