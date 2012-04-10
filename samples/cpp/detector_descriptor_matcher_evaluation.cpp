/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>

#include <limits>
#include <cstdio>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

/*
The algorithm:

for each tested combination of detector+descriptor+matcher:
    
    create detector, descriptor and matcher,
    load their params if they are there, otherwise use the default ones and save them
    
    for each dataset:
    
        load reference image
        detect keypoints in it, compute descriptors
        
        for each transformed image:
            load the image
            load the transformation matrix
            detect keypoints in it too, compute descriptors
            
            find matches
            transform keypoints from the first image using the ground-truth matrix
            
            compute the number of matched keypoints, i.e. for each pair (i,j) found by a matcher compare
            j-th keypoint from the second image with the transformed i-th keypoint. If they are close, +1.
            
            so, we have:
               N - number of keypoints in the first image that are also visible
               (after transformation) on the second image
               
               N1 - number of keypoints in the first image that have been matched.
               
               n - number of the correct matches found by the matcher
               
               n/N1 - precision
               n/N - recall (?)
               
            we store (N, n/N1, n/N) (where N is stored primarily for tuning the detector's thresholds,
                                     in order to semi-equalize their keypoints counts)
            
*/

typedef Vec3f TVec; // (N, n/N1, n/N) - see above

static void saveloadDDM( const string& params_filename,
                         Ptr<FeatureDetector>& detector,
                         Ptr<DescriptorExtractor>& descriptor,
                         Ptr<DescriptorMatcher>& matcher )
{
    FileStorage fs(params_filename, FileStorage::READ);
    if( fs.isOpened() )
    {
        detector->read(fs["detector"]);
        descriptor->read(fs["descriptor"]);
        matcher->read(fs["matcher"]);
    }
    else
    {
        fs.open(params_filename, FileStorage::WRITE);
        fs << "detector" << "{";
        detector->write(fs);
        fs << "}" << "descriptor" << "{";
        descriptor->write(fs);
        fs << "}" << "matcher" << "{";
        matcher->write(fs);
        fs << "}";
    }
}

static Mat loadMat(const string& fsname)
{
    FileStorage fs(fsname, FileStorage::READ);
    Mat m;
    fs.getFirstTopLevelNode() >> m;
    return m;
}

static void transformKeypoints( const vector<KeyPoint>& kp,
                                vector<vector<Point2f> >& contours,
                                const Mat& H )
{
    const float scale = 256.f;
    size_t i, n = kp.size();
    contours.resize(n);
    vector<Point> temp;
    
    for( i = 0; i < n; i++ )
    {
        ellipse2Poly(Point2f(kp[i].pt.x*scale, kp[i].pt.y*scale),
                     Size2f(kp[i].size*scale, kp[i].size*scale),
                     0, 0, 360, 12, temp);
        Mat(temp).convertTo(contours[i], CV_32F, 1./scale);
        perspectiveTransform(contours[i], contours[i], H);
    }
}


static TVec proccessMatches( Size imgsize,
                             const vector<DMatch>& matches,
                             const vector<vector<Point2f> >& kp1t_contours,
                             const vector<vector<Point2f> >& kp_contours,
                             double overlapThreshold )
{
    const double visibilityThreshold = 0.6;
    
    // 1. [preprocessing] find bounding rect for each element of kp1t_contours and kp_contours.
    // 2. [cross-check] for each DMatch (iK, i1)
    //        update best_match[i1] using DMatch::distance.
    // 3. [compute overlapping] for each i1 (keypoint from the first image) do:
    //        if i1-th keypoint is outside of image, skip it
    //        increment N
    //        if best_match[i1] is initialized, increment N1
    //        if kp_contours[best_match[i1]] and kp1t_contours[i1] overlap by overlapThreshold*100%,
    //        increment n. Use bounding rects to speedup this step
    
    int i, size1 = (int)kp1t_contours.size(), size = (int)kp_contours.size(), msize = (int)matches.size();
    vector<DMatch> best_match(size1);
    vector<Rect> rects1(size1), rects(size);
    
    // proprocess
    for( i = 0; i < size1; i++ )
        rects1[i] = boundingRect(kp1t_contours[i]);
        
    for( i = 0; i < size; i++ )
        rects[i] = boundingRect(kp_contours[i]);
    
    // cross-check
    for( i = 0; i < msize; i++ )
    {
        DMatch m = matches[i];
        int i1 = m.trainIdx, iK = m.queryIdx;
        CV_Assert( 0 <= i1 && i1 < size1 && 0 <= iK && iK < size );
        if( best_match[i1].trainIdx < 0 || best_match[i1].distance > m.distance )
            best_match[i1] = m;
    }
    
    int N = 0, N1 = 0, n = 0;
    
    // overlapping
    for( i = 0; i < size1; i++ )
    {
        int i1 = i, iK = best_match[i].queryIdx;
        if( iK >= 0 )
            N1++;
        
        Rect r = rects1[i] & Rect(0, 0, imgsize.width, imgsize.height);
        if( r.area() < visibilityThreshold*rects1[i].area() )
            continue;
        N++;
        
        if( iK < 0 || (rects1[i1] & rects[iK]).area() == 0 )
            continue;
        
        double n_area = intersectConvexConvex(kp1t_contours[i1], kp_contours[iK], noArray(), true);
        if( n_area == 0 )
            continue;
        
        double area1 = contourArea(kp1t_contours[i1], false);
        double area = contourArea(kp_contours[iK], false);
        
        double ratio = n_area/(area1 + area - n_area);
        n += ratio >= overlapThreshold;
    }
    
    return TVec((float)N, (float)n/std::max(N1, 1), (float)n/std::max(N, 1));
}


static void saveResults(const string& dir, const string& name, const string& dsname,
                        const vector<TVec>& results, const int* xvals)
{
    string fname1 = format("%s%s_%s_precision.csv", dir.c_str(), name.c_str(), dsname.c_str());
    string fname2 = format("%s%s_%s_recall.csv", dir.c_str(), name.c_str(), dsname.c_str());
    FILE* f1 = fopen(fname1.c_str(), "wt");
    FILE* f2 = fopen(fname2.c_str(), "wt");
    
    for( size_t i = 0; i < results.size(); i++ )
    {
        fprintf(f1, "%d, %.1f\n", xvals[i], results[i][1]*100);
        fprintf(f2, "%d, %.1f\n", xvals[i], results[i][2]*100);
    }
    fclose(f1);
    fclose(f2);
}


int main(int argc, char** argv)
{
    static const char* ddms[] =
    {
        "ORBX_BF", "ORB", "ORB", "BruteForce-Hamming",
        //"ORB_BF", "ORB", "ORB", "BruteForce-Hamming",
        //"ORB3_BF", "ORB", "ORB", "BruteForce-Hamming(2)",
        //"ORB4_BF", "ORB", "ORB", "BruteForce-Hamming(2)",
        //"ORB_LSH", "ORB", "ORB", "LSH"
        //"SURF_BF", "SURF", "SURF", "BruteForce",
        0
    };
    
    static const char* datasets[] =
    {
        "bark", "bikes", "boat", "graf", "leuven", "trees", "ubc", "wall", 0
    };
    
    static const int imgXVals[] = { 2, 3, 4, 5, 6 }; // if scale, blur or light changes
    static const int viewpointXVals[] = { 20, 30, 40, 50, 60 }; // if viewpoint changes
    static const int jpegXVals[] = { 60, 80, 90, 95, 98 }; // if jpeg compression
    
    const double overlapThreshold = 0.6;
    
    vector<vector<vector<TVec> > > results; // indexed as results[ddm][dataset][testcase]
    
    string dataset_dir = string(getenv("OPENCV_TEST_DATA_PATH")) +
        "/cv/detectors_descriptors_evaluation/images_datasets";
                                
    string dir=argc > 1 ? argv[1] : ".";
    
    if( dir[dir.size()-1] != '\\' && dir[dir.size()-1] != '/' )
        dir += "/";
    
    int result = system(("mkdir " + dir).c_str());
    CV_Assert(result == 0);
    
    for( int i = 0; ddms[i*4] != 0; i++ )
    {
        const char* name = ddms[i*4];
        const char* detector_name = ddms[i*4+1];
        const char* descriptor_name = ddms[i*4+2];
        const char* matcher_name = ddms[i*4+3];
        string params_filename = dir + string(name) + "_params.yml";
        
        cout << "Testing " << name << endl;
        
        Ptr<FeatureDetector> detector = FeatureDetector::create(detector_name);
        Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create(descriptor_name);
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(matcher_name);
        
        saveloadDDM( params_filename, detector, descriptor, matcher );
        
        results.push_back(vector<vector<TVec> >());
        
        for( int j = 0; datasets[j] != 0; j++ )
        {
            const char* dsname = datasets[j];
            
            cout << "\ton " << dsname << " ";
            cout.flush();
            
            const int* xvals = strcmp(dsname, "ubc") == 0 ? jpegXVals :
                strcmp(dsname, "graf") == 0 || strcmp(dsname, "wall") == 0 ? viewpointXVals : imgXVals;
            
            vector<KeyPoint> kp1, kp;
            vector<DMatch> matches;
            vector<vector<Point2f> > kp1t_contours, kp_contours;
            Mat desc1, desc;
            
            Mat img1 = imread(format("%s/%s/img1.png", dataset_dir.c_str(), dsname), 0);
            CV_Assert( !img1.empty() );
            
            detector->detect(img1, kp1);
            descriptor->compute(img1, kp1, desc1);
            
            results[i].push_back(vector<TVec>());
            
            for( int k = 2; ; k++ )
            {
                cout << ".";
                cout.flush();
                Mat imgK = imread(format("%s/%s/img%d.png", dataset_dir.c_str(), dsname, k), 0);
                if( imgK.empty() )
                    break;
                
                detector->detect(imgK, kp);
                descriptor->compute(imgK, kp, desc);
                matcher->match( desc, desc1, matches );
                
                Mat H = loadMat(format("%s/%s/H1to%dp.xml", dataset_dir.c_str(), dsname, k));
                
                transformKeypoints( kp1, kp1t_contours, H );
                transformKeypoints( kp, kp_contours, Mat::eye(3, 3, CV_64F));
                
                TVec r = proccessMatches( imgK.size(), matches, kp1t_contours, kp_contours, overlapThreshold );
                results[i][j].push_back(r);
            }
            
            saveResults(dir, name, dsname, results[i][j], xvals);
            cout << endl;
        }
    }
}
