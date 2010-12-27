#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

using namespace cv;
void help()
{
    printf( "This program shows the use of the \"fern\" plannar PlanarObjectDetector point\n"
    		"descriptor classifier"
    		"Usage:\n"
    		"./find_obj_ferns [<object_filename default: box.png> <scene_filename default:box_in_scene.png>]\n"
    		"\n");
}
int main(int argc, char** argv)
{
    const char* object_filename = argc > 1 ? argv[1] : "box.png";
    const char* scene_filename = argc > 2 ? argv[2] : "box_in_scene.png";
    int i;
    help();
    cvNamedWindow("Object", 1);
    cvNamedWindow("Image", 1);
    cvNamedWindow("Object Correspondence", 1);
    
    Mat object = imread( object_filename, CV_LOAD_IMAGE_GRAYSCALE );
    Mat image;
    
    double imgscale = 1;

    Mat _image = imread( scene_filename, CV_LOAD_IMAGE_GRAYSCALE );
    resize(_image, image, Size(), 1./imgscale, 1./imgscale, INTER_CUBIC);


    if( !object.data || !image.data )
    {
        fprintf( stderr, "Can not load %s and/or %s\n"
                "Usage: find_obj_ferns [<object_filename> <scene_filename>]\n",
                object_filename, scene_filename );
        exit(-1);
    }

    Size patchSize(32, 32);
    LDetector ldetector(7, 20, 2, 2000, patchSize.width, 2);
    ldetector.setVerbose(true);
    PlanarObjectDetector detector;
    
    vector<Mat> objpyr, imgpyr;
    int blurKSize = 3;
    double sigma = 0;
    GaussianBlur(object, object, Size(blurKSize, blurKSize), sigma, sigma);
    GaussianBlur(image, image, Size(blurKSize, blurKSize), sigma, sigma);
    buildPyramid(object, objpyr, ldetector.nOctaves-1);
    buildPyramid(image, imgpyr, ldetector.nOctaves-1);
    
    vector<KeyPoint> objKeypoints, imgKeypoints;
	PatchGenerator gen(0,256,5,true,0.8,1.2,-CV_PI/2,CV_PI/2,-CV_PI/2,CV_PI/2);
    
    string model_filename = format("%s_model.xml.gz", object_filename);
    printf("Trying to load %s ...\n", model_filename.c_str());
    FileStorage fs(model_filename, FileStorage::READ);
    if( fs.isOpened() )
    {
        detector.read(fs.getFirstTopLevelNode());
        printf("Successfully loaded %s.\n", model_filename.c_str());
    }
    else
    {
        printf("The file not found and can not be read. Let's train the model.\n");
        printf("Step 1. Finding the robust keypoints ...\n");
        ldetector.setVerbose(true);
        ldetector.getMostStable2D(object, objKeypoints, 100, gen);
        printf("Done.\nStep 2. Training ferns-based planar object detector ...\n");
        detector.setVerbose(true);
    
        detector.train(objpyr, objKeypoints, patchSize.width, 100, 11, 10000, ldetector, gen);
        printf("Done.\nStep 3. Saving the model to %s ...\n", model_filename.c_str());
        if( fs.open(model_filename, FileStorage::WRITE) )
            detector.write(fs, "ferns_model");
    }
    printf("Now find the keypoints in the image, try recognize them and compute the homography matrix\n");
    fs.release();
        
    vector<Point2f> dst_corners;
    Mat correspond( object.rows + image.rows, std::max(object.cols, image.cols), CV_8UC3);
    correspond = Scalar(0.);
    Mat part(correspond, Rect(0, 0, object.cols, object.rows));
    cvtColor(object, part, CV_GRAY2BGR);
    part = Mat(correspond, Rect(0, object.rows, image.cols, image.rows));
    cvtColor(image, part, CV_GRAY2BGR);
 
    vector<int> pairs;
    Mat H;
    
    double t = (double)getTickCount();
    objKeypoints = detector.getModelPoints();
    ldetector(imgpyr, imgKeypoints, 300);
    
    std::cout << "Object keypoints: " << objKeypoints.size() << "\n";
    std::cout << "Image keypoints: " << imgKeypoints.size() << "\n";
    bool found = detector(imgpyr, imgKeypoints, H, dst_corners, &pairs);
    t = (double)getTickCount() - t;
    printf("%gms\n", t*1000/getTickFrequency());
    
    if( found )
    {
        for( i = 0; i < 4; i++ )
        {
            Point r1 = dst_corners[i%4];
            Point r2 = dst_corners[(i+1)%4];
            line( correspond, Point(r1.x, r1.y+object.rows),
                 Point(r2.x, r2.y+object.rows), Scalar(0,0,255) );
        }
    }
    
    for( i = 0; i < (int)pairs.size(); i += 2 )
    {
        line( correspond, objKeypoints[pairs[i]].pt,
             imgKeypoints[pairs[i+1]].pt + Point2f(0,object.rows),
             Scalar(0,255,0) );
    }
    
    imshow( "Object Correspondence", correspond );
    Mat objectColor;
    cvtColor(object, objectColor, CV_GRAY2BGR);
    for( i = 0; i < (int)objKeypoints.size(); i++ )
    {
        circle( objectColor, objKeypoints[i].pt, 2, Scalar(0,0,255), -1 );
        circle( objectColor, objKeypoints[i].pt, (1 << objKeypoints[i].octave)*15, Scalar(0,255,0), 1 );
    }
    Mat imageColor;
    cvtColor(image, imageColor, CV_GRAY2BGR);
    for( i = 0; i < (int)imgKeypoints.size(); i++ )
    {
        circle( imageColor, imgKeypoints[i].pt, 2, Scalar(0,0,255), -1 );
        circle( imageColor, imgKeypoints[i].pt, (1 << imgKeypoints[i].octave)*15, Scalar(0,255,0), 1 );
    }
    imwrite("correspond.png", correspond );
    imshow( "Object", objectColor );
    imshow( "Image", imageColor );
    
    waitKey(0);
    return 0;
}
