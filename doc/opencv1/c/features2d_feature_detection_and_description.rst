Feature detection and description
=================================

.. highlight:: c




    
    * **image** The image. Keypoints (corners) will be detected on this. 
    
    
    * **keypoints** Keypoints detected on the image. 
    
    
    * **threshold** Threshold on difference between intensity of center pixel and 
                pixels on circle around this pixel. See description of the algorithm. 
    
    
    * **nonmaxSupression** If it is true then non-maximum supression will be applied to detected corners (keypoints).  
    
    
    

.. index:: ExtractSURF

.. _ExtractSURF:

ExtractSURF
-----------






.. cfunction:: void cvExtractSURF(  const CvArr* image, const CvArr* mask, CvSeq** keypoints, CvSeq** descriptors, CvMemStorage* storage, CvSURFParams params )

    Extracts Speeded Up Robust Features from an image.





    
    :param image: The input 8-bit grayscale image 
    
    
    :param mask: The optional input 8-bit mask. The features are only found in the areas that contain more than 50 %  of non-zero mask pixels 
    
    
    :param keypoints: The output parameter; double pointer to the sequence of keypoints. The sequence of CvSURFPoint structures is as follows: 
    
    
    
    
    ::
    
    
        
         typedef struct CvSURFPoint
         {
            CvPoint2D32f pt; // position of the feature within the image
            int laplacian;   // -1, 0 or +1. sign of the laplacian at the point.
                             // can be used to speedup feature comparison
                             // (normally features with laplacians of different 
                     // signs can not match)
            int size;        // size of the feature
            float dir;       // orientation of the feature: 0..360 degrees
            float hessian;   // value of the hessian (can be used to 
                     // approximately estimate the feature strengths;
                             // see also params.hessianThreshold)
         }
         CvSURFPoint;
        
    
    ..
    
    
    :param descriptors: The optional output parameter; double pointer to the sequence of descriptors. Depending on the params.extended value, each element of the sequence will be either a 64-element or a 128-element floating-point ( ``CV_32F`` ) vector. If the parameter is NULL, the descriptors are not computed 
    
    
    :param storage: Memory storage where keypoints and descriptors will be stored 
    
    
    :param params: Various algorithm parameters put to the structure CvSURFParams: 
    
    
    
    
    ::
    
    
        
         typedef struct CvSURFParams
         {
            int extended; // 0 means basic descriptors (64 elements each),
                          // 1 means extended descriptors (128 elements each)
            double hessianThreshold; // only features with keypoint.hessian 
                  // larger than that are extracted.
                          // good default value is ~300-500 (can depend on the 
                  // average local contrast and sharpness of the image).
                          // user can further filter out some features based on 
                  // their hessian values and other characteristics.
            int nOctaves; // the number of octaves to be used for extraction.
                          // With each next octave the feature size is doubled 
                  // (3 by default)
            int nOctaveLayers; // The number of layers within each octave 
                  // (4 by default)
         }
         CvSURFParams;
        
         CvSURFParams cvSURFParams(double hessianThreshold, int extended=0); 
                  // returns default parameters
        
    
    ..
    
    
    
The function cvExtractSURF finds robust features in the image, as
described in 
Bay06
. For each feature it returns its location, size,
orientation and optionally the descriptor, basic or extended. The function
can be used for object tracking and localization, image stitching etc.

See the
``find_obj.cpp``
demo in OpenCV samples directory.

.. index:: GetStarKeypoints

.. _GetStarKeypoints:

GetStarKeypoints
----------------






.. cfunction:: CvSeq* cvGetStarKeypoints(  const CvArr* image, CvMemStorage* storage, CvStarDetectorParams params=cvStarDetectorParams() )

    Retrieves keypoints using the StarDetector algorithm.





    
    :param image: The input 8-bit grayscale image 
    
    
    :param storage: Memory storage where the keypoints will be stored 
    
    
    :param params: Various algorithm parameters given to the structure CvStarDetectorParams: 
    
    
    
    
    ::
    
    
        
         typedef struct CvStarDetectorParams
         {
            int maxSize; // maximal size of the features detected. The following 
                         // values of the parameter are supported:
                         // 4, 6, 8, 11, 12, 16, 22, 23, 32, 45, 46, 64, 90, 128
            int responseThreshold; // threshold for the approximatd laplacian,
                                   // used to eliminate weak features
            int lineThresholdProjected; // another threshold for laplacian to 
                        // eliminate edges
            int lineThresholdBinarized; // another threshold for the feature 
                        // scale to eliminate edges
            int suppressNonmaxSize; // linear size of a pixel neighborhood 
                        // for non-maxima suppression
         }
         CvStarDetectorParams;
        
    
    ..
    
    
    
The function GetStarKeypoints extracts keypoints that are local
scale-space extremas. The scale-space is constructed by computing
approximate values of laplacians with different sigma's at each
pixel. Instead of using pyramids, a popular approach to save computing
time, all of the laplacians are computed at each pixel of the original
high-resolution image. But each approximate laplacian value is computed
in O(1) time regardless of the sigma, thanks to the use of integral
images. The algorithm is based on the paper 
Agrawal08
, but instead
of a square, hexagon or octagon it uses an 8-end star shape, hence the name,
consisting of overlapping upright and tilted squares.

Each computed feature is represented by the following structure:




::


    
    typedef struct CvStarKeypoint
    {
        CvPoint pt; // coordinates of the feature
        int size; // feature size, see CvStarDetectorParams::maxSize
        float response; // the approximated laplacian value at that point.
    }
    CvStarKeypoint;
    
    inline CvStarKeypoint cvStarKeypoint(CvPoint pt, int size, float response);
    

..

Below is the small usage sample:




::


    
    #include "cv.h"
    #include "highgui.h"
    
    int main(int argc, char** argv)
    {
        const char* filename = argc > 1 ? argv[1] : "lena.jpg";
        IplImage* img = cvLoadImage( filename, 0 ), *cimg;
        CvMemStorage* storage = cvCreateMemStorage(0);
        CvSeq* keypoints = 0;
        int i;
    
        if( !img )
            return 0;
        cvNamedWindow( "image", 1 );
        cvShowImage( "image", img );
        cvNamedWindow( "features", 1 );
        cimg = cvCreateImage( cvGetSize(img), 8, 3 );
        cvCvtColor( img, cimg, CV_GRAY2BGR );
    
        keypoints = cvGetStarKeypoints( img, storage, cvStarDetectorParams(45) );
    
        for( i = 0; i < (keypoints ? keypoints->total : 0); i++ )
        {
            CvStarKeypoint kpt = *(CvStarKeypoint*)cvGetSeqElem(keypoints, i);
            int r = kpt.size/2;
            cvCircle( cimg, kpt.pt, r, CV_RGB(0,255,0));
            cvLine( cimg, cvPoint(kpt.pt.x + r, kpt.pt.y + r),
                cvPoint(kpt.pt.x - r, kpt.pt.y - r), CV_RGB(0,255,0));
            cvLine( cimg, cvPoint(kpt.pt.x - r, kpt.pt.y + r),
                cvPoint(kpt.pt.x + r, kpt.pt.y - r), CV_RGB(0,255,0));
        }
        cvShowImage( "features", cimg );
        cvWaitKey();
    }
    

..

