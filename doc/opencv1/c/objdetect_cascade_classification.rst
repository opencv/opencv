Cascade Classification
======================

.. highlight:: c



Haar Feature-based Cascade Classifier for Object Detection
----------------------------------------------------------


The object detector described below has been initially proposed by Paul Viola
:ref:`Viola01`
and improved by Rainer Lienhart
:ref:`Lienhart02`
. First, a classifier (namely a 
*cascade of boosted classifiers working with haar-like features*
) is trained with a few hundred sample views of a particular object (i.e., a face or a car), called positive examples, that are scaled to the same size (say, 20x20), and negative examples - arbitrary images of the same size.

After a classifier is trained, it can be applied to a region of interest
(of the same size as used during the training) in an input image. The
classifier outputs a "1" if the region is likely to show the object
(i.e., face/car), and "0" otherwise. To search for the object in the
whole image one can move the search window across the image and check
every location using the classifier. The classifier is designed so that
it can be easily "resized" in order to be able to find the objects of
interest at different sizes, which is more efficient than resizing the
image itself. So, to find an object of an unknown size in the image the
scan procedure should be done several times at different scales.

The word "cascade" in the classifier name means that the resultant
classifier consists of several simpler classifiers (
*stages*
) that
are applied subsequently to a region of interest until at some stage the
candidate is rejected or all the stages are passed. The word "boosted"
means that the classifiers at every stage of the cascade are complex
themselves and they are built out of basic classifiers using one of four
different 
``boosting``
techniques (weighted voting). Currently
Discrete Adaboost, Real Adaboost, Gentle Adaboost and Logitboost are
supported. The basic classifiers are decision-tree classifiers with at
least 2 leaves. Haar-like features are the input to the basic classifers,
and are calculated as described below. The current algorithm uses the
following Haar-like features:



.. image:: ../pics/haarfeatures.png



The feature used in a particular classifier is specified by its shape (1a, 2b etc.), position within the region of interest and the scale (this scale is not the same as the scale used at the detection stage, though these two scales are multiplied). For example, in the case of the third line feature (2c) the response is calculated as the difference between the sum of image pixels under the rectangle covering the whole feature (including the two white stripes and the black stripe in the middle) and the sum of the image pixels under the black stripe multiplied by 3 in order to compensate for the differences in the size of areas. The sums of pixel values over a rectangular regions are calculated rapidly using integral images (see below and the 
:ref:`Integral`
description).

To see the object detector at work, have a look at the HaarFaceDetect demo.

The following reference is for the detection part only. There
is a separate application called 
``haartraining``
that can
train a cascade of boosted classifiers from a set of samples. See
``opencv/apps/haartraining``
for details.


.. index:: CvHaarFeature, CvHaarClassifier, CvHaarStageClassifier, CvHaarClassifierCascade

.. _CvHaarFeature, CvHaarClassifier, CvHaarStageClassifier, CvHaarClassifierCascade:

CvHaarFeature, CvHaarClassifier, CvHaarStageClassifier, CvHaarClassifierCascade
-------------------------------------------------------------------------------



.. ctype:: CvHaarFeature, CvHaarClassifier, CvHaarStageClassifier, CvHaarClassifierCascade



Boosted Haar classifier structures.




::


    
    #define CV_HAAR_FEATURE_MAX  3
    
    /* a haar feature consists of 2-3 rectangles with appropriate weights */
    typedef struct CvHaarFeature
    {
        int  tilted;  /* 0 means up-right feature, 1 means 45--rotated feature */
    
        /* 2-3 rectangles with weights of opposite signs and
           with absolute values inversely proportional to the areas of the 
           rectangles.  If rect[2].weight !=0, then
           the feature consists of 3 rectangles, otherwise it consists of 2 */
        struct
        {
            CvRect r;
            float weight;
        } rect[CV_HAAR_FEATURE_MAX];
    }
    CvHaarFeature;
    
    /* a single tree classifier (stump in the simplest case) that returns the 
       response for the feature at the particular image location (i.e. pixel 
       sum over subrectangles of the window) and gives out a value depending 
       on the response */
    typedef struct CvHaarClassifier
    {
        int count;  /* number of nodes in the decision tree */
    
        /* these are "parallel" arrays. Every index ``i``
           corresponds to a node of the decision tree (root has 0-th index).
    
           left[i] - index of the left child (or negated index if the 
             left child is a leaf)
           right[i] - index of the right child (or negated index if the 
              right child is a leaf)
           threshold[i] - branch threshold. if feature responce is <= threshold, 
                        left branch is chosen, otherwise right branch is chosen.
           alpha[i] - output value correponding to the leaf. */
        CvHaarFeature* haar_feature;
        float* threshold;
        int* left;
        int* right;
        float* alpha;
    }
    CvHaarClassifier;
    
    /* a boosted battery of classifiers(=stage classifier):
       the stage classifier returns 1
       if the sum of the classifiers responses
       is greater than ``threshold`` and 0 otherwise */
    typedef struct CvHaarStageClassifier
    {
        int  count;  /* number of classifiers in the battery */
        float threshold; /* threshold for the boosted classifier */
        CvHaarClassifier* classifier; /* array of classifiers */
    
        /* these fields are used for organizing trees of stage classifiers,
           rather than just stright cascades */
        int next;
        int child;
        int parent;
    }
    CvHaarStageClassifier;
    
    typedef struct CvHidHaarClassifierCascade CvHidHaarClassifierCascade;
    
    /* cascade or tree of stage classifiers */
    typedef struct CvHaarClassifierCascade
    {
        int  flags; /* signature */
        int  count; /* number of stages */
        CvSize orig_window_size; /* original object size (the cascade is 
                                trained for) */
    
        /* these two parameters are set by cvSetImagesForHaarClassifierCascade */
        CvSize real_window_size; /* current object size */
        double scale; /* current scale */
        CvHaarStageClassifier* stage_classifier; /* array of stage classifiers */
        CvHidHaarClassifierCascade* hid_cascade; /* hidden optimized 
                            representation of the 
                            cascade, created by 
                    cvSetImagesForHaarClassifierCascade */
    }
    CvHaarClassifierCascade;
    

..

All the structures are used for representing a cascaded of boosted Haar classifiers. The cascade has the following hierarchical structure:


\begin{verbatim}
Cascade:
        Stage,,1,,:
            Classifier,,11,,:
                Feature,,11,,
            Classifier,,12,,:
                Feature,,12,,
            ...
        Stage,,2,,:
            Classifier,,21,,:
                Feature,,21,,
            ...
        ...

\end{verbatim}
The whole hierarchy can be constructed manually or loaded from a file or an embedded base using the function 
:ref:`LoadHaarClassifierCascade`
.


.. index:: LoadHaarClassifierCascade

.. _LoadHaarClassifierCascade:

LoadHaarClassifierCascade
-------------------------






.. cfunction:: CvHaarClassifierCascade* cvLoadHaarClassifierCascade(  const char* directory, CvSize orig_window_size )

    Loads a trained cascade classifier from a file or the classifier database embedded in OpenCV.





    
    :param directory: Name of the directory containing the description of a trained cascade classifier 
    
    
    :param orig_window_size: Original size of the objects the cascade has been trained on. Note that it is not stored in the cascade and therefore must be specified separately 
    
    
    
The function loads a trained cascade
of haar classifiers from a file or the classifier database embedded in
OpenCV. The base can be trained using the 
``haartraining``
application
(see opencv/apps/haartraining for details).

**The function is obsolete**
. Nowadays object detection classifiers are stored in XML or YAML files, rather than in directories. To load a cascade from a file, use the 
:ref:`Load`
function.


.. index:: HaarDetectObjects

.. _HaarDetectObjects:

HaarDetectObjects
-----------------







::


    
    

..



.. cfunction:: CvSeq* cvHaarDetectObjects(  const CvArr* image, CvHaarClassifierCascade* cascade, CvMemStorage* storage, double scaleFactor=1.1, int minNeighbors=3, int flags=0, CvSize minSize=cvSize(0, 0), CvSize maxSize=cvSize(0,0) )

    Detects objects in the image.

typedef struct CvAvgComp
{
    CvRect rect; /* bounding rectangle for the object (average rectangle of a group) */
    int neighbors; /* number of neighbor rectangles in the group */
}
CvAvgComp;




    
    :param image: Image to detect objects in 
    
    
    :param cascade: Haar classifier cascade in internal representation 
    
    
    :param storage: Memory storage to store the resultant sequence of the object candidate rectangles 
    
    
    :param scaleFactor: The factor by which the search window is scaled between the subsequent scans, 1.1 means increasing window by 10 %   
    
    
    :param minNeighbors: Minimum number (minus 1) of neighbor rectangles that makes up an object. All the groups of a smaller number of rectangles than  ``min_neighbors`` -1 are rejected. If  ``minNeighbors``  is 0, the function does not any grouping at all and returns all the detected candidate rectangles, which may be useful if the user wants to apply a customized grouping procedure 
    
    
    :param flags: Mode of operation. Currently the only flag that may be specified is  ``CV_HAAR_DO_CANNY_PRUNING`` . If it is set, the function uses Canny edge detector to reject some image regions that contain too few or too much edges and thus can not contain the searched object. The particular threshold values are tuned for face detection and in this case the pruning speeds up the processing 
    
    
    :param minSize: Minimum window size. By default, it is set to the size of samples the classifier has been trained on ( :math:`\sim 20\times 20`  for face detection) 
    
    
    :param maxSize: Maximum window size to use. By default, it is set to the size of the image. 
    
    
    
The function finds rectangular regions in the given image that are likely to contain objects the cascade has been trained for and returns those regions as a sequence of rectangles. The function scans the image several times at different scales (see 
:ref:`SetImagesForHaarClassifierCascade`
). Each time it considers overlapping regions in the image and applies the classifiers to the regions using 
:ref:`RunHaarClassifierCascade`
. It may also apply some heuristics to reduce number of analyzed regions, such as Canny prunning. After it has proceeded and collected the candidate rectangles (regions that passed the classifier cascade), it groups them and returns a sequence of average rectangles for each large enough group. The default parameters (
``scale_factor``
=1.1, 
``min_neighbors``
=3, 
``flags``
=0) are tuned for accurate yet slow object detection. For a faster operation on real video images the settings are: 
``scale_factor``
=1.2, 
``min_neighbors``
=2, 
``flags``
=
``CV_HAAR_DO_CANNY_PRUNING``
, 
``min_size``
=
*minimum possible face size*
(for example, 
:math:`\sim`
1/4 to 1/16 of the image area in the case of video conferencing).




::


    
    #include "cv.h"
    #include "highgui.h"
    
    CvHaarClassifierCascade* load_object_detector( const char* cascade_path )
    {
        return (CvHaarClassifierCascade*)cvLoad( cascade_path );
    }
    
    void detect_and_draw_objects( IplImage* image,
                                  CvHaarClassifierCascade* cascade,
                                  int do_pyramids )
    {
        IplImage* small_image = image;
        CvMemStorage* storage = cvCreateMemStorage(0);
        CvSeq* faces;
        int i, scale = 1;
    
        /* if the flag is specified, down-scale the input image to get a
           performance boost w/o loosing quality (perhaps) */
        if( do_pyramids )
        {
            small_image = cvCreateImage( cvSize(image->width/2,image->height/2), IPL_DEPTH_8U, 3 );
            cvPyrDown( image, small_image, CV_GAUSSIAN_5x5 );
            scale = 2;
        }
    
        /* use the fastest variant */
        faces = cvHaarDetectObjects( small_image, cascade, storage, 1.2, 2, CV_HAAR_DO_CANNY_PRUNING );
    
        /* draw all the rectangles */
        for( i = 0; i < faces->total; i++ )
        {
            /* extract the rectanlges only */
            CvRect face_rect = *(CvRect*)cvGetSeqElem( faces, i );
            cvRectangle( image, cvPoint(face_rect.x*scale,face_rect.y*scale),
                         cvPoint((face_rect.x+face_rect.width)*scale,
                                 (face_rect.y+face_rect.height)*scale),
                         CV_RGB(255,0,0), 3 );
        }
    
        if( small_image != image )
            cvReleaseImage( &small_image );
        cvReleaseMemStorage( &storage );
    }
    
    /* takes image filename and cascade path from the command line */
    int main( int argc, char** argv )
    {
        IplImage* image;
        if( argc==3 && (image = cvLoadImage( argv[1], 1 )) != 0 )
        {
            CvHaarClassifierCascade* cascade = load_object_detector(argv[2]);
            detect_and_draw_objects( image, cascade, 1 );
            cvNamedWindow( "test", 0 );
            cvShowImage( "test", image );
            cvWaitKey(0);
            cvReleaseHaarClassifierCascade( &cascade );
            cvReleaseImage( &image );
        }
    
        return 0;
    }
    

..


.. index:: SetImagesForHaarClassifierCascade

.. _SetImagesForHaarClassifierCascade:

SetImagesForHaarClassifierCascade
---------------------------------






.. cfunction:: void cvSetImagesForHaarClassifierCascade(  CvHaarClassifierCascade* cascade, const CvArr* sum, const CvArr* sqsum, const CvArr* tilted_sum, double scale )

    Assigns images to the hidden cascade.





    
    :param cascade: Hidden Haar classifier cascade, created by  :ref:`CreateHidHaarClassifierCascade` 
    
    
    :param sum: Integral (sum) single-channel image of 32-bit integer format. This image as well as the two subsequent images are used for fast feature evaluation and brightness/contrast normalization. They all can be retrieved from input 8-bit or floating point single-channel image using the function  :ref:`Integral` 
    
    
    :param sqsum: Square sum single-channel image of 64-bit floating-point format 
    
    
    :param tilted_sum: Tilted sum single-channel image of 32-bit integer format 
    
    
    :param scale: Window scale for the cascade. If  ``scale``  =1, the original window size is used (objects of that size are searched) - the same size as specified in  :ref:`LoadHaarClassifierCascade`  (24x24 in the case of  ``default_face_cascade`` ), if  ``scale``  =2, a two times larger window is used (48x48 in the case of default face cascade). While this will speed-up search about four times, faces smaller than 48x48 cannot be detected 
    
    
    
The function assigns images and/or window scale to the hidden classifier cascade. If image pointers are NULL, the previously set images are used further (i.e. NULLs mean "do not change images"). Scale parameter has no such a "protection" value, but the previous value can be retrieved by the 
:ref:`GetHaarClassifierCascadeScale`
function and reused again. The function is used to prepare cascade for detecting object of the particular size in the particular image. The function is called internally by 
:ref:`HaarDetectObjects`
, but it can be called by the user if they are using the lower-level function 
:ref:`RunHaarClassifierCascade`
.


.. index:: ReleaseHaarClassifierCascade

.. _ReleaseHaarClassifierCascade:

ReleaseHaarClassifierCascade
----------------------------






.. cfunction:: void cvReleaseHaarClassifierCascade(  CvHaarClassifierCascade** cascade )

    Releases the haar classifier cascade.





    
    :param cascade: Double pointer to the released cascade. The pointer is cleared by the function 
    
    
    
The function deallocates the cascade that has been created manually or loaded using 
:ref:`LoadHaarClassifierCascade`
or 
:ref:`Load`
.


.. index:: RunHaarClassifierCascade

.. _RunHaarClassifierCascade:

RunHaarClassifierCascade
------------------------






.. cfunction:: int cvRunHaarClassifierCascade(  CvHaarClassifierCascade* cascade, CvPoint pt, int start_stage=0 )

    Runs a cascade of boosted classifiers at the given image location.





    
    :param cascade: Haar classifier cascade 
    
    
    :param pt: Top-left corner of the analyzed region. Size of the region is a original window size scaled by the currenly set scale. The current window size may be retrieved using the  :ref:`GetHaarClassifierCascadeWindowSize`  function 
    
    
    :param start_stage: Initial zero-based index of the cascade stage to start from. The function assumes that all the previous stages are passed. This feature is used internally by  :ref:`HaarDetectObjects`  for better processor cache utilization 
    
    
    
The function runs the Haar classifier
cascade at a single image location. Before using this function the
integral images and the appropriate scale (window size) should be set
using 
:ref:`SetImagesForHaarClassifierCascade`
. The function returns
a positive value if the analyzed rectangle passed all the classifier stages
(it is a candidate) and a zero or negative value otherwise.

