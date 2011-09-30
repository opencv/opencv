Latent SVM
===============================================================

.. highlight:: cpp


Discriminatively Trained Part Based Models for Object Detection
---------------------------------------------------------------

The object detector described below has been initially proposed by
P.F. Felzenszwalb in [Felzenszwalb2010]_.  It is based on a
Dalal-Triggs detector that uses a single filter on histogram of
oriented gradients (HOG) features to represent an object category.
This detector uses a sliding window approach, where a filter is
applied at all positions and scales of an image. The first
innovation is enriching the Dalal-Triggs model using a
star-structured part-based model defined by a "root" filter
(analogous to the Dalal-Triggs filter) plus a set of parts filters
and associated deformation models. The score of one of star models
at a particular position and scale within an image is the score of
the root filter at the given location plus the sum over parts of the
maximum, over placements of that part, of the part filter score on
its location minus a deformation cost easuring the deviation of the
part from its ideal location relative to the root. Both root and
part filter scores are defined by the dot product between a filter
(a set of weights) and a subwindow of a feature pyramid computed
from the input image. Another improvement is a representation of the
class of models by a mixture of star models. The score of a mixture
model at a particular position and scale is the maximum over
components, of the score of that component model at the given
location.


CvLSVMFilterPosition
--------------------
.. ocv:struct:: CvLSVMFilterPosition

Structure describes the position of the filter in the feature pyramid.

    .. ocv:member:: unsigned int l
    
        level in the feature pyramid
        
    .. ocv:member:: unsigned int x
    
        x-coordinate in level l
        
    .. ocv:member:: unsigned int y
    
        y-coordinate in level l
        
        
CvLSVMFilterObject
------------------
.. ocv:struct:: CvLSVMFilterObject

Description of the filter, which corresponds to the part of the object.

    .. ocv:member:: CvLSVMFilterPosition V
        
        ideal (penalty = 0) position of the partial filter
        from the root filter position (V_i in the paper)
        
    .. ocv:member:: float fineFunction[4]
        
        vector describes penalty function (d_i in the paper)
        pf[0] * x + pf[1] * y + pf[2] * x^2 + pf[3] * y^2
        
    .. ocv:member:: int sizeX, sizeY
        
        Rectangular map (sizeX x sizeY),
        every cell stores feature vector (dimension = p)
        
    .. ocv:member:: int numFeatures
    
        number of features
        
    .. ocv:member:: float *H
    
        matrix of feature vectors to set and get 
        feature vectors (i,j) used formula H[(j * sizeX + i) * p + k], 
        where k - component of feature vector in cell (i, j)
        
CvLatentSvmDetector
-------------------
.. ocv:struct:: CvLatentSvmDetector

Structure contains internal representation of trained Latent SVM detector.

    .. ocv:member:: int num_filters
    
        total number of filters (root plus part) in model
        
    .. ocv:member:: int num_components
    
        number of components in model
    
    .. ocv:member:: int* num_part_filters
    
        array containing number of part filters for each component
        
    .. ocv:member:: CvLSVMFilterObject** filters
    
        root and part filters for all model components
        
    .. ocv:member:: float* b
    
        biases for all model components
        
    .. ocv:member:: float score_threshold
    
        confidence level threshold
        
        
CvObjectDetection
-----------------
.. ocv:struct:: CvObjectDetection

Structure contains the bounding box and confidence level for detected object.

    .. ocv:member:: CvRect rect
    
        bounding box for a detected object
        
    .. ocv:member:: float score
    
        confidence level


cvLoadLatentSvmDetector
-----------------------

Loads trained detector from a file.

.. ocv:function:: CvLatentSvmDetector* cvLoadLatentSvmDetector(const char* filename)

    :param filename: Name of the file containing the description of a trained detector
    

cvReleaseLatentSvmDetector
--------------------------

Release memory allocated for CvLatentSvmDetector structure.

.. ocv:function:: void cvReleaseLatentSvmDetector(CvLatentSvmDetector** detector)

    :param detector: CvLatentSvmDetector structure to be released


cvLatentSvmDetectObjects
------------------------

Find rectangular regions in the given image that are likely to contain objects 
and corresponding confidence levels.

.. ocv:function:: CvSeq* cvLatentSvmDetectObjects(IplImage* image, CvLatentSvmDetector* detector,  CvMemStorage* storage,  float overlap_threshold, int numThreads)
    
    :param image: image 
    :param detector: LatentSVM detector in internal representation
    :param storage: Memory storage to store the resultant sequence of the object candidate rectangles
    :param overlap_threshold: Threshold for the non-maximum suppression algorithm
    :param numThreads: Number of threads used in parallel version of the algorithm
    
    
.. [Felzenszwalb2010] Felzenszwalb, P. F. and Girshick, R. B. and McAllester, D. and Ramanan, D. *Object Detection with Discriminatively Trained Part Based Models*. PAMI, vol. 32, no. 9, pp. 1627-1645, September 2010 


