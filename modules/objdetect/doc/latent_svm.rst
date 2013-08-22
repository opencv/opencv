Latent SVM
===============================================================

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

In OpenCV there are C implementation of Latent SVM and C++ wrapper of it.
C version is the structure :ocv:struct:`CvObjectDetection` and a set of functions
working with this structure (see :ocv:func:`cvLoadLatentSvmDetector`,
:ocv:func:`cvReleaseLatentSvmDetector`, :ocv:func:`cvLatentSvmDetectObjects`).
C++ version is the class :ocv:class:`LatentSvmDetector` and has slightly different
functionality in contrast with C version - it supports loading and detection
of several models.

There are two examples of Latent SVM usage: ``samples/c/latentsvmdetect.cpp``
and ``samples/cpp/latentsvm_multidetect.cpp``.

.. highlight:: c


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

  .. ocv:member:: int sizeX
  .. ocv:member:: int sizeY

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

.. ocv:function:: CvSeq* cvLatentSvmDetectObjects( IplImage* image, CvLatentSvmDetector* detector, CvMemStorage* storage, float overlap_threshold=0.5f, int numThreads=-1 )

    :param image: image
    :param detector: LatentSVM detector in internal representation
    :param storage: Memory storage to store the resultant sequence of the object candidate rectangles
    :param overlap_threshold: Threshold for the non-maximum suppression algorithm
    :param numThreads: Number of threads used in parallel version of the algorithm

.. highlight:: cpp

LatentSvmDetector
-----------------
.. ocv:class:: LatentSvmDetector

This is a C++ wrapping class of Latent SVM. It contains internal representation of several
trained Latent SVM detectors (models) and a set of methods to load the detectors and detect objects
using them.

LatentSvmDetector::ObjectDetection
----------------------------------
.. ocv:struct:: LatentSvmDetector::ObjectDetection

  Structure contains the detection information.

  .. ocv:member:: Rect rect

     bounding box for a detected object

  .. ocv:member:: float score

     confidence level

  .. ocv:member:: int classID

     class (model or detector) ID that detect an object


LatentSvmDetector::LatentSvmDetector
------------------------------------
Two types of constructors.

.. ocv:function:: LatentSvmDetector::LatentSvmDetector()

.. ocv:function:: LatentSvmDetector::LatentSvmDetector(const vector<string>& filenames, const vector<string>& classNames=vector<string>())



    :param filenames: A set of filenames storing the trained detectors (models). Each file contains one model. See examples of such files here /opencv_extra/testdata/cv/latentsvmdetector/models_VOC2007/.

    :param classNames: A set of trained models names. If it's empty then the name of each model will be constructed from the name of file containing the model. E.g. the model stored in "/home/user/cat.xml" will get the name "cat".

LatentSvmDetector::~LatentSvmDetector
-------------------------------------
Destructor.

.. ocv:function:: LatentSvmDetector::~LatentSvmDetector()

LatentSvmDetector::~clear
-------------------------
Clear all trained models and their names stored in an class object.

.. ocv:function:: void LatentSvmDetector::clear()

LatentSvmDetector::load
-----------------------
Load the trained models from given ``.xml`` files and return ``true`` if at least one model was loaded.

.. ocv:function:: bool LatentSvmDetector::load( const vector<string>& filenames, const vector<string>& classNames=vector<string>() )

    :param filenames: A set of filenames storing the trained detectors (models). Each file contains one model. See examples of such files here /opencv_extra/testdata/cv/latentsvmdetector/models_VOC2007/.

    :param classNames: A set of trained models names. If it's empty then the name of each model will be constructed from the name of file containing the model. E.g. the model stored in "/home/user/cat.xml" will get the name "cat".

LatentSvmDetector::detect
-------------------------
Find rectangular regions in the given image that are likely to contain objects of loaded classes (models)
and corresponding confidence levels.

.. ocv:function:: void LatentSvmDetector::detect( const Mat& image, vector<ObjectDetection>& objectDetections, float overlapThreshold=0.5f, int numThreads=-1 )

    :param image: An image.
    :param objectDetections: The detections: rectangulars, scores and class IDs.
    :param overlapThreshold: Threshold for the non-maximum suppression algorithm.
    :param numThreads: Number of threads used in parallel version of the algorithm.

LatentSvmDetector::getClassNames
--------------------------------
Return the class (model) names that were passed in constructor or method ``load`` or extracted from models filenames in those methods.

.. ocv:function:: const vector<string>& LatentSvmDetector::getClassNames() const

LatentSvmDetector::getClassCount
--------------------------------
Return a count of loaded models (classes).

.. ocv:function:: size_t LatentSvmDetector::getClassCount() const


.. [Felzenszwalb2010] Felzenszwalb, P. F. and Girshick, R. B. and McAllester, D. and Ramanan, D. *Object Detection with Discriminatively Trained Part Based Models*. PAMI, vol. 32, no. 9, pp. 1627-1645, September 2010
