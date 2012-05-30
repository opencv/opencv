Feature Detection and Description
=================================

.. highlight:: cpp

FAST
--------
Detects corners using the FAST algorithm

.. ocv:function:: void FAST( InputArray image, vector<KeyPoint>& keypoints, int threshold, bool nonmaxSupression=true )

    :param image: Image where keypoints (corners) are detected.

    :param keypoints: Keypoints detected on the image.

    :param threshold: Threshold on difference between intensity of the central pixel and pixels on a circle around this pixel. See the algorithm description below.

    :param nonmaxSupression: If it is true, non-maximum suppression is applied to detected corners (keypoints).

Detects corners using the FAST algorithm by [Rosten06]_.

.. [Rosten06] E. Rosten. Machine Learning for High-speed Corner Detection, 2006.


MSER
----
.. ocv:class:: MSER : public FeatureDetector

Maximally stable extremal region extractor. ::

    class MSER : public CvMSERParams
    {
    public:
        // default constructor
        MSER();
        // constructor that initializes all the algorithm parameters
        MSER( int _delta, int _min_area, int _max_area,
              float _max_variation, float _min_diversity,
              int _max_evolution, double _area_threshold,
              double _min_margin, int _edge_blur_size );
        // runs the extractor on the specified image; returns the MSERs,
        // each encoded as a contour (vector<Point>, see findContours)
        // the optional mask marks the area where MSERs are searched for
        void operator()( const Mat& image, vector<vector<Point> >& msers, const Mat& mask ) const;
    };

The class encapsulates all the parameters of the MSER extraction algorithm (see
http://en.wikipedia.org/wiki/Maximally_stable_extremal_regions). Also see http://opencv.willowgarage.com/wiki/documentation/cpp/features2d/MSER for useful comments and parameters description.


ORB
---
.. ocv:class:: ORB : public Feature2D

Class implementing the ORB (*oriented BRIEF*) keypoint detector and descriptor extractor, described in [RRKB11]_. The algorithm uses FAST in pyramids to detect stable keypoints, selects the strongest features using FAST or Harris response, finds their orientation using first-order moments and computes the descriptors using BRIEF (where the coordinates of random point pairs (or k-tuples) are rotated according to the measured orientation).

.. [RRKB11] Ethan Rublee, Vincent Rabaud, Kurt Konolige, Gary R. Bradski: ORB: An efficient alternative to SIFT or SURF. ICCV 2011: 2564-2571.

ORB::ORB
--------
The ORB constructor

.. ocv:function:: ORB::ORB(int nfeatures = 500, float scaleFactor = 1.2f, int nlevels = 8, int edgeThreshold = 31, int firstLevel = 0, int WTA_K=2, int scoreType=HARRIS_SCORE, int patchSize=31)

    :param nfeatures: The maximum number of features to retain.

    :param scaleFactor: Pyramid decimation ratio, greater than 1. ``scaleFactor==2`` means the classical pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor will mean that to cover certain scale range you will need more pyramid levels and so the speed will suffer.

    :param nlevels: The number of pyramid levels. The smallest level will have linear size equal to ``input_image_linear_size/pow(scaleFactor, nlevels)``.

    :param edgeThreshold: This is size of the border where the features are not detected. It should roughly match the ``patchSize`` parameter.

    :param firstLevel: It should be 0 in the current implementation.

    :param WTA_K: The number of points that produce each element of the oriented BRIEF descriptor. The default value 2 means the BRIEF where we take a random point pair and compare their brightnesses, so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3 random points (of course, those point coordinates are random, but they are generated from the pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such output will occupy 2 bits, and therefore it will need a special variant of Hamming distance, denoted as ``NORM_HAMMING2`` (2 bits per bin).  When ``WTA_K=4``, we take 4 random points to compute each bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).

    :param scoreType: The default HARRIS_SCORE means that Harris algorithm is used to rank features (the score is written to ``KeyPoint::score`` and is used to retain best ``nfeatures`` features); FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints, but it is a little faster to compute.

    :param patchSize: size of the patch used by the oriented BRIEF descriptor. Of course, on smaller pyramid layers the perceived image area covered by a feature will be larger.

ORB::operator()
---------------
Finds keypoints in an image and computes their descriptors

.. ocv:function:: void ORB::operator()(InputArray image, InputArray mask, vector<KeyPoint>& keypoints, OutputArray descriptors, bool useProvidedKeypoints=false ) const

    :param image: The input 8-bit grayscale image.

    :param mask: The operation mask.

    :param keypoints: The output vector of keypoints.

    :param descriptors: The output descriptors. Pass ``cv::noArray()`` if you do not need it.

    :param useProvidedKeypoints: If it is true, then the method will use the provided vector of keypoints instead of detecting them.

