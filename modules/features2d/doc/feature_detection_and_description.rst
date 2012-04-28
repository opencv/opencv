Feature Detection and Description
=================================

.. highlight:: cpp

FAST
--------
Detects corners using the FAST algorithm

.. ocv:function:: void FAST( const Mat& image, vector<KeyPoint>& keypoints,            int threshold, bool nonmaxSupression=true )

    :param image: Image where keypoints (corners) are detected.

    :param keypoints: Keypoints detected on the image.

    :param threshold: Threshold on difference between intensity of the central pixel and pixels on a circle around this pixel. See the algorithm description below.

    :param nonmaxSupression: If it is true, non-maximum suppression is applied to detected corners (keypoints).

Detects corners using the FAST algorithm by E. Rosten (*Machine Learning for High-speed Corner Detection*, 2006).


MSER
----
.. ocv:class:: MSER

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


StarDetector
------------
.. ocv:class:: StarDetector

Class implementing the ``Star`` keypoint detector, a modified version of the ``CenSurE`` keypoint detector described in [Agrawal08]_.

.. [Agrawal08] Agrawal, M. and Konolige, K. and Blas, M.R. "CenSurE: Center Surround Extremas for Realtime Feature Detection and Matching", ECCV08, 2008

StarDetector::StarDetector
--------------------------
The Star Detector constructor

.. ocv:function:: StarDetector::StarDetector()

.. ocv:function:: StarDetector::StarDetector(int maxSize, int responseThreshold, int lineThresholdProjected, int lineThresholdBinarized, int suppressNonmaxSize)

.. ocv:pyfunction:: cv2.StarDetector(maxSize, responseThreshold, lineThresholdProjected, lineThresholdBinarized, suppressNonmaxSize) -> <StarDetector object>

    :param maxSize: maximum size of the features. The following values are supported: 4, 6, 8, 11, 12, 16, 22, 23, 32, 45, 46, 64, 90, 128. In the case of a different value the result is undefined.
    
    :param responseThreshold: threshold for the approximated laplacian, used to eliminate weak features. The larger it is, the less features will be retrieved
    
    :param lineThresholdProjected: another threshold for the laplacian to eliminate edges    

    :param lineThresholdBinarized: yet another threshold for the feature size to eliminate edges. The larger the 2nd threshold, the more points you get.

StarDetector::operator()
------------------------
Finds keypoints in an image
        
.. ocv:function:: void StarDetector::operator()(const Mat& image, vector<KeyPoint>& keypoints)

.. ocv:pyfunction:: cv2.StarDetector.detect(image) -> keypoints

.. ocv:cfunction:: CvSeq* cvGetStarKeypoints( const CvArr* image, CvMemStorage* storage, CvStarDetectorParams params=cvStarDetectorParams() )

.. ocv:pyoldfunction:: cv.GetStarKeypoints(image, storage, params)-> keypoints

    :param image: The input 8-bit grayscale image
    
    :param keypoints: The output vector of keypoints
    
    :param storage: The memory storage used to store the keypoints (OpenCV 1.x API only)
    
    :param params: The algorithm parameters stored in ``CvStarDetectorParams`` (OpenCV 1.x API only)

ORB
----
.. ocv:class:: ORB

Class for extracting ORB features and descriptors from an image. ::

    class ORB
    {
    public:
        /** The patch sizes that can be used (only one right now) */
        struct CommonParams
        {
            enum { DEFAULT_N_LEVELS = 3, DEFAULT_FIRST_LEVEL = 0};

            /** default constructor */
            CommonParams(float scale_factor = 1.2f, unsigned int n_levels = DEFAULT_N_LEVELS,
                 int edge_threshold = 31, unsigned int first_level = DEFAULT_FIRST_LEVEL);
            void read(const FileNode& fn);
            void write(FileStorage& fs) const;

            /** Coefficient by which we divide the dimensions from one scale pyramid level to the next */
            float scale_factor_;
            /** The number of levels in the scale pyramid */
            unsigned int n_levels_;
            /** The level at which the image is given
             * if 1, that means we will also look at the image scale_factor_ times bigger
             */
            unsigned int first_level_;
            /** How far from the boundary the points should be */
            int edge_threshold_;
        };

        // constructor that initializes all the algorithm parameters
        // n_features is the number of desired features
        ORB(size_t n_features = 500, const CommonParams & detector_params = CommonParams());
        // returns the number of elements in each descriptor (32 bytes)
        int descriptorSize() const;
        // detects keypoints using ORB
        void operator()(const Mat& img, const Mat& mask,
                        vector<KeyPoint>& keypoints) const;
        // detects ORB keypoints and computes the ORB descriptors for them;
        // output vector "descriptors" stores elements of descriptors and has size
        // equal descriptorSize()*keypoints.size() as each descriptor is
        // descriptorSize() elements of this vector.
        void operator()(const Mat& img, const Mat& mask,
                        vector<KeyPoint>& keypoints,
                        cv::Mat& descriptors,
                        bool useProvidedKeypoints=false) const;
    };

The class implements ORB.

..