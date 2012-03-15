Feature Detection and Description
=================================

SIFT
----
.. ocv:class:: SIFT

Class for extracting keypoints and computing descriptors using the Scale Invariant Feature Transform (SIFT) approach. ::

    class CV_EXPORTS SIFT
    {
    public:
        struct CommonParams
        {
            static const int DEFAULT_NOCTAVES = 4;
            static const int DEFAULT_NOCTAVE_LAYERS = 3;
            static const int DEFAULT_FIRST_OCTAVE = -1;
            enum{ FIRST_ANGLE = 0, AVERAGE_ANGLE = 1 };

            CommonParams();
            CommonParams( int _nOctaves, int _nOctaveLayers, int _firstOctave,
                                              int _angleMode );
            int nOctaves, nOctaveLayers, firstOctave;
            int angleMode;
        };

        struct DetectorParams
        {
            static double GET_DEFAULT_THRESHOLD()
              { return 0.04 / SIFT::CommonParams::DEFAULT_NOCTAVE_LAYERS / 2.0; }
            static double GET_DEFAULT_EDGE_THRESHOLD() { return 10.0; }

            DetectorParams();
            DetectorParams( double _threshold, double _edgeThreshold );
            double threshold, edgeThreshold;
        };

        struct DescriptorParams
        {
            static double GET_DEFAULT_MAGNIFICATION() { return 3.0; }
            static const bool DEFAULT_IS_NORMALIZE = true;
            static const int DESCRIPTOR_SIZE = 128;

            DescriptorParams();
            DescriptorParams( double _magnification, bool _isNormalize,
                                                      bool _recalculateAngles );
            double magnification;
            bool isNormalize;
            bool recalculateAngles;
        };

        SIFT();
        //! sift-detector constructor
        SIFT( double _threshold, double _edgeThreshold,
              int _nOctaves=CommonParams::DEFAULT_NOCTAVES,
              int _nOctaveLayers=CommonParams::DEFAULT_NOCTAVE_LAYERS,
              int _firstOctave=CommonParams::DEFAULT_FIRST_OCTAVE,
              int _angleMode=CommonParams::FIRST_ANGLE );
        //! sift-descriptor constructor
        SIFT( double _magnification, bool _isNormalize=true,
              bool _recalculateAngles = true,
              int _nOctaves=CommonParams::DEFAULT_NOCTAVES,
              int _nOctaveLayers=CommonParams::DEFAULT_NOCTAVE_LAYERS,
              int _firstOctave=CommonParams::DEFAULT_FIRST_OCTAVE,
              int _angleMode=CommonParams::FIRST_ANGLE );
        SIFT( const CommonParams& _commParams,
              const DetectorParams& _detectorParams = DetectorParams(),
              const DescriptorParams& _descriptorParams = DescriptorParams() );

        //! returns the descriptor size in floats (128)
        int descriptorSize() const { return DescriptorParams::DESCRIPTOR_SIZE; }
        //! finds the keypoints using the SIFT algorithm
        void operator()(const Mat& img, const Mat& mask,
                        vector<KeyPoint>& keypoints) const;
        //! finds the keypoints and computes descriptors for them using SIFT algorithm.
        //! Optionally it can compute descriptors for the user-provided keypoints
        void operator()(const Mat& img, const Mat& mask,
                        vector<KeyPoint>& keypoints,
                        Mat& descriptors,
                        bool useProvidedKeypoints=false) const;

        CommonParams getCommonParams () const { return commParams; }
        DetectorParams getDetectorParams () const { return detectorParams; }
        DescriptorParams getDescriptorParams () const { return descriptorParams; }
    protected:
        ...
    };




SURF
----
.. ocv:class:: SURF

Class for extracting Speeded Up Robust Features from an image [Bay06]_. The class is derived from ``CvSURFParams`` structure, which specifies the algorithm parameters:

    .. ocv:member:: int extended
    
        * 0 means that the basic descriptors (64 elements each) shall be computed
        * 1 means that the extended descriptors (128 elements each) shall be computed
       
    .. ocv:member:: int upright
    
        * 0 means that detector computes orientation of each feature.
        * 1 means that the orientation is not computed (which is much, much faster). For example, if you match images from a stereo pair, or do image stitching, the matched features likely have very similar angles, and you can speed up feature extraction by setting ``upright=1``.
        
    .. ocv:member:: double hessianThreshold
    
        Threshold for the keypoint detector. Only features, whose hessian is larger than ``hessianThreshold`` are retained by the detector. Therefore, the larger the value, the less keypoints you will get. A good default value could be from 300 to 500, depending from the image contrast.
        
    .. ocv:member:: int nOctaves
    
        The number of a gaussian pyramid octaves that the detector uses. It is set to 4 by default. If you want to get very large features, use the larger value. If you want just small features, decrease it.
        
    .. ocv:member:: int nOctaveLayers
    
        The number of images within each octave of a gaussian pyramid. It is set to 2 by default.


.. [Bay06] Bay, H. and Tuytelaars, T. and Van Gool, L. "SURF: Speeded Up Robust Features", 9th European Conference on Computer Vision, 2006


SURF::SURF
----------
The SURF extractor constructors.

.. ocv:function:: SURF::SURF()

.. ocv:function:: SURF::SURF(double hessianThreshold, int nOctaves=4, int nOctaveLayers=2, bool extended=false, bool upright=false)

.. ocv:pyfunction:: cv2.SURF(_hessianThreshold[, _nOctaves[, _nOctaveLayers[, _extended[, _upright]]]]) -> <SURF object>

    :param hessianThreshold: Threshold for hessian keypoint detector used in SURF.
    
    :param nOctaves: Number of pyramid octaves the keypoint detector will use.
    
    :param nOctaveLayers: Number of octave layers within each octave.
    
    :param extended: Extended descriptor flag (true - use extended 128-element descriptors; false - use 64-element descriptors).
    
    :param upright: Up-right or rotated features flag (true - do not compute orientation of features; false - compute orientation).


SURF::operator()
----------------
Detects keypoints and computes SURF descriptors for them.

.. ocv:function:: void SURF::operator()(const Mat& image, const Mat& mask, vector<KeyPoint>& keypoints)
.. ocv:function:: void SURF::operator()(const Mat& image, const Mat& mask, vector<KeyPoint>& keypoints, vector<float>& descriptors, bool useProvidedKeypoints=false)

.. ocv:pyfunction:: cv2.SURF.detect(img, mask) -> keypoints
.. ocv:pyfunction:: cv2.SURF.detect(img, mask[, useProvidedKeypoints]) -> keypoints, descriptors

.. ocv:cfunction:: void cvExtractSURF( const CvArr* image, const CvArr* mask, CvSeq** keypoints, CvSeq** descriptors, CvMemStorage* storage, CvSURFParams params )

.. ocv:pyoldfunction:: cv.ExtractSURF(image, mask, storage, params)-> (keypoints, descriptors)

    :param image: Input 8-bit grayscale image
    
    :param mask: Optional input mask that marks the regions where we should detect features.
    
    :param keypoints: The input/output vector of keypoints
    
    :param descriptors: The output concatenated vectors of descriptors. Each descriptor is 64- or 128-element vector, as returned by ``SURF::descriptorSize()``. So the total size of ``descriptors`` will be ``keypoints.size()*descriptorSize()``.
    
    :param useProvidedKeypoints: Boolean flag. If it is true, the keypoint detector is not run. Instead, the provided vector of keypoints is used and the algorithm just computes their descriptors.
    
    :param storage: Memory storage for the output keypoints and descriptors in OpenCV 1.x API.
    
    :param params: SURF algorithm parameters in OpenCV 1.x API.

