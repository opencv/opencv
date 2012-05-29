Feature Detection and Description
=================================

SIFT
----
.. ocv:class:: SIFT : public Feature2D

Class for extracting keypoints and computing descriptors using the Scale Invariant Feature Transform (SIFT) algorithm by D. Lowe [Lowe04]_.

.. [Lowe04] Lowe, D. G., “Distinctive Image Features from Scale-Invariant Keypoints”, International Journal of Computer Vision, 60, 2, pp. 91-110, 2004.


SIFT::SIFT
----------
The SIFT constructors.

.. ocv:function:: SIFT::SIFT( int nfeatures=0, int nOctaveLayers=3, double contrastThreshold=0.04, double edgeThreshold=10, double sigma=1.6)

    :param nfeatures: The number of best features to retain. The features are ranked by their scores (measured in SIFT algorithm as the local contrast)

    :param nOctaveLayers: The number of layers in each octave. 3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.

    :param contrastThreshold: The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector.

    :param edgeThreshold: The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the ``edgeThreshold``, the less features are filtered out (more features are retained).

    :param sigma: The sigma of the Gaussian applied to the input image at the octave #0. If your image is captured with a weak camera with soft lenses, you might want to reduce the number.


SIFT::operator ()
-----------------
Extract features and computes their descriptors using SIFT algorithm

.. ocv:function:: void SIFT::operator()(InputArray img, InputArray mask, vector<KeyPoint>& keypoints, OutputArray descriptors, bool useProvidedKeypoints=false)

    :param img: Input 8-bit grayscale image

    :param mask: Optional input mask that marks the regions where we should detect features.

    :param keypoints: The input/output vector of keypoints

    :param descriptors: The output matrix of descriptors. Pass ``cv::noArray()`` if you do not need them.

    :param useProvidedKeypoints: Boolean flag. If it is true, the keypoint detector is not run. Instead, the provided vector of keypoints is used and the algorithm just computes their descriptors.


SURF
----
.. ocv:class:: SURF : public Feature2D

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

.. ocv:function:: SURF::SURF( double hessianThreshold, int nOctaves=4, int nOctaveLayers=2, bool extended=true, bool upright=false )

.. ocv:pyfunction:: cv2.SURF([hessianThreshold[, nOctaves[, nOctaveLayers[, extended[, upright]]]]]) -> <SURF object>

    :param hessianThreshold: Threshold for hessian keypoint detector used in SURF.

    :param nOctaves: Number of pyramid octaves the keypoint detector will use.

    :param nOctaveLayers: Number of octave layers within each octave.

    :param extended: Extended descriptor flag (true - use extended 128-element descriptors; false - use 64-element descriptors).

    :param upright: Up-right or rotated features flag (true - do not compute orientation of features; false - compute orientation).


SURF::operator()
----------------
Detects keypoints and computes SURF descriptors for them.

.. ocv:function:: void SURF::operator()(InputArray img, InputArray mask, vector<KeyPoint>& keypoints) const
.. ocv:function:: void SURF::operator()(InputArray img, InputArray mask, vector<KeyPoint>& keypoints, OutputArray descriptors, bool useProvidedKeypoints=false)

.. ocv:pyfunction:: cv2.SURF.detect(img, mask) -> keypoints
.. ocv:pyfunction:: cv2.SURF.detect(img, mask[, descriptors[, useProvidedKeypoints]]) -> keypoints, descriptors

.. ocv:cfunction:: void cvExtractSURF( const CvArr* image, const CvArr* mask, CvSeq** keypoints, CvSeq** descriptors, CvMemStorage* storage, CvSURFParams params )

.. ocv:pyoldfunction:: cv.ExtractSURF(image, mask, storage, params)-> (keypoints, descriptors)

    :param image: Input 8-bit grayscale image

    :param mask: Optional input mask that marks the regions where we should detect features.

    :param keypoints: The input/output vector of keypoints

    :param descriptors: The output matrix of descriptors. Pass ``cv::noArray()`` if you do not need them.

    :param useProvidedKeypoints: Boolean flag. If it is true, the keypoint detector is not run. Instead, the provided vector of keypoints is used and the algorithm just computes their descriptors.

    :param storage: Memory storage for the output keypoints and descriptors in OpenCV 1.x API.

    :param params: SURF algorithm parameters in OpenCV 1.x API.

The function is parallelized with the TBB library.
