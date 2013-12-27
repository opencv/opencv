Feature Detection And Description
=================================

.. highlight:: cpp

ocl::Canny
-------------------
Finds edges in an image using the [Canny86]_ algorithm.

.. ocv:function:: void ocl::Canny(const oclMat& image, oclMat& edges, double low_thresh, double high_thresh, int apperture_size = 3, bool L2gradient = false)

.. ocv:function:: void ocl::Canny(const oclMat& image, CannyBuf& buf, oclMat& edges, double low_thresh, double high_thresh, int apperture_size = 3, bool L2gradient = false)

.. ocv:function:: void ocl::Canny(const oclMat& dx, const oclMat& dy, oclMat& edges, double low_thresh, double high_thresh, bool L2gradient = false)

.. ocv:function:: void ocl::Canny(const oclMat& dx, const oclMat& dy, CannyBuf& buf, oclMat& edges, double low_thresh, double high_thresh, bool L2gradient = false)

    :param image: Single-channel 8-bit input image.

    :param dx: First derivative of image in the vertical direction. Support only ``CV_32S`` type.

    :param dy: First derivative of image in the horizontal direction. Support only ``CV_32S`` type.

    :param edges: Output edge map. It has the same size and type as  ``image`` .

    :param low_thresh: First threshold for the hysteresis procedure.

    :param high_thresh: Second threshold for the hysteresis procedure.

    :param apperture_size: Aperture size for the  :ocv:func:`Sobel`  operator.

    :param L2gradient: Flag indicating whether a more accurate  :math:`L_2`  norm  :math:`=\sqrt{(dI/dx)^2 + (dI/dy)^2}`  should be used to compute the image gradient magnitude ( ``L2gradient=true`` ), or a faster default  :math:`L_1`  norm  :math:`=|dI/dx|+|dI/dy|`  is enough ( ``L2gradient=false`` ).

    :param buf: Optional buffer to avoid extra memory allocations (for many calls with the same sizes).

.. seealso:: :ocv:func:`Canny`


ocl::BruteForceMatcher_OCL_base
-----------------------------------
.. ocv:class:: ocl::BruteForceMatcher_OCL_base

Brute-force descriptor matcher. For each descriptor in the first set, this matcher finds the closest descriptor in the second set by trying each one. This descriptor matcher supports masking permissible matches between descriptor sets. ::

    class BruteForceMatcher_OCL_base
    {
    public:
            enum DistType {L1Dist = 0, L2Dist, HammingDist};

        // Add descriptors to train descriptor collection.
        void add(const std::vector<oclMat>& descCollection);

        // Get train descriptors collection.
        const std::vector<oclMat>& getTrainDescriptors() const;

        // Clear train descriptors collection.
        void clear();

        // Return true if there are no train descriptors in collection.
        bool empty() const;

        // Return true if the matcher supports mask in match methods.
        bool isMaskSupported() const;

        void matchSingle(const oclMat& query, const oclMat& train,
            oclMat& trainIdx, oclMat& distance,
            const oclMat& mask = oclMat());

        static void matchDownload(const oclMat& trainIdx,
            const oclMat& distance, std::vector<DMatch>& matches);
        static void matchConvert(const Mat& trainIdx,
            const Mat& distance, std::vector<DMatch>& matches);

        void match(const oclMat& query, const oclMat& train,
            std::vector<DMatch>& matches, const oclMat& mask = oclMat());

        void makeGpuCollection(oclMat& trainCollection, oclMat& maskCollection,
            const vector<oclMat>& masks = std::vector<oclMat>());

        void matchCollection(const oclMat& query, const oclMat& trainCollection,
            oclMat& trainIdx, oclMat& imgIdx, oclMat& distance,
            const oclMat& maskCollection);

        static void matchDownload(const oclMat& trainIdx, oclMat& imgIdx,
            const oclMat& distance, std::vector<DMatch>& matches);
        static void matchConvert(const Mat& trainIdx, const Mat& imgIdx,
            const Mat& distance, std::vector<DMatch>& matches);

        void match(const oclMat& query, std::vector<DMatch>& matches,
            const std::vector<oclMat>& masks = std::vector<oclMat>());

        void knnMatchSingle(const oclMat& query, const oclMat& train,
            oclMat& trainIdx, oclMat& distance, oclMat& allDist, int k,
            const oclMat& mask = oclMat());

        static void knnMatchDownload(const oclMat& trainIdx, const oclMat& distance,
            std::vector< std::vector<DMatch> >& matches, bool compactResult = false);
        static void knnMatchConvert(const Mat& trainIdx, const Mat& distance,
            std::vector< std::vector<DMatch> >& matches, bool compactResult = false);

        void knnMatch(const oclMat& query, const oclMat& train,
            std::vector< std::vector<DMatch> >& matches, int k,
            const oclMat& mask = oclMat(), bool compactResult = false);

        void knnMatch2Collection(const oclMat& query, const oclMat& trainCollection,
            oclMat& trainIdx, oclMat& imgIdx, oclMat& distance,
            const oclMat& maskCollection = oclMat());

        static void knnMatch2Download(const oclMat& trainIdx, const oclMat& imgIdx, const oclMat& distance,
            std::vector< std::vector<DMatch> >& matches, bool compactResult = false);
        static void knnMatch2Convert(const Mat& trainIdx, const Mat& imgIdx, const Mat& distance,
            std::vector< std::vector<DMatch> >& matches, bool compactResult = false);

        void knnMatch(const oclMat& query, std::vector< std::vector<DMatch> >& matches, int k,
            const std::vector<oclMat>& masks = std::vector<oclMat>(),
            bool compactResult = false);

        void radiusMatchSingle(const oclMat& query, const oclMat& train,
            oclMat& trainIdx, oclMat& distance, oclMat& nMatches, float maxDistance,
            const oclMat& mask = oclMat());

        static void radiusMatchDownload(const oclMat& trainIdx, const oclMat& distance, const oclMat& nMatches,
            std::vector< std::vector<DMatch> >& matches, bool compactResult = false);
        static void radiusMatchConvert(const Mat& trainIdx, const Mat& distance, const Mat& nMatches,
            std::vector< std::vector<DMatch> >& matches, bool compactResult = false);

        void radiusMatch(const oclMat& query, const oclMat& train,
            std::vector< std::vector<DMatch> >& matches, float maxDistance,
            const oclMat& mask = oclMat(), bool compactResult = false);

        void radiusMatchCollection(const oclMat& query, oclMat& trainIdx, oclMat& imgIdx, oclMat& distance, oclMat& nMatches, float maxDistance,
            const std::vector<oclMat>& masks = std::vector<oclMat>());

        static void radiusMatchDownload(const oclMat& trainIdx, const oclMat& imgIdx, const oclMat& distance, const oclMat& nMatches,
            std::vector< std::vector<DMatch> >& matches, bool compactResult = false);
        static void radiusMatchConvert(const Mat& trainIdx, const Mat& imgIdx, const Mat& distance, const Mat& nMatches,
            std::vector< std::vector<DMatch> >& matches, bool compactResult = false);

        void radiusMatch(const oclMat& query, std::vector< std::vector<DMatch> >& matches, float maxDistance,
            const std::vector<oclMat>& masks = std::vector<oclMat>(), bool compactResult = false);

                DistType distType;

    private:
        std::vector<oclMat> trainDescCollection;
    };


The class ``BruteForceMatcher_OCL_base`` has an interface similar to the class :ocv:class:`DescriptorMatcher`. It has two groups of ``match`` methods: for matching descriptors of one image with another image or with an image set. Also, all functions have an alternative to save results either to the GPU memory or to the CPU memory. ``BruteForceMatcher_OCL_base`` supports only the ``L1<float>``, ``L2<float>``, and ``Hamming`` distance types.

.. seealso:: :ocv:class:`DescriptorMatcher`, :ocv:class:`BFMatcher`



ocl::BruteForceMatcher_OCL_base::match
------------------------------------------
Finds the best match for each descriptor from a query set with train descriptors.

.. ocv:function:: void ocl::BruteForceMatcher_OCL_base::match(const oclMat& query, const oclMat& train, std::vector<DMatch>& matches, const oclMat& mask = oclMat())

.. ocv:function:: void ocl::BruteForceMatcher_OCL_base::matchSingle(const oclMat& query, const oclMat& train, oclMat& trainIdx, oclMat& distance, const oclMat& mask = oclMat())

.. ocv:function:: void ocl::BruteForceMatcher_OCL_base::match(const oclMat& query, std::vector<DMatch>& matches, const std::vector<oclMat>& masks = std::vector<oclMat>())

.. ocv:function:: void ocl::BruteForceMatcher_OCL_base::matchCollection( const oclMat& query, const oclMat& trainCollection, oclMat& trainIdx, oclMat& imgIdx, oclMat& distance, const oclMat& masks=oclMat() )

.. seealso:: :ocv:func:`DescriptorMatcher::match`



ocl::BruteForceMatcher_OCL_base::makeGpuCollection
------------------------------------------------------
Performs a GPU collection of train descriptors and masks in a suitable format for the :ocv:func:`ocl::BruteForceMatcher_OCL_base::matchCollection` function.

.. ocv:function:: void ocl::BruteForceMatcher_OCL_base::makeGpuCollection(oclMat& trainCollection, oclMat& maskCollection, const vector<oclMat>& masks = std::vector<oclMat>())


ocl::BruteForceMatcher_OCL_base::matchDownload
--------------------------------------------------
Downloads matrices obtained via :ocv:func:`ocl::BruteForceMatcher_OCL_base::matchSingle` or :ocv:func:`ocl::BruteForceMatcher_OCL_base::matchCollection` to vector with :ocv:class:`DMatch`.

.. ocv:function:: static void ocl::BruteForceMatcher_OCL_base::matchDownload( const oclMat& trainIdx, const oclMat& distance, std::vector<DMatch>& matches )

.. ocv:function:: static void ocl::BruteForceMatcher_OCL_base::matchDownload( const oclMat& trainIdx, const oclMat& imgIdx, const oclMat& distance, std::vector<DMatch>& matches )


ocl::BruteForceMatcher_OCL_base::matchConvert
-------------------------------------------------
Converts matrices obtained via :ocv:func:`ocl::BruteForceMatcher_OCL_base::matchSingle` or :ocv:func:`ocl::BruteForceMatcher_OCL_base::matchCollection` to vector with :ocv:class:`DMatch`.

.. ocv:function:: void ocl::BruteForceMatcher_OCL_base::matchConvert(const Mat& trainIdx, const Mat& distance, std::vector<DMatch>&matches)

.. ocv:function:: void ocl::BruteForceMatcher_OCL_base::matchConvert(const Mat& trainIdx, const Mat& imgIdx, const Mat& distance, std::vector<DMatch>&matches)



ocl::BruteForceMatcher_OCL_base::knnMatch
---------------------------------------------
Finds the ``k`` best matches for each descriptor from a query set with train descriptors.

.. ocv:function:: void ocl::BruteForceMatcher_OCL_base::knnMatch(const oclMat& query, const oclMat& train, std::vector< std::vector<DMatch> >&matches, int k, const oclMat& mask = oclMat(), bool compactResult = false)

.. ocv:function:: void ocl::BruteForceMatcher_OCL_base::knnMatchSingle(const oclMat& query, const oclMat& train, oclMat& trainIdx, oclMat& distance, oclMat& allDist, int k, const oclMat& mask = oclMat())

.. ocv:function:: void ocl::BruteForceMatcher_OCL_base::knnMatch(const oclMat& query, std::vector< std::vector<DMatch> >&matches, int k, const std::vector<oclMat>&masks = std::vector<oclMat>(), bool compactResult = false )

.. ocv:function:: void ocl::BruteForceMatcher_OCL_base::knnMatch2Collection(const oclMat& query, const oclMat& trainCollection, oclMat& trainIdx, oclMat& imgIdx, oclMat& distance, const oclMat& maskCollection = oclMat())

    :param query: Query set of descriptors.

    :param train: Training set of descriptors. It is not be added to train descriptors collection stored in the class object.

    :param k: Number of the best matches per each query descriptor (or less if it is not possible).

    :param mask: Mask specifying permissible matches between the input query and train matrices of descriptors.

    :param compactResult: If ``compactResult`` is ``true`` , the ``matches`` vector does not contain matches for fully masked-out query descriptors.


The function returns detected ``k`` (or less if not possible) matches in the increasing order by distance.

The third variant of the method stores the results in GPU memory.

.. seealso:: :ocv:func:`DescriptorMatcher::knnMatch`



ocl::BruteForceMatcher_OCL_base::knnMatchDownload
-----------------------------------------------------
Downloads matrices obtained via :ocv:func:`ocl::BruteForceMatcher_OCL_base::knnMatchSingle` or :ocv:func:`ocl::BruteForceMatcher_OCL_base::knnMatch2Collection` to vector with :ocv:class:`DMatch`.

.. ocv:function:: void ocl::BruteForceMatcher_OCL_base::knnMatchDownload(const oclMat& trainIdx, const oclMat& distance, std::vector< std::vector<DMatch> >&matches, bool compactResult = false)

.. ocv:function:: void ocl::BruteForceMatcher_OCL_base::knnMatch2Download(const oclMat& trainIdx, const oclMat& imgIdx, const oclMat& distance, std::vector< std::vector<DMatch> >& matches, bool compactResult = false)

If ``compactResult`` is ``true`` , the ``matches`` vector does not contain matches for fully masked-out query descriptors.



ocl::BruteForceMatcher_OCL_base::knnMatchConvert
----------------------------------------------------
Converts matrices obtained via :ocv:func:`ocl::BruteForceMatcher_OCL_base::knnMatchSingle` or :ocv:func:`ocl::BruteForceMatcher_OCL_base::knnMatch2Collection` to CPU vector with :ocv:class:`DMatch`.

.. ocv:function:: void ocl::BruteForceMatcher_OCL_base::knnMatchConvert(const Mat& trainIdx, const Mat& distance, std::vector< std::vector<DMatch> >&matches, bool compactResult = false)

.. ocv:function:: void ocl::BruteForceMatcher_OCL_base::knnMatch2Convert(const Mat& trainIdx, const Mat& imgIdx, const Mat& distance, std::vector< std::vector<DMatch> >& matches, bool compactResult = false)

If ``compactResult`` is ``true`` , the ``matches`` vector does not contain matches for fully masked-out query descriptors.



ocl::BruteForceMatcher_OCL_base::radiusMatch
------------------------------------------------
For each query descriptor, finds the best matches with a distance less than a given threshold.

.. ocv:function:: void ocl::BruteForceMatcher_OCL_base::radiusMatch(const oclMat& query, const oclMat& train, std::vector< std::vector<DMatch> >&matches, float maxDistance, const oclMat& mask = oclMat(), bool compactResult = false)

.. ocv:function:: void ocl::BruteForceMatcher_OCL_base::radiusMatchSingle(const oclMat& query, const oclMat& train, oclMat& trainIdx, oclMat& distance, oclMat& nMatches, float maxDistance, const oclMat& mask = oclMat())

.. ocv:function:: void ocl::BruteForceMatcher_OCL_base::radiusMatch(const oclMat& query, std::vector< std::vector<DMatch> >&matches, float maxDistance, const std::vector<oclMat>& masks = std::vector<oclMat>(), bool compactResult = false)

.. ocv:function:: void ocl::BruteForceMatcher_OCL_base::radiusMatchCollection(const oclMat& query, oclMat& trainIdx, oclMat& imgIdx, oclMat& distance, oclMat& nMatches, float maxDistance, const std::vector<oclMat>& masks = std::vector<oclMat>())

    :param query: Query set of descriptors.

    :param train: Training set of descriptors. It is not added to train descriptors collection stored in the class object.

    :param maxDistance: Distance threshold.

    :param mask: Mask specifying permissible matches between the input query and train matrices of descriptors.

    :param compactResult: If ``compactResult`` is ``true`` , the ``matches`` vector does not contain matches for fully masked-out query descriptors.


The function returns detected matches in the increasing order by distance.

The methods work only on devices with the compute capability  :math:`>=` 1.1.

The third variant of the method stores the results in GPU memory and does not store the points by the distance.

.. seealso:: :ocv:func:`DescriptorMatcher::radiusMatch`



ocl::BruteForceMatcher_OCL_base::radiusMatchDownload
--------------------------------------------------------
Downloads matrices obtained via :ocv:func:`ocl::BruteForceMatcher_OCL_base::radiusMatchSingle` or :ocv:func:`ocl::BruteForceMatcher_OCL_base::radiusMatchCollection` to vector with :ocv:class:`DMatch`.

.. ocv:function:: void ocl::BruteForceMatcher_OCL_base::radiusMatchDownload(const oclMat& trainIdx, const oclMat& distance, const oclMat& nMatches, std::vector< std::vector<DMatch> >&matches, bool compactResult = false)

.. ocv:function:: void ocl::BruteForceMatcher_OCL_base::radiusMatchDownload(const oclMat& trainIdx, const oclMat& imgIdx, const oclMat& distance, const oclMat& nMatches, std::vector< std::vector<DMatch> >& matches, bool compactResult = false)

If ``compactResult`` is ``true`` , the ``matches`` vector does not contain matches for fully masked-out query descriptors.




ocl::BruteForceMatcher_OCL_base::radiusMatchConvert
-------------------------------------------------------
Converts matrices obtained via :ocv:func:`ocl::BruteForceMatcher_OCL_base::radiusMatchSingle` or :ocv:func:`ocl::BruteForceMatcher_OCL_base::radiusMatchCollection` to vector with :ocv:class:`DMatch`.

.. ocv:function:: void ocl::BruteForceMatcher_OCL_base::radiusMatchConvert(const Mat& trainIdx, const Mat& distance, const Mat& nMatches, std::vector< std::vector<DMatch> >&matches, bool compactResult = false)

.. ocv:function:: void ocl::BruteForceMatcher_OCL_base::radiusMatchConvert(const Mat& trainIdx, const Mat& imgIdx, const Mat& distance, const Mat& nMatches, std::vector< std::vector<DMatch> >& matches, bool compactResult = false)

If ``compactResult`` is ``true`` , the ``matches`` vector does not contain matches for fully masked-out query descriptors.


ocl::FAST_OCL
------------------
.. ocv:class:: ocl::FAST_OCL

Class used for corner detection using the FAST algorithm. ::

        class CV_EXPORTS FAST_OCL
        {
        public:
            enum
            {
                X_ROW = 0,
                Y_ROW,
                RESPONSE_ROW,
                ROWS_COUNT
            };

            // all features have same size
            static const int FEATURE_SIZE = 7;

            explicit FAST_OCL(int threshold, bool nonmaxSupression = true, double keypointsRatio = 0.05);

            //! finds the keypoints using FAST detector
            //! supports only CV_8UC1 images
            void operator ()(const oclMat& image, const oclMat& mask, oclMat& keypoints);
            void operator ()(const oclMat& image, const oclMat& mask, std::vector<KeyPoint>& keypoints);

            //! download keypoints from device to host memory
            static void downloadKeypoints(const oclMat& d_keypoints, std::vector<KeyPoint>& keypoints);

            //! convert keypoints to KeyPoint vector
            static void convertKeypoints(const Mat& h_keypoints, std::vector<KeyPoint>& keypoints);

            //! release temporary buffer's memory
            void release();

            bool nonmaxSupression;

            int threshold;

            //! max keypoints = keypointsRatio * img.size().area()
            double keypointsRatio;

            //! find keypoints and compute it's response if nonmaxSupression is true
            //! return count of detected keypoints
            int calcKeyPointsLocation(const oclMat& image, const oclMat& mask);

            //! get final array of keypoints
            //! performs nonmax supression if needed
            //! return final count of keypoints
            int getKeyPoints(oclMat& keypoints);

        private:
            // Hidden
        };


The class ``FAST_OCL`` implements FAST corner detection algorithm.

.. seealso:: :ocv:func:`FAST`



ocl::FAST_OCL::FAST_OCL
--------------------------
Constructor.

.. ocv:function:: ocl::FAST_OCL::FAST_OCL(int threshold, bool nonmaxSupression = true, double keypointsRatio = 0.05)

    :param threshold: Threshold on difference between intensity of the central pixel and pixels on a circle around this pixel.

    :param nonmaxSupression: If it is true, non-maximum suppression is applied to detected corners (keypoints).

    :param keypointsRatio: Inner buffer size for keypoints store is determined as (keypointsRatio * image_width * image_height).



ocl::FAST_OCL::operator ()
----------------------------
Finds the keypoints using FAST detector.

.. ocv:function:: void ocl::FAST_OCL::operator ()(const oclMat& image, const oclMat& mask, oclMat& keypoints)
.. ocv:function:: void ocl::FAST_OCL::operator ()(const oclMat& image, const oclMat& mask, std::vector<KeyPoint>& keypoints)

    :param image: Image where keypoints (corners) are detected. Only 8-bit grayscale images are supported.

    :param mask: Optional input mask that marks the regions where we should detect features.

    :param keypoints: The output vector of keypoints. Can be stored both in host or device memory. For device memory:

            * X_ROW of keypoints will contain the horizontal coordinate of the i'th point
            * Y_ROW of keypoints will contain the vertical coordinate of the i'th point
            * RESPONSE_ROW will contain response of i'th point (if non-maximum suppression is applied)



ocl::FAST_OCL::downloadKeypoints
----------------------------------
Download keypoints from device to host memory.

.. ocv:function:: void ocl::FAST_OCL::downloadKeypoints(const oclMat& d_keypoints, std::vector<KeyPoint>& keypoints)



ocl::FAST_OCL::convertKeypoints
---------------------------------
Converts keypoints from OpenCL representation to vector of ``KeyPoint``.

.. ocv:function:: void ocl::FAST_OCL::convertKeypoints(const Mat& h_keypoints, std::vector<KeyPoint>& keypoints)



ocl::FAST_OCL::release
------------------------
Releases inner buffer memory.

.. ocv:function:: void ocl::FAST_OCL::release()



ocl::FAST_OCL::calcKeyPointsLocation
--------------------------------------
Find keypoints. If ``nonmaxSupression`` is true, responses are computed and eliminates keypoints with the smaller responses from 9-neighborhood regions.

.. ocv:function:: int ocl::FAST_OCL::calcKeyPointsLocation(const oclMat& image, const oclMat& mask)

    :param image: Image where keypoints (corners) are detected. Only 8-bit grayscale images are supported.

    :param mask: Optional input mask that marks the regions where we should detect features.

The function returns the amount of detected keypoints.



ocl::FAST_OCL::getKeyPoints
-----------------------------
Gets final array of keypoints.

.. ocv:function:: int ocl::FAST_OCL::getKeyPoints(oclMat& keypoints)

    :param keypoints: The output vector of keypoints.

The function performs non-max suppression if needed and returns the final amount of keypoints.



ocl::HOGDescriptor
----------------------

.. ocv:struct:: ocl::HOGDescriptor

The class implements Histogram of Oriented Gradients ([Dalal2005]_) object detector. ::

    struct CV_EXPORTS HOGDescriptor
    {
        enum { DEFAULT_WIN_SIGMA = -1 };
        enum { DEFAULT_NLEVELS = 64 };
        enum { DESCR_FORMAT_ROW_BY_ROW, DESCR_FORMAT_COL_BY_COL };

        HOGDescriptor(Size win_size=Size(64, 128), Size block_size=Size(16, 16),
                      Size block_stride=Size(8, 8), Size cell_size=Size(8, 8),
                      int nbins=9, double win_sigma=DEFAULT_WIN_SIGMA,
                      double threshold_L2hys=0.2, bool gamma_correction=true,
                      int nlevels=DEFAULT_NLEVELS);

        size_t getDescriptorSize() const;
        size_t getBlockHistogramSize() const;

        void setSVMDetector(const vector<float>& detector);

        static vector<float> getDefaultPeopleDetector();
        static vector<float> getPeopleDetector48x96();
        static vector<float> getPeopleDetector64x128();

        void detect(const oclMat& img, vector<Point>& found_locations,
                    double hit_threshold=0, Size win_stride=Size(),
                    Size padding=Size());

        void detectMultiScale(const oclMat& img, vector<Rect>& found_locations,
                              double hit_threshold=0, Size win_stride=Size(),
                              Size padding=Size(), double scale0=1.05,
                              int group_threshold=2);

        void getDescriptors(const oclMat& img, Size win_stride,
                            oclMat& descriptors,
                            int descr_format=DESCR_FORMAT_COL_BY_COL);

        Size win_size;
        Size block_size;
        Size block_stride;
        Size cell_size;
        int nbins;
        double win_sigma;
        double threshold_L2hys;
        bool gamma_correction;
        int nlevels;

    private:
        // Hidden
    }


Interfaces of all methods are kept similar to the ``CPU HOG`` descriptor and detector analogues as much as possible.

.. note::

   (Ocl) An example using the HOG descriptor can be found at opencv_source_code/samples/ocl/hog.cpp

ocl::HOGDescriptor::HOGDescriptor
-------------------------------------
Creates the ``HOG`` descriptor and detector.

.. ocv:function:: ocl::HOGDescriptor::HOGDescriptor(Size win_size=Size(64, 128), Size block_size=Size(16, 16), Size block_stride=Size(8, 8), Size cell_size=Size(8, 8), int nbins=9, double win_sigma=DEFAULT_WIN_SIGMA, double threshold_L2hys=0.2, bool gamma_correction=true, int nlevels=DEFAULT_NLEVELS)

   :param win_size: Detection window size. Align to block size and block stride.

   :param block_size: Block size in pixels. Align to cell size. Only (16,16) is supported for now.

   :param block_stride: Block stride. It must be a multiple of cell size.

   :param cell_size: Cell size. Only (8, 8) is supported for now.

   :param nbins: Number of bins. Only 9 bins per cell are supported for now.

   :param win_sigma: Gaussian smoothing window parameter.

   :param threshold_L2hys: L2-Hys normalization method shrinkage.

   :param gamma_correction: Flag to specify whether the gamma correction preprocessing is required or not.

   :param nlevels: Maximum number of detection window increases.



ocl::HOGDescriptor::getDescriptorSize
-----------------------------------------
Returns the number of coefficients required for the classification.

.. ocv:function:: size_t ocl::HOGDescriptor::getDescriptorSize() const



ocl::HOGDescriptor::getBlockHistogramSize
---------------------------------------------
Returns the block histogram size.

.. ocv:function:: size_t ocl::HOGDescriptor::getBlockHistogramSize() const



ocl::HOGDescriptor::setSVMDetector
--------------------------------------
Sets coefficients for the linear SVM classifier.

.. ocv:function:: void ocl::HOGDescriptor::setSVMDetector(const vector<float>& detector)



ocl::HOGDescriptor::getDefaultPeopleDetector
------------------------------------------------
Returns coefficients of the classifier trained for people detection (for default window size).

.. ocv:function:: static vector<float> ocl::HOGDescriptor::getDefaultPeopleDetector()



ocl::HOGDescriptor::getPeopleDetector48x96
----------------------------------------------
Returns coefficients of the classifier trained for people detection (for 48x96 windows).

.. ocv:function:: static vector<float> ocl::HOGDescriptor::getPeopleDetector48x96()



ocl::HOGDescriptor::getPeopleDetector64x128
-----------------------------------------------
Returns coefficients of the classifier trained for people detection (for 64x128 windows).

.. ocv:function:: static vector<float> ocl::HOGDescriptor::getPeopleDetector64x128()



ocl::HOGDescriptor::detect
------------------------------
Performs object detection without a multi-scale window.

.. ocv:function:: void ocl::HOGDescriptor::detect(const oclMat& img, vector<Point>& found_locations, double hit_threshold=0, Size win_stride=Size(), Size padding=Size())

   :param img: Source image.  ``CV_8UC1``  and  ``CV_8UC4`` types are supported for now.

   :param found_locations: Left-top corner points of detected objects boundaries.

   :param hit_threshold: Threshold for the distance between features and SVM classifying plane. Usually it is 0 and should be specfied in the detector coefficients (as the last free coefficient). But if the free coefficient is omitted (which is allowed), you can specify it manually here.

   :param win_stride: Window stride. It must be a multiple of block stride.

   :param padding: Mock parameter to keep the CPU interface compatibility. It must be (0,0).



ocl::HOGDescriptor::detectMultiScale
----------------------------------------
Performs object detection with a multi-scale window.

.. ocv:function:: void ocl::HOGDescriptor::detectMultiScale(const oclMat& img, vector<Rect>& found_locations, double hit_threshold=0, Size win_stride=Size(), Size padding=Size(), double scale0=1.05, int group_threshold=2)

   :param img: Source image. See  :ocv:func:`ocl::HOGDescriptor::detect`  for type limitations.

   :param found_locations: Detected objects boundaries.

   :param hit_threshold: Threshold for the distance between features and SVM classifying plane. See  :ocv:func:`ocl::HOGDescriptor::detect`  for details.

   :param win_stride: Window stride. It must be a multiple of block stride.

   :param padding: Mock parameter to keep the CPU interface compatibility. It must be (0,0).

   :param scale0: Coefficient of the detection window increase.

   :param group_threshold: Coefficient to regulate the similarity threshold. When detected, some objects can be covered by many rectangles. 0 means not to perform grouping. See  :ocv:func:`groupRectangles` .



ocl::HOGDescriptor::getDescriptors
--------------------------------------
Returns block descriptors computed for the whole image.

.. ocv:function:: void ocl::HOGDescriptor::getDescriptors(const oclMat& img, Size win_stride, oclMat& descriptors, int descr_format=DESCR_FORMAT_COL_BY_COL)

   :param img: Source image. See  :ocv:func:`ocl::HOGDescriptor::detect`  for type limitations.

   :param win_stride: Window stride. It must be a multiple of block stride.

   :param descriptors: 2D array of descriptors.

   :param descr_format: Descriptor storage format:

        * **DESCR_FORMAT_ROW_BY_ROW** - Row-major order.

        * **DESCR_FORMAT_COL_BY_COL** - Column-major order.

The function is mainly used to learn the classifier.



ocl::ORB_OCL
--------------
.. ocv:class:: ocl::ORB_OCL

Class for extracting ORB features and descriptors from an image. ::

    class ORB_OCL
    {
    public:
        enum
        {
            X_ROW = 0,
            Y_ROW,
            RESPONSE_ROW,
            ANGLE_ROW,
            OCTAVE_ROW,
            SIZE_ROW,
            ROWS_COUNT
        };

        enum
        {
            DEFAULT_FAST_THRESHOLD = 20
        };

        explicit ORB_OCL(int nFeatures = 500, float scaleFactor = 1.2f,
                         int nLevels = 8, int edgeThreshold = 31,
                         int firstLevel = 0, int WTA_K = 2,
                         int scoreType = 0, int patchSize = 31);

        void operator()(const oclMat& image, const oclMat& mask,
                        std::vector<KeyPoint>& keypoints);
        void operator()(const oclMat& image, const oclMat& mask, oclMat& keypoints);

        void operator()(const oclMat& image, const oclMat& mask,
                        std::vector<KeyPoint>& keypoints, oclMat& descriptors);
        void operator()(const oclMat& image, const oclMat& mask,
                        oclMat& keypoints, oclMat& descriptors);

        void downloadKeyPoints(oclMat& d_keypoints, std::vector<KeyPoint>& keypoints);

        void convertKeyPoints(Mat& d_keypoints, std::vector<KeyPoint>& keypoints);

        int descriptorSize() const;
        int descriptorType() const;
        int defaultNorm() const;

        void setFastParams(int threshold, bool nonmaxSupression = true);

        void release();

        bool blurForDescriptor;
    };

The class implements ORB feature detection and description algorithm.



ocl::ORB_OCL::ORB_OCL
------------------------
Constructor.

.. ocv:function:: ocl::ORB_OCL::ORB_OCL(int nFeatures = 500, float scaleFactor = 1.2f, int nLevels = 8, int edgeThreshold = 31, int firstLevel = 0, int WTA_K = 2, int scoreType = 0, int patchSize = 31)

    :param nfeatures: The maximum number of features to retain.

    :param scaleFactor: Pyramid decimation ratio, greater than 1. ``scaleFactor==2`` means the classical pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor will mean that to cover certain scale range you will need more pyramid levels and so the speed will suffer.

    :param nlevels: The number of pyramid levels. The smallest level will have linear size equal to ``input_image_linear_size/pow(scaleFactor, nlevels)``.

    :param edgeThreshold: This is size of the border where the features are not detected. It should roughly match the ``patchSize`` parameter.

    :param firstLevel: It should be 0 in the current implementation.

    :param WTA_K: The number of points that produce each element of the oriented BRIEF descriptor. The default value 2 means the BRIEF where we take a random point pair and compare their brightnesses, so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3 random points (of course, those point coordinates are random, but they are generated from the pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such output will occupy 2 bits, and therefore it will need a special variant of Hamming distance, denoted as ``NORM_HAMMING2`` (2 bits per bin).  When ``WTA_K=4``, we take 4 random points to compute each bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).

    :param scoreType: The default HARRIS_SCORE means that Harris algorithm is used to rank features (the score is written to ``KeyPoint::score`` and is used to retain best ``nfeatures`` features); FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints, but it is a little faster to compute.

    :param patchSize: size of the patch used by the oriented BRIEF descriptor. Of course, on smaller pyramid layers the perceived image area covered by a feature will be larger.



ocl::ORB_OCL::operator()
--------------------------
Detects keypoints and computes descriptors for them.

.. ocv:function:: void ocl::ORB_OCL::operator()(const oclMat& image, const oclMat& mask, std::vector<KeyPoint>& keypoints)

.. ocv:function:: void ocl::ORB_OCL::operator()(const oclMat& image, const oclMat& mask, oclMat& keypoints)

.. ocv:function:: void ocl::ORB_OCL::operator()(const oclMat& image, const oclMat& mask, std::vector<KeyPoint>& keypoints, oclMat& descriptors)

.. ocv:function:: void ocl::ORB_OCL::operator()(const oclMat& image, const oclMat& mask, oclMat& keypoints, oclMat& descriptors)

    :param image: Input 8-bit grayscale image.

    :param mask: Optional input mask that marks the regions where we should detect features.

    :param keypoints: The input/output vector of keypoints. Can be stored both in host and device memory. For device memory:

            * ``X_ROW`` contains the horizontal coordinate of the i'th feature.
            * ``Y_ROW`` contains the vertical coordinate of the i'th feature.
            * ``RESPONSE_ROW`` contains the response of the i'th feature.
            * ``ANGLE_ROW`` contains the orientation of the i'th feature.
            * ``RESPONSE_ROW`` contains the octave of the i'th feature.
            * ``ANGLE_ROW`` contains the size of the i'th feature.

    :param descriptors: Computed descriptors. if ``blurForDescriptor`` is true, image will be blurred before descriptors calculation.



ocl::ORB_OCL::downloadKeyPoints
---------------------------------
Download keypoints from device to host memory.

.. ocv:function:: static void ocl::ORB_OCL::downloadKeyPoints( const oclMat& d_keypoints, std::vector<KeyPoint>& keypoints )



ocl::ORB_OCL::convertKeyPoints
--------------------------------
Converts keypoints from OCL representation to vector of ``KeyPoint``.

.. ocv:function:: static void ocl::ORB_OCL::convertKeyPoints( const Mat& d_keypoints, std::vector<KeyPoint>& keypoints )



ocl::ORB_OCL::release
-----------------------
Releases inner buffer memory.

.. ocv:function:: void ocl::ORB_OCL::release()
