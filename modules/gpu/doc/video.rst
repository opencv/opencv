Video Analysis
==============

.. highlight:: cpp


gpu::BroxOpticalFlow
--------------------
.. ocv:class:: gpu::BroxOpticalFlow

Class computing the optical flow for two images using Brox et al Optical Flow algorithm ([Brox2004]_). ::

    class BroxOpticalFlow
    {
    public:
        BroxOpticalFlow(float alpha_, float gamma_, float scale_factor_, int inner_iterations_, int outer_iterations_, int solver_iterations_);

        //! Compute optical flow
        //! frame0 - source frame (supports only CV_32FC1 type)
        //! frame1 - frame to track (with the same size and type as frame0)
        //! u      - flow horizontal component (along x axis)
        //! v      - flow vertical component (along y axis)
        void operator ()(const GpuMat& frame0, const GpuMat& frame1, GpuMat& u, GpuMat& v, Stream& stream = Stream::Null());

        //! flow smoothness
        float alpha;

        //! gradient constancy importance
        float gamma;

        //! pyramid scale factor
        float scale_factor;

        //! number of lagged non-linearity iterations (inner loop)
        int inner_iterations;

        //! number of warping iterations (number of pyramid levels)
        int outer_iterations;

        //! number of linear system solver iterations
        int solver_iterations;

        GpuMat buf;
    };



gpu::GoodFeaturesToTrackDetector_GPU
------------------------------------

Class used for strong corners detection on an image. ::

    class GoodFeaturesToTrackDetector_GPU
    {
    public:
        explicit GoodFeaturesToTrackDetector_GPU(int maxCorners_ = 1000, double qualityLevel_ = 0.01, double minDistance_ = 0.0,
            int blockSize_ = 3, bool useHarrisDetector_ = false, double harrisK_ = 0.04);

        void operator ()(const GpuMat& image, GpuMat& corners, const GpuMat& mask = GpuMat());

        int maxCorners;
        double qualityLevel;
        double minDistance;

        int blockSize;
        bool useHarrisDetector;
        double harrisK;

        void releaseMemory();
    };

The class finds the most prominent corners in the image.

.. seealso:: :ocv:func:`goodFeaturesToTrack`



gpu::GoodFeaturesToTrackDetector_GPU::GoodFeaturesToTrackDetector_GPU
---------------------------------------------------------------------
Constructor.

.. ocv:function:: gpu::GoodFeaturesToTrackDetector_GPU::GoodFeaturesToTrackDetector_GPU(int maxCorners = 1000, double qualityLevel = 0.01, double minDistance = 0.0, int blockSize = 3, bool useHarrisDetector = false, double harrisK = 0.04)

    :param maxCorners: Maximum number of corners to return. If there are more corners than are found, the strongest of them is returned.

    :param qualityLevel: Parameter characterizing the minimal accepted quality of image corners. The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue (see  :ocv:func:`gpu::cornerMinEigenVal` ) or the Harris function response (see  :ocv:func:`gpu::cornerHarris` ). The corners with the quality measure less than the product are rejected. For example, if the best corner has the quality measure = 1500, and the  ``qualityLevel=0.01`` , then all the corners with the quality measure less than 15 are rejected.

    :param minDistance: Minimum possible Euclidean distance between the returned corners.

    :param blockSize: Size of an average block for computing a derivative covariation matrix over each pixel neighborhood. See  :ocv:func:`cornerEigenValsAndVecs` .
    
    :param useHarrisDetector: Parameter indicating whether to use a Harris detector (see :ocv:func:`gpu::cornerHarris`) or :ocv:func:`gpu::cornerMinEigenVal`.
    
    :param harrisK: Free parameter of the Harris detector.



gpu::GoodFeaturesToTrackDetector_GPU::operator ()
-------------------------------------------------
Finds the most prominent corners in the image.

.. ocv:function:: void gpu::GoodFeaturesToTrackDetector_GPU::operator ()(const GpuMat& image, GpuMat& corners, const GpuMat& mask = GpuMat()) 

    :param image: Input 8-bit, single-channel image.

    :param corners: Output vector of detected corners (it will be one row matrix with CV_32FC2 type).

    :param mask: Optional region of interest. If the image is not empty (it needs to have the type  ``CV_8UC1``  and the same size as  ``image`` ), it  specifies the region in which the corners are detected.

.. seealso:: :ocv:func:`goodFeaturesToTrack`



gpu::GoodFeaturesToTrackDetector_GPU::releaseMemory
---------------------------------------------------
Releases inner buffers memory.

.. ocv:function:: void gpu::GoodFeaturesToTrackDetector_GPU::releaseMemory()


gpu::FarnebackOpticalFlow
-------------------------
Class computing a dense optical flow using the Gunnar Farneback’s algorithm. ::

    class CV_EXPORTS FarnebackOpticalFlow
    {
    public:
        FarnebackOpticalFlow()
        {
            numLevels = 5;
            pyrScale = 0.5;
            fastPyramids = false;
            winSize = 13;
            numIters = 10;
            polyN = 5;
            polySigma = 1.1;
            flags = 0;
        }

        int numLevels;
        double pyrScale;
        bool fastPyramids;
        int winSize;
        int numIters;
        int polyN;
        double polySigma;
        int flags;

        void operator ()(const GpuMat &frame0, const GpuMat &frame1, GpuMat &flowx, GpuMat &flowy, Stream &s = Stream::Null());

        void releaseMemory();

    private:
        /* hidden */
    };


gpu::FarnebackOpticalFlow::operator ()
--------------------------------------
Computes a dense optical flow using the Gunnar Farneback’s algorithm.

.. ocv:function:: void gpu::FarnebackOpticalFlow::operator ()(const GpuMat &frame0, const GpuMat &frame1, GpuMat &flowx, GpuMat &flowy, Stream &s = Stream::Null())

    :param frame0: First 8-bit gray-scale input image
    :param frame1: Second 8-bit gray-scale input image
    :param flowx: Flow horizontal component
    :param flowy: Flow vertical component
    :param s: Stream

.. seealso:: :ocv:func:`calcOpticalFlowFarneback`


gpu::FarnebackOpticalFlow::releaseMemory
----------------------------------------
Releases unused auxiliary memory buffers.

.. ocv:function:: void gpu::FarnebackOpticalFlow::releaseMemory()


gpu::PyrLKOpticalFlow
---------------------

Class used for calculating an optical flow. ::

    class PyrLKOpticalFlow
    {
    public:
        PyrLKOpticalFlow();

        void sparse(const GpuMat& prevImg, const GpuMat& nextImg, const GpuMat& prevPts, GpuMat& nextPts,
            GpuMat& status, GpuMat* err = 0);

        void dense(const GpuMat& prevImg, const GpuMat& nextImg, GpuMat& u, GpuMat& v, GpuMat* err = 0);

        Size winSize;
        int maxLevel;
        int iters;
        double derivLambda;
        bool useInitialFlow;
        float minEigThreshold;

        void releaseMemory();
    };

The class can calculate an optical flow for a sparse feature set or dense optical flow using the iterative Lucas-Kanade method with pyramids.

.. seealso:: :ocv:func:`calcOpticalFlowPyrLK`



gpu::PyrLKOpticalFlow::sparse
-----------------------------
Calculate an optical flow for a sparse feature set.

.. ocv:function:: void gpu::PyrLKOpticalFlow::sparse(const GpuMat& prevImg, const GpuMat& nextImg, const GpuMat& prevPts, GpuMat& nextPts, GpuMat& status, GpuMat* err = 0)

    :param prevImg: First 8-bit input image (supports both grayscale and color images).

    :param nextImg: Second input image of the same size and the same type as  ``prevImg`` .

    :param prevPts: Vector of 2D points for which the flow needs to be found. It must be one row matrix with CV_32FC2 type.

    :param nextPts: Output vector of 2D points (with single-precision floating-point coordinates) containing the calculated new positions of input features in the second image. When ``useInitialFlow`` is true, the vector must have the same size as in the input.

    :param status: Output status vector (CV_8UC1 type). Each element of the vector is set to 1 if the flow for the corresponding features has been found. Otherwise, it is set to 0.

    :param err: Output vector (CV_32FC1 type) that contains min eigen value. It can be NULL, if not needed.

.. seealso:: :ocv:func:`calcOpticalFlowPyrLK`



gpu::PyrLKOpticalFlow::dense
-----------------------------
Calculate dense optical flow.

.. ocv:function:: void gpu::PyrLKOpticalFlow::dense(const GpuMat& prevImg, const GpuMat& nextImg, GpuMat& u, GpuMat& v, GpuMat* err = 0)

    :param prevImg: First 8-bit grayscale input image.

    :param nextImg: Second input image of the same size and the same type as  ``prevImg`` .

    :param u: Horizontal component of the optical flow of the same size as input images, 32-bit floating-point, single-channel 

    :param v: Vertical component of the optical flow of the same size as input images, 32-bit floating-point, single-channel 

    :param err: Output vector (CV_32FC1 type) that contains min eigen value. It can be NULL, if not needed.



gpu::PyrLKOpticalFlow::releaseMemory
------------------------------------
Releases inner buffers memory.

.. ocv:function:: void gpu::PyrLKOpticalFlow::releaseMemory()



gpu::interpolateFrames
----------------------
Interpolate frames (images) using provided optical flow (displacement field).

.. ocv:function:: void gpu::interpolateFrames(const GpuMat& frame0, const GpuMat& frame1, const GpuMat& fu, const GpuMat& fv, const GpuMat& bu, const GpuMat& bv, float pos, GpuMat& newFrame, GpuMat& buf, Stream& stream = Stream::Null())

    :param frame0: First frame (32-bit floating point images, single channel).

    :param frame1: Second frame. Must have the same type and size as ``frame0`` .

    :param fu: Forward horizontal displacement.

    :param fv: Forward vertical displacement.

    :param bu: Backward horizontal displacement.

    :param bv: Backward vertical displacement.

    :param pos: New frame position.

    :param newFrame: Output image.

    :param buf: Temporary buffer, will have width x 6*height size, CV_32FC1 type and contain 6 GpuMat: occlusion masks for first frame, occlusion masks for second, interpolated forward horizontal flow, interpolated forward vertical flow, interpolated backward horizontal flow, interpolated backward vertical flow.

    :param stream: Stream for the asynchronous version.



.. [Brox2004] T. Brox, A. Bruhn, N. Papenberg, J. Weickert. *High accuracy optical flow estimation based on a theory for warping*. ECCV 2004.
