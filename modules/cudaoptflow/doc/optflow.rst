Optical Flow
============

.. highlight:: cpp

.. note::

   * A general optical flow example can be found at opencv_source_code/samples/gpu/optical_flow.cpp
   * A general optical flow example using the Nvidia API can be found at opencv_source_code/samples/gpu/opticalflow_nvidia_api.cpp



cuda::BroxOpticalFlow
---------------------
.. ocv:class:: cuda::BroxOpticalFlow

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

.. note::

   * An example illustrating the Brox et al optical flow algorithm can be found at opencv_source_code/samples/gpu/brox_optical_flow.cpp



cuda::FarnebackOpticalFlow
--------------------------
.. ocv:class:: cuda::FarnebackOpticalFlow

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



cuda::FarnebackOpticalFlow::operator ()
---------------------------------------
Computes a dense optical flow using the Gunnar Farneback’s algorithm.

.. ocv:function:: void cuda::FarnebackOpticalFlow::operator ()(const GpuMat &frame0, const GpuMat &frame1, GpuMat &flowx, GpuMat &flowy, Stream &s = Stream::Null())

    :param frame0: First 8-bit gray-scale input image
    :param frame1: Second 8-bit gray-scale input image
    :param flowx: Flow horizontal component
    :param flowy: Flow vertical component
    :param s: Stream

.. seealso:: :ocv:func:`calcOpticalFlowFarneback`



cuda::FarnebackOpticalFlow::releaseMemory
-----------------------------------------
Releases unused auxiliary memory buffers.

.. ocv:function:: void cuda::FarnebackOpticalFlow::releaseMemory()



cuda::PyrLKOpticalFlow
----------------------
.. ocv:class:: cuda::PyrLKOpticalFlow

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
        bool useInitialFlow;

        void releaseMemory();
    };

The class can calculate an optical flow for a sparse feature set or dense optical flow using the iterative Lucas-Kanade method with pyramids.

.. seealso:: :ocv:func:`calcOpticalFlowPyrLK`

.. note::

   * An example of the Lucas Kanade optical flow algorithm can be found at opencv_source_code/samples/gpu/pyrlk_optical_flow.cpp



cuda::PyrLKOpticalFlow::sparse
------------------------------
Calculate an optical flow for a sparse feature set.

.. ocv:function:: void cuda::PyrLKOpticalFlow::sparse(const GpuMat& prevImg, const GpuMat& nextImg, const GpuMat& prevPts, GpuMat& nextPts, GpuMat& status, GpuMat* err = 0)

    :param prevImg: First 8-bit input image (supports both grayscale and color images).

    :param nextImg: Second input image of the same size and the same type as  ``prevImg`` .

    :param prevPts: Vector of 2D points for which the flow needs to be found. It must be one row matrix with CV_32FC2 type.

    :param nextPts: Output vector of 2D points (with single-precision floating-point coordinates) containing the calculated new positions of input features in the second image. When ``useInitialFlow`` is true, the vector must have the same size as in the input.

    :param status: Output status vector (CV_8UC1 type). Each element of the vector is set to 1 if the flow for the corresponding features has been found. Otherwise, it is set to 0.

    :param err: Output vector (CV_32FC1 type) that contains the difference between patches around the original and moved points or min eigen value if ``getMinEigenVals`` is checked. It can be NULL, if not needed.

.. seealso:: :ocv:func:`calcOpticalFlowPyrLK`



cuda::PyrLKOpticalFlow::dense
-----------------------------
Calculate dense optical flow.

.. ocv:function:: void cuda::PyrLKOpticalFlow::dense(const GpuMat& prevImg, const GpuMat& nextImg, GpuMat& u, GpuMat& v, GpuMat* err = 0)

    :param prevImg: First 8-bit grayscale input image.

    :param nextImg: Second input image of the same size and the same type as  ``prevImg`` .

    :param u: Horizontal component of the optical flow of the same size as input images, 32-bit floating-point, single-channel

    :param v: Vertical component of the optical flow of the same size as input images, 32-bit floating-point, single-channel

    :param err: Output vector (CV_32FC1 type) that contains the difference between patches around the original and moved points or min eigen value if ``getMinEigenVals`` is checked. It can be NULL, if not needed.



cuda::PyrLKOpticalFlow::releaseMemory
-------------------------------------
Releases inner buffers memory.

.. ocv:function:: void cuda::PyrLKOpticalFlow::releaseMemory()



cuda::interpolateFrames
-----------------------
Interpolates frames (images) using provided optical flow (displacement field).

.. ocv:function:: void cuda::interpolateFrames(const GpuMat& frame0, const GpuMat& frame1, const GpuMat& fu, const GpuMat& fv, const GpuMat& bu, const GpuMat& bv, float pos, GpuMat& newFrame, GpuMat& buf, Stream& stream = Stream::Null())

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
