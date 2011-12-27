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
