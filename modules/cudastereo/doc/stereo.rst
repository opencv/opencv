Stereo Correspondence
=====================

.. highlight:: cpp

.. note::

   * A basic stereo matching example can be found at opencv_source_code/samples/gpu/stereo_match.cpp
   * A stereo matching example using several GPU's can be found at opencv_source_code/samples/gpu/stereo_multi.cpp
   * A stereo matching example using several GPU's and driver API can be found at opencv_source_code/samples/gpu/driver_api_stereo_multi.cpp



cuda::StereoBM
--------------
.. ocv:class:: cuda::StereoBM : public cv::StereoBM

Class computing stereo correspondence (disparity map) using the block matching algorithm. ::

.. seealso:: :ocv:class:`StereoBM`



cuda::createStereoBM
--------------------
Creates StereoBM object.

.. ocv:function:: Ptr<cuda::StereoBM> cuda::createStereoBM(int numDisparities = 64, int blockSize = 19)

    :param numDisparities: the disparity search range. For each pixel algorithm will find the best disparity from 0 (default minimum disparity) to ``numDisparities``. The search range can then be shifted by changing the minimum disparity.

    :param blockSize: the linear size of the blocks compared by the algorithm. The size should be odd (as the block is centered at the current pixel). Larger block size implies smoother, though less accurate disparity map. Smaller block size gives more detailed disparity map, but there is higher chance for algorithm to find a wrong correspondence.



cuda::StereoBeliefPropagation
-----------------------------
.. ocv:class:: cuda::StereoBeliefPropagation : public cv::StereoMatcher

Class computing stereo correspondence using the belief propagation algorithm. ::

    class CV_EXPORTS StereoBeliefPropagation : public cv::StereoMatcher
    {
    public:
        using cv::StereoMatcher::compute;

        virtual void compute(InputArray left, InputArray right, OutputArray disparity, Stream& stream) = 0;

        //! version for user specified data term
        virtual void compute(InputArray data, OutputArray disparity, Stream& stream = Stream::Null()) = 0;

        //! number of BP iterations on each level
        virtual int getNumIters() const = 0;
        virtual void setNumIters(int iters) = 0;

        //! number of levels
        virtual int getNumLevels() const = 0;
        virtual void setNumLevels(int levels) = 0;

        //! truncation of data cost
        virtual double getMaxDataTerm() const = 0;
        virtual void setMaxDataTerm(double max_data_term) = 0;

        //! data weight
        virtual double getDataWeight() const = 0;
        virtual void setDataWeight(double data_weight) = 0;

        //! truncation of discontinuity cost
        virtual double getMaxDiscTerm() const = 0;
        virtual void setMaxDiscTerm(double max_disc_term) = 0;

        //! discontinuity single jump
        virtual double getDiscSingleJump() const = 0;
        virtual void setDiscSingleJump(double disc_single_jump) = 0;

        virtual int getMsgType() const = 0;
        virtual void setMsgType(int msg_type) = 0;

        static void estimateRecommendedParams(int width, int height, int& ndisp, int& iters, int& levels);
    };


The class implements algorithm described in [Felzenszwalb2006]_ . It can compute own data cost (using a truncated linear model) or use a user-provided data cost.

.. note::

    ``StereoBeliefPropagation`` requires a lot of memory for message storage:

    .. math::

        width \_ step  \cdot height  \cdot ndisp  \cdot 4  \cdot (1 + 0.25)

    and for data cost storage:

    .. math::

        width\_step \cdot height \cdot ndisp \cdot (1 + 0.25 + 0.0625 +  \dotsm + \frac{1}{4^{levels}})

    ``width_step`` is the number of bytes in a line including padding.

``StereoBeliefPropagation`` uses a truncated linear model for the data cost and discontinuity terms:

.. math::

    DataCost = data \_ weight  \cdot \min ( \lvert Img_Left(x,y)-Img_Right(x-d,y)  \rvert , max \_ data \_ term)

.. math::

    DiscTerm =  \min (disc \_ single \_ jump  \cdot \lvert f_1-f_2  \rvert , max \_ disc \_ term)

For more details, see [Felzenszwalb2006]_.

By default, ``StereoBeliefPropagation`` uses floating-point arithmetics and the ``CV_32FC1`` type for messages. But it can also use fixed-point arithmetics and the ``CV_16SC1`` message type for better performance. To avoid an overflow in this case, the parameters must satisfy the following requirement:

.. math::

    10  \cdot 2^{levels-1}  \cdot max \_ data \_ term < SHRT \_ MAX

.. seealso:: :ocv:class:`StereoMatcher`



cuda::createStereoBeliefPropagation
-----------------------------------
Creates StereoBeliefPropagation object.

.. ocv:function:: Ptr<cuda::StereoBeliefPropagation> cuda::createStereoBeliefPropagation(int ndisp = 64, int iters = 5, int levels = 5, int msg_type = CV_32F)

    :param ndisp: Number of disparities.

    :param iters: Number of BP iterations on each level.

    :param levels: Number of levels.

    :param msg_type: Type for messages.  ``CV_16SC1``  and  ``CV_32FC1`` types are supported.



cuda::StereoBeliefPropagation::estimateRecommendedParams
--------------------------------------------------------
Uses a heuristic method to compute the recommended parameters ( ``ndisp``, ``iters`` and ``levels`` ) for the specified image size ( ``width`` and ``height`` ).

.. ocv:function:: void cuda::StereoBeliefPropagation::estimateRecommendedParams(int width, int height, int& ndisp, int& iters, int& levels)



cuda::StereoBeliefPropagation::compute
--------------------------------------
Enables the stereo correspondence operator that finds the disparity for the specified data cost.

.. ocv:function:: void cuda::StereoBeliefPropagation::compute(InputArray data, OutputArray disparity, Stream& stream = Stream::Null())

    :param data: User-specified data cost, a matrix of ``msg_type`` type and ``Size(<image columns>*ndisp, <image rows>)`` size.

    :param disparity: Output disparity map. If  ``disparity``  is empty, the output type is  ``CV_16SC1`` . Otherwise, the type is retained.

    :param stream: Stream for the asynchronous version.



cuda::StereoConstantSpaceBP
---------------------------
.. ocv:class:: cuda::StereoConstantSpaceBP : public cuda::StereoBeliefPropagation

Class computing stereo correspondence using the constant space belief propagation algorithm. ::

    class CV_EXPORTS StereoConstantSpaceBP : public cuda::StereoBeliefPropagation
    {
    public:
        //! number of active disparity on the first level
        virtual int getNrPlane() const = 0;
        virtual void setNrPlane(int nr_plane) = 0;

        virtual bool getUseLocalInitDataCost() const = 0;
        virtual void setUseLocalInitDataCost(bool use_local_init_data_cost) = 0;

        static void estimateRecommendedParams(int width, int height, int& ndisp, int& iters, int& levels, int& nr_plane);
    };


The class implements algorithm described in [Yang2010]_. ``StereoConstantSpaceBP`` supports both local minimum and global minimum data cost initialization algorithms. For more details, see the paper mentioned above. By default, a local algorithm is used. To enable a global algorithm, set ``use_local_init_data_cost`` to ``false`` .

``StereoConstantSpaceBP`` uses a truncated linear model for the data cost and discontinuity terms:

.. math::

    DataCost = data \_ weight  \cdot \min ( \lvert I_2-I_1  \rvert , max \_ data \_ term)

.. math::

    DiscTerm =  \min (disc \_ single \_ jump  \cdot \lvert f_1-f_2  \rvert , max \_ disc \_ term)

For more details, see [Yang2010]_.

By default, ``StereoConstantSpaceBP`` uses floating-point arithmetics and the ``CV_32FC1`` type for messages. But it can also use fixed-point arithmetics and the ``CV_16SC1`` message type for better performance. To avoid an overflow in this case, the parameters must satisfy the following requirement:

.. math::

    10  \cdot 2^{levels-1}  \cdot max \_ data \_ term < SHRT \_ MAX



cuda::createStereoConstantSpaceBP
---------------------------------
Creates StereoConstantSpaceBP object.

.. ocv:function:: Ptr<cuda::StereoConstantSpaceBP> cuda::createStereoConstantSpaceBP(int ndisp = 128, int iters = 8, int levels = 4, int nr_plane = 4, int msg_type = CV_32F)

    :param ndisp: Number of disparities.

    :param iters: Number of BP iterations on each level.

    :param levels: Number of levels.

    :param nr_plane: Number of disparity levels on the first level.

    :param msg_type: Type for messages.  ``CV_16SC1``  and  ``CV_32FC1`` types are supported.



cuda::StereoConstantSpaceBP::estimateRecommendedParams
------------------------------------------------------
Uses a heuristic method to compute parameters (ndisp, iters, levelsand nrplane) for the specified image size (widthand height).

.. ocv:function:: void cuda::StereoConstantSpaceBP::estimateRecommendedParams(int width, int height, int& ndisp, int& iters, int& levels, int& nr_plane)



cuda::DisparityBilateralFilter
------------------------------
.. ocv:class:: cuda::DisparityBilateralFilter : public cv::Algorithm

Class refining a disparity map using joint bilateral filtering. ::

    class CV_EXPORTS DisparityBilateralFilter : public cv::Algorithm
    {
    public:
        //! the disparity map refinement operator. Refine disparity map using joint bilateral filtering given a single color image.
        //! disparity must have CV_8U or CV_16S type, image must have CV_8UC1 or CV_8UC3 type.
        virtual void apply(InputArray disparity, InputArray image, OutputArray dst, Stream& stream = Stream::Null()) = 0;

        virtual int getNumDisparities() const = 0;
        virtual void setNumDisparities(int numDisparities) = 0;

        virtual int getRadius() const = 0;
        virtual void setRadius(int radius) = 0;

        virtual int getNumIters() const = 0;
        virtual void setNumIters(int iters) = 0;

        //! truncation of data continuity
        virtual double getEdgeThreshold() const = 0;
        virtual void setEdgeThreshold(double edge_threshold) = 0;

        //! truncation of disparity continuity
        virtual double getMaxDiscThreshold() const = 0;
        virtual void setMaxDiscThreshold(double max_disc_threshold) = 0;

        //! filter range sigma
        virtual double getSigmaRange() const = 0;
        virtual void setSigmaRange(double sigma_range) = 0;
    };


The class implements [Yang2010]_ algorithm.



cuda::createDisparityBilateralFilter
------------------------------------
Creates DisparityBilateralFilter object.

.. ocv:function:: Ptr<cuda::DisparityBilateralFilter> cuda::createDisparityBilateralFilter(int ndisp = 64, int radius = 3, int iters = 1)

    :param ndisp: Number of disparities.

    :param radius: Filter radius.

    :param iters: Number of iterations.



cuda::DisparityBilateralFilter::apply
-------------------------------------
Refines a disparity map using joint bilateral filtering.

.. ocv:function:: void cuda::DisparityBilateralFilter::apply(InputArray disparity, InputArray image, OutputArray dst, Stream& stream = Stream::Null())

    :param disparity: Input disparity map.  ``CV_8UC1``  and  ``CV_16SC1``  types are supported.

    :param image: Input image. ``CV_8UC1``  and  ``CV_8UC3``  types are supported.

    :param dst: Destination disparity map. It has the same size and type as  ``disparity`` .

    :param stream: Stream for the asynchronous version.



cuda::reprojectImageTo3D
------------------------
Reprojects a disparity image to 3D space.

.. ocv:function:: void cuda::reprojectImageTo3D(InputArray disp, OutputArray xyzw, InputArray Q, int dst_cn = 4, Stream& stream = Stream::Null())

    :param disp: Input disparity image.  ``CV_8U``  and  ``CV_16S``  types are supported.

    :param xyzw: Output 3- or 4-channel floating-point image of the same size as  ``disp`` . Each element of  ``xyzw(x,y)``  contains 3D coordinates ``(x,y,z)`` or ``(x,y,z,1)``  of the point  ``(x,y)`` , computed from the disparity map.

    :param Q: :math:`4 \times 4`  perspective transformation matrix that can be obtained via  :ocv:func:`stereoRectify` .

    :param dst_cn: The number of channels for output image. Can be 3 or 4.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`reprojectImageTo3D`



cuda::drawColorDisp
-------------------
Colors a disparity image.

.. ocv:function:: void cuda::drawColorDisp(InputArray src_disp, OutputArray dst_disp, int ndisp, Stream& stream = Stream::Null())

    :param src_disp: Source disparity image.  ``CV_8UC1``  and  ``CV_16SC1``  types are supported.

    :param dst_disp: Output disparity image. It has the same size as  ``src_disp`` . The  type is ``CV_8UC4``  in  ``BGRA``  format (alpha = 255).

    :param ndisp: Number of disparities.

    :param stream: Stream for the asynchronous version.

This function draws a colored disparity map by converting disparity values from ``[0..ndisp)`` interval first to ``HSV`` color space (where different disparity values correspond to different hues) and then converting the pixels to ``RGB`` for visualization.



.. [Felzenszwalb2006] Pedro F. Felzenszwalb algorithm [Pedro F. Felzenszwalb and Daniel P. Huttenlocher. *Efficient belief propagation for early vision*. International Journal of Computer Vision, 70(1), October 2006
.. [Yang2010] Q. Yang, L. Wang, and N. Ahuja. *A constant-space belief propagation algorithm for stereo matching*. In CVPR, 2010.
