Background Segmentation
=======================

.. highlight:: cpp



gpu::BackgroundSubtractorMOG
----------------------------
Gaussian Mixture-based Background/Foreground Segmentation Algorithm.

.. ocv:class:: gpu::BackgroundSubtractorMOG : public cv::BackgroundSubtractorMOG

The class discriminates between foreground and background pixels by building and maintaining a model of the background. Any pixel which does not fit this model is then deemed to be foreground. The class implements algorithm described in [MOG2001]_.

.. seealso:: :ocv:class:`BackgroundSubtractorMOG`

.. note::

   * An example on gaussian mixture based background/foreground segmantation can be found at opencv_source_code/samples/gpu/bgfg_segm.cpp


gpu::createBackgroundSubtractorMOG
----------------------------------
Creates mixture-of-gaussian background subtractor

.. ocv:function:: Ptr<gpu::BackgroundSubtractorMOG> gpu::createBackgroundSubtractorMOG(int history=200, int nmixtures=5, double backgroundRatio=0.7, double noiseSigma=0)

    :param history: Length of the history.

    :param nmixtures: Number of Gaussian mixtures.

    :param backgroundRatio: Background ratio.

    :param noiseSigma: Noise strength (standard deviation of the brightness or each color channel). 0 means some automatic value.



gpu::BackgroundSubtractorMOG2
-----------------------------
Gaussian Mixture-based Background/Foreground Segmentation Algorithm.

.. ocv:class:: gpu::BackgroundSubtractorMOG2 : public cv::BackgroundSubtractorMOG2

The class discriminates between foreground and background pixels by building and maintaining a model of the background. Any pixel which does not fit this model is then deemed to be foreground. The class implements algorithm described in [MOG2004]_.

.. seealso:: :ocv:class:`BackgroundSubtractorMOG2`



gpu::createBackgroundSubtractorMOG2
-----------------------------------
Creates MOG2 Background Subtractor

.. ocv:function:: Ptr<gpu::BackgroundSubtractorMOG2> gpu::createBackgroundSubtractorMOG2( int history=500, double varThreshold=16, bool detectShadows=true )

  :param history: Length of the history.

  :param varThreshold: Threshold on the squared Mahalanobis distance between the pixel and the model to decide whether a pixel is well described by the background model. This parameter does not affect the background update.

  :param detectShadows: If true, the algorithm will detect shadows and mark them. It decreases the speed a bit, so if you do not need this feature, set the parameter to false.



gpu::BackgroundSubtractorGMG
----------------------------
Background/Foreground Segmentation Algorithm.

.. ocv:class:: gpu::BackgroundSubtractorGMG : public cv::BackgroundSubtractorGMG

The class discriminates between foreground and background pixels by building and maintaining a model of the background. Any pixel which does not fit this model is then deemed to be foreground. The class implements algorithm described in [GMG2012]_.



gpu::createBackgroundSubtractorGMG
----------------------------------
Creates GMG Background Subtractor

.. ocv:function:: Ptr<gpu::BackgroundSubtractorGMG> gpu::createBackgroundSubtractorGMG(int initializationFrames = 120, double decisionThreshold = 0.8)

    :param initializationFrames: Number of frames of video to use to initialize histograms.

    :param decisionThreshold: Value above which pixel is determined to be FG.



gpu::BackgroundSubtractorFGD
----------------------------

.. ocv:class:: gpu::BackgroundSubtractorFGD : public cv::BackgroundSubtractor

The class discriminates between foreground and background pixels by building and maintaining a model of the background. Any pixel which does not fit this model is then deemed to be foreground. The class implements algorithm described in [FGD2003]_. ::

    class CV_EXPORTS BackgroundSubtractorFGD : public cv::BackgroundSubtractor
    {
    public:
        virtual void getForegroundRegions(OutputArrayOfArrays foreground_regions) = 0;
    };

.. seealso:: :ocv:class:`BackgroundSubtractor`



gpu::BackgroundSubtractorFGD::getForegroundRegions
--------------------------------------------------
Returns the output foreground regions calculated by :ocv:func:`findContours`.

.. ocv:function:: void gpu::BackgroundSubtractorFGD::getForegroundRegions(OutputArrayOfArrays foreground_regions)

    :params foreground_regions: Output array (CPU memory).



gpu::createBackgroundSubtractorFGD
----------------------------------
Creates FGD Background Subtractor

.. ocv:function:: Ptr<gpu::BackgroundSubtractorGMG> gpu::createBackgroundSubtractorFGD(const FGDParams& params = FGDParams())

    :param params: Algorithm's parameters. See [FGD2003]_ for explanation.



.. [FGD2003] Liyuan Li, Weimin Huang, Irene Y.H. Gu, and Qi Tian. *Foreground Object Detection from Videos Containing Complex Background*. ACM MM2003 9p, 2003.
.. [MOG2001] P. KadewTraKuPong and R. Bowden. *An improved adaptive background mixture model for real-time tracking with shadow detection*. Proc. 2nd European Workshop on Advanced Video-Based Surveillance Systems, 2001
.. [MOG2004] Z. Zivkovic. *Improved adaptive Gausian mixture model for background subtraction*. International Conference Pattern Recognition, UK, August, 2004
.. [GMG2012] A. Godbehere, A. Matsukawa and K. Goldberg. *Visual Tracking of Human Visitors under Variable-Lighting Conditions for a Responsive Audio Art Installation*. American Control Conference, Montreal, June 2012
