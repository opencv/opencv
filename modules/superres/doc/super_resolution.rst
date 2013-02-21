Super Resolution
================

The Super Resolution module contains a set of functions and classes that can be used to solve the problem of resolution enhancement. There are a few methods implemented, most of them are descibed in the papers [Farsiu03]_ and [Mitzel09]_.



superes::SuperResolution
------------------------
Base class for Super Resolution algorithms.

.. ocv:class:: superres::SuperResolution : public Algorithm, public FrameSource

::

    class CV_EXPORTS SuperResolution : public Algorithm, public FrameSource
    {
    public:
        void setInput(const Ptr<FrameSource>& frameSource);

        void nextFrame(OutputArray frame);
        void reset();
    
        virtual void collectGarbage();
    
    protected:
        ...
    };


The class is only used to define the common interface for the whole family of Super Resolution algorithms.



superes::SuperResolution::setInput
----------------------------------
Set input frame source for Super Resolution algorithm.

.. ocv:function:: void superes::SuperResolution::setInput(const Ptr<FrameSource>& frameSource);



superes::SuperResolution::nextFrame
-----------------------------------
Process next frame from input and return output result.

.. ocv:function:: void superes::SuperResolution::nextFrame(OutputArray frame);



superes::SuperResolution::collectGarbage
----------------------------------------
Clear all inner buffers.

.. ocv:function:: void superes::SuperResolution::collectGarbage();



superes::createSuperResolution_BTVL1
------------------------------------
Create Bilateral TV-L1 Super Resolution.

.. ocv:function:: Ptr<SuperResolution> superes::createSuperResolution_BTVL1();

.. ocv:function:: Ptr<SuperResolution> superes::createSuperResolution_BTVL1_GPU();

This class implements Super Resolution algorithm described in the papers [Farsiu03]_ and [Mitzel09]_ .

Here are important members of the class that control the algorithm, which you can set after constructing the class instance:

    .. ocv:member:: int scale

        Scale factor.

    .. ocv:member:: int iterations

        Iteration count.

    .. ocv:member:: double tau

        Asymptotic value of steepest descent method.

    .. ocv:member:: double lambda

        Weight parameter to balance data term and smoothness term.

    .. ocv:member:: double alpha

        Parameter of spacial distribution in Bilateral-TV.

    .. ocv:member:: int btvKernelSize

        Kernel size of Bilateral-TV filter.

    .. ocv:member:: int blurKernelSize

        Gaussian blur kernel size.

    .. ocv:member:: double blurSigma

        Gaussian blur sigma.

    .. ocv:member:: int temporalAreaRadius

        Radius of the temporal search area.

    .. ocv:member:: Ptr<DenseOpticalFlow> opticalFlow

        Dense optical flow algorithm.



References
----------

.. [Farsiu03] S. Farsiu, D. Robinson, M. Elad, P. Milanfar. Fast and robust Super-Resolution. Proc 2003 IEEE Int Conf on Image Process, pp. 291–294, 2003.

.. [Mitzel09] D. Mitzel, T. Pock, T. Schoenemann, D. Cremers. Video super resolution using duality based TV-L1 optical flow. DAGM, 2009.
