Super Resolution
================

.. highlight:: cpp

The Super Resolution module contains a set of functions and classes that can be used to solve the problem of resolution enhancement. There are a few methods implemented, most of them are descibed in the papers [Farsiu03]_ and [Mitzel09]_.



superres::SuperResolution
-------------------------
Base class for Super Resolution algorithms.

.. ocv:class:: superres::SuperResolution : public Algorithm, public superres::FrameSource

The class is only used to define the common interface for the whole family of Super Resolution algorithms.



superres::SuperResolution::setInput
-----------------------------------
Set input frame source for Super Resolution algorithm.

.. ocv:function:: void superres::SuperResolution::setInput(const Ptr<FrameSource>& frameSource)

    :param frameSource: Input frame source



superres::SuperResolution::nextFrame
------------------------------------
Process next frame from input and return output result.

.. ocv:function:: void superres::SuperResolution::nextFrame(OutputArray frame)

    :param frame: Output result



superres::SuperResolution::collectGarbage
-----------------------------------------
Clear all inner buffers.

.. ocv:function:: void superres::SuperResolution::collectGarbage()



superres::createSuperResolution_BTVL1
-------------------------------------
Create Bilateral TV-L1 Super Resolution.

.. ocv:function:: Ptr<SuperResolution> superres::createSuperResolution_BTVL1()

.. ocv:function:: Ptr<SuperResolution> superres::createSuperResolution_BTVL1_GPU()

This class implements Super Resolution algorithm described in the papers [Farsiu03]_ and [Mitzel09]_ .

Here are important members of the class that control the algorithm, which you can set after constructing the class instance:

    * **int scale** Scale factor.

    * **int iterations** Iteration count.

    * **double tau** Asymptotic value of steepest descent method.

    * **double lambda** Weight parameter to balance data term and smoothness term.

    * **double alpha** Parameter of spacial distribution in Bilateral-TV.

    * **int btvKernelSize** Kernel size of Bilateral-TV filter.

    * **int blurKernelSize** Gaussian blur kernel size.

    * **double blurSigma** Gaussian blur sigma.

    * **int temporalAreaRadius** Radius of the temporal search area.

    * **Ptr<DenseOpticalFlowExt> opticalFlow** Dense optical flow algorithm.



.. [Farsiu03] S. Farsiu, D. Robinson, M. Elad, P. Milanfar. Fast and robust Super-Resolution. Proc 2003 IEEE Int Conf on Image Process, pp. 291–294, 2003.

.. [Mitzel09] D. Mitzel, T. Pock, T. Schoenemann, D. Cremers. Video super resolution using duality based TV-L1 optical flow. DAGM, 2009.
