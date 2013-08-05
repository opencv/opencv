HDR imaging
==========

.. highlight:: cpp

This section describes high dynamic range imaging algorithms, namely tonemapping, exposure alignment, camera calibration with multiple exposures and exposure fusion.

Tonemap
-------------
.. ocv:class:: Tonemap : public Algorithm

The base class for tonemapping algorithms - tools, that are used to map HDR image to 8-bit range.

Tonemap::process
-----------------------
Tonemaps image

.. ocv:function:: void Tonemap::process(InputArray src, OutputArray dst)
    :param src: source image - 32-bit 3-channel Mat
    :param dst: destination image - 32-bit 3-channel Mat with values in [0, 1] range

createTonemap
------------------
Creates simple linear mapper with gamma correction

.. ocv:function:: Ptr<Tonemap> createTonemap(float gamma = 1.0f);

    :param gamma: gamma value for gamma correction
    
TonemapDrago
--------
.. ocv:class:: TonemapDrago : public Tonemap

"Adaptive Logarithmic Mapping For Displaying HighContrast Scenes", Drago et al., 2003

createTonemapDrago
------------------
Creates TonemapDrago object

.. ocv:function:: Ptr<TonemapDrago> createTonemapDrago(float gamma = 1.0f, float bias = 0.85f);

    :param gamma: gamma value for gamma correction
    
    :param saturation:  saturation enhancement value
    
    :param bias: value for bias function in [0, 1] range
    
TonemapDurand
--------
.. ocv:class:: TonemapDurand : public Tonemap

"Fast Bilateral Filtering for the Display of High-Dynamic-Range Images", Durand, Dorsey, 2002

This implementation uses regular bilateral filter from opencv.

createTonemapDurand
------------------
Creates TonemapDurand object

.. ocv:function:: Ptr<TonemapDurand> createTonemapDurand(float gamma = 1.0f, float contrast = 4.0f, float saturation = 1.0f, float sigma_space = 2.0f, float sigma_color = 2.0f);

    :param gamma: gamma value for gamma correction
    
    :param contrast: resulting contrast on logarithmic scale
    
    :param saturation:  saturation enhancement value
    
    :param sigma_space: filter sigma in color space
    
    :param sigma_color: filter sigma in coordinate space
    
TonemapReinhardDevlin
--------
.. ocv:class:: TonemapReinhardDevlin : public Tonemap

"Dynamic Range Reduction Inspired by Photoreceptor Physiology", Reinhard, Devlin, 2005

createTonemapReinhardDevlin
------------------
Creates TonemapReinhardDevlin object

.. ocv:function:: Ptr<TonemapReinhardDevlin> createTonemapReinhardDevlin(float gamma = 1.0f, float intensity = 0.0f, float light_adapt = 1.0f, float color_adapt = 0.0f)

    :param gamma: gamma value for gamma correction
    
    :param intensity: result intensity. Range in [-8, 8] range
    
    :param light_adapt:  light adaptation in [0, 1] range. If 1 adaptation is based on pixel value, if 0 it's global
    
    :param color_adapt: chromatic adaptation in [0, 1] range. If 1 channels are treated independently, if 0 adaptation level is the same for each channel
    
TonemapMantiuk
--------
.. ocv:class:: TonemapMantiuk : public Tonemap

"Perceptual Framework for Contrast Processing of High Dynamic Range Images", Mantiuk et al., 2006

createTonemapMantiuk
------------------
Creates TonemapMantiuk object

.. ocv:function:: CV_EXPORTS_W Ptr<TonemapMantiuk> createTonemapMantiuk(float gamma = 1.0f, float scale = 0.7f, float saturation = 1.0f);

    :param gamma: gamma value for gamma correction
    
    :param scale: contrast scale factor
    
    :param saturation:  saturation enhancement value
    
ExposureAlign
-------------
.. ocv:class:: ExposureAlign : public Algorithm

The base class for algorithms that align images of the same scene with different exposures

ExposureAlign::process
-----------------------
Aligns images

.. ocv:function:: void ExposureAlign::process(InputArrayOfArrays src, OutputArrayOfArrays dst, const std::vector<float>& times, InputArray response)

    :param src: vector of input images
    
    :param dst: vector of aligned images
    
    :param times: vector of exposure time values for each image
    
    :param response: matrix with camera response, one column per channel
    
AlignMTB
--------
.. ocv:class:: AlignMTB : public ExposureAlign

"Fast, Robust Image Registration for Compositing High Dynamic Range Photographs from Handheld Exposures", Ward, 2003

This algorithm does not use exposure values and camera response, new image regions are filled with zeros.

AlignMTB::process
-----------------------
Short version of process, that doesn't take extra arguments.

.. ocv:function:: void AlignMTB::process(InputArrayOfArrays src, OutputArrayOfArrays dst)

    :param src: vector of input images
   
    :param dst: vector of aligned images

AlignMTB::calculateShift
-----------------------
Calculates shift between two images.

.. ocv:function:: void AlignMTB::calculateShift(InputArray img0, InputArray img1, Point& shift)

    :param img0: first image
    
    :param img1: second image
    
    :param shift: how to shift the second image to correspond it with the first

AlignMTB::shiftMat
-----------------------
Gelper function, that shift Mat filling new regions with zeros.
    
.. ocv:function:: void AlignMTB::shiftMat(InputArray src, OutputArray dst, const Point shift)

    :param src: input image
    
    :param dst: result image
    
    :param shift: shift value
    
createAlignMTB
------------------
Creates AlignMTB object

.. ocv:function:: Ptr<AlignMTB> createAlignMTB(int max_bits = 6, int exclude_range = 4)
    
    :param max_bits: logarithm to the base 2 of maximal shift in each dimension
    
    :param exclude_range: range for exclusion bitmap
    
ExposureCalibrate
-------------
.. ocv:class:: ExposureCalibrate : public Algorithm

The base class for camera response calibration algorithms.

ExposureCalibrate::process
-----------------------
Recovers camera response.

.. ocv:function:: void ExposureCalibrate::process(InputArrayOfArrays src, OutputArray dst, std::vector<float>& times)

    :param src: vector of input images
    
    :param dst: matrix with calculated camera response, one column per channel
    
    :param times: vector of exposure time values for each image
    
CalibrateDebevec
--------
.. ocv:class:: CalibrateDebevec : public ExposureCalibrate

"Recovering High Dynamic Range Radiance Maps from Photographs", Debevec, Malik, 1997

createCalibrateDebevec
------------------
Creates CalibrateDebevec object

.. ocv:function:: Ptr<CalibrateDebevec> createCalibrateDebevec(int samples = 50, float lambda = 10.0f)

    :param samples: number of pixel locations to use
    
    :param lambda: smoothness term weight
    
ExposureMerge
-------------
.. ocv:class:: ExposureMerge : public Algorithm

The base class algorithms that can merge exposure sequence to a single image.

ExposureMerge::process
-----------------------
Merges images.

.. ocv:function:: void process(InputArrayOfArrays src, OutputArray dst, const std::vector<float>& times, InputArray response)

    :param src: vector of input images
    
    :param dst: result image
    
    :param times: vector of exposure time values for each image
    
    :param response: matrix with camera response, one column per channel
    
MergeDebevec
--------
.. ocv:class:: MergeDebevec : public ExposureMerge

"Recovering High Dynamic Range Radiance Maps from Photographs", Debevec, Malik, 1997

createMergeDebevec
------------------
Creates MergeDebevec object

.. ocv:function:: Ptr<MergeDebevec> createMergeDebevec();

MergeMertens
--------
.. ocv:class:: MergeMertens : public ExposureMerge

"Exposure Fusion", Mertens et al., 2007

The resulting image doesn't require tonemapping and can be converted to 8-bit image by multiplying by 255.

MergeMertens::process
-----------------------
Short version of process, that doesn't take extra arguments.

.. ocv:function:: void MergeMertens::process(InputArrayOfArrays src, OutputArray dst)

    :param src: vector of input images
   
    :param dst: result image


createMergeMertens
------------------
Creates MergeMertens object

.. ocv:function:: Ptr<MergeMertens> createMergeMertens(float contrast_weight = 1.0f, float saturation_weight = 1.0f, float exposure_weight = 0.0f)

    :param contrast_weight: contrast factor weight
    
    :param saturation_weight: saturation factor weight
    
    :param exposure_weight: well-exposedness factor weight