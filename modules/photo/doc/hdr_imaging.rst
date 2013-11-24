HDR imaging
=============

.. highlight:: cpp

This section describes high dynamic range imaging algorithms namely tonemapping, exposure alignment, camera calibration with multiple exposures and exposure fusion.

Tonemap
---------------------------
.. ocv:class:: Tonemap : public Algorithm

Base class for tonemapping algorithms - tools that are used to map HDR image to 8-bit range.

Tonemap::process
---------------------------
Tonemaps image

.. ocv:function:: void Tonemap::process(InputArray src, OutputArray dst)

    :param src: source image - 32-bit 3-channel Mat
    :param dst: destination image - 32-bit 3-channel Mat with values in [0, 1] range

createTonemap
---------------------------
Creates simple linear mapper with gamma correction

.. ocv:function:: Ptr<Tonemap> createTonemap(float gamma = 1.0f)

    :param gamma: positive value for gamma correction. Gamma value of 1.0 implies no correction, gamma equal to 2.2f is suitable for most displays.

                  Generally gamma > 1 brightens the image and gamma < 1 darkens it.

TonemapDrago
---------------------------
.. ocv:class:: TonemapDrago : public Tonemap

Adaptive logarithmic mapping is a fast global tonemapping algorithm that scales the image in logarithmic domain.

Since it's a global operator the same function is applied to all the pixels, it is controlled by the bias parameter.

Optional saturation enhancement is possible as described in [FL02]_.

For more information see [DM03]_.

createTonemapDrago
---------------------------
Creates TonemapDrago object

.. ocv:function:: Ptr<TonemapDrago> createTonemapDrago(float gamma = 1.0f, float saturation = 1.0f, float bias = 0.85f)

    :param gamma: gamma value for gamma correction. See :ocv:func:`createTonemap`

    :param saturation: positive saturation enhancement value. 1.0 preserves saturation, values greater than 1 increase saturation and values less than 1 decrease it.

    :param bias: value for bias function in [0, 1] range. Values from 0.7 to 0.9 usually give best results, default value is 0.85.

TonemapDurand
---------------------------
.. ocv:class:: TonemapDurand : public Tonemap

This algorithm decomposes image into two layers: base layer and detail layer using bilateral filter and compresses contrast of the base layer thus preserving all the details.

This implementation uses regular bilateral filter from opencv.

Saturation enhancement is possible as in ocv:class:`TonemapDrago`.

For more information see [DD02]_.

createTonemapDurand
---------------------------
Creates TonemapDurand object

.. ocv:function:: Ptr<TonemapDurand> createTonemapDurand(float gamma = 1.0f, float contrast = 4.0f, float saturation = 1.0f, float sigma_space = 2.0f, float sigma_color = 2.0f)

    :param gamma: gamma value for gamma correction. See :ocv:func:`createTonemap`

    :param contrast: resulting contrast on logarithmic scale, i. e. log(max / min), where max and min are maximum and minimum luminance values of the resulting image.

    :param saturation:  saturation enhancement value. See :ocv:func:`createTonemapDrago`

    :param sigma_space: bilateral filter sigma in color space

    :param sigma_color: bilateral filter sigma in coordinate space

TonemapReinhard
---------------------------
.. ocv:class:: TonemapReinhard : public Tonemap

This is a global tonemapping operator that models human visual system.

Mapping function is controlled by adaptation parameter, that is computed using light adaptation and color adaptation.

For more information see [RD05]_.

createTonemapReinhard
---------------------------
Creates TonemapReinhard object

.. ocv:function:: Ptr<TonemapReinhard> createTonemapReinhard(float gamma = 1.0f, float intensity = 0.0f, float light_adapt = 1.0f, float color_adapt = 0.0f)

    :param gamma: gamma value for gamma correction. See :ocv:func:`createTonemap`

    :param intensity: result intensity in [-8, 8] range. Greater intensity produces brighter results.

    :param light_adapt:  light adaptation in [0, 1] range. If 1 adaptation is based only on pixel value, if 0 it's global, otherwise it's a weighted mean of this two cases.

    :param color_adapt: chromatic adaptation in [0, 1] range. If 1 channels are treated independently, if 0 adaptation level is the same for each channel.

TonemapMantiuk
---------------------------
.. ocv:class:: TonemapMantiuk : public Tonemap

This algorithm transforms image to contrast using gradients on all levels of gaussian pyramid, transforms contrast values to HVS response and scales the response.
After this the image is reconstructed from new contrast values.

For more information see [MM06]_.

createTonemapMantiuk
---------------------------
Creates TonemapMantiuk object

.. ocv:function:: Ptr<TonemapMantiuk> createTonemapMantiuk(float gamma = 1.0f, float scale = 0.7f, float saturation = 1.0f)

    :param gamma: gamma value for gamma correction. See :ocv:func:`createTonemap`

    :param scale: contrast scale factor. HVS response is multiplied by this parameter, thus compressing dynamic range. Values from 0.6 to 0.9 produce best results.

    :param saturation: saturation enhancement value. See :ocv:func:`createTonemapDrago`

AlignExposures
---------------------------
.. ocv:class:: AlignExposures : public Algorithm

The base class for algorithms that align images of the same scene with different exposures

AlignExposures::process
---------------------------
Aligns images

.. ocv:function:: void AlignExposures::process(InputArrayOfArrays src, std::vector<Mat>& dst, InputArray times, InputArray response)

    :param src: vector of input images

    :param dst: vector of aligned images

    :param times: vector of exposure time values for each image

    :param response: 256x1 matrix with inverse camera response function for each pixel value, it should have the same number of channels as images.

AlignMTB
---------------------------
.. ocv:class:: AlignMTB : public AlignExposures

This algorithm converts images to median threshold bitmaps (1 for pixels brighter than median luminance and 0 otherwise) and than aligns the resulting bitmaps using bit operations.

It is invariant to exposure, so exposure values and camera response are not necessary.

In this implementation new image regions are filled with zeros.

For more information see [GW03]_.

AlignMTB::process
---------------------------
Short version of process, that doesn't take extra arguments.

.. ocv:function:: void AlignMTB::process(InputArrayOfArrays src, std::vector<Mat>& dst)

    :param src: vector of input images

    :param dst: vector of aligned images

AlignMTB::calculateShift
---------------------------
Calculates shift between two images, i. e. how to shift the second image to correspond it with the first.

.. ocv:function:: Point AlignMTB::calculateShift(InputArray img0, InputArray img1)

    :param img0: first image

    :param img1: second image

AlignMTB::shiftMat
---------------------------
Helper function, that shift Mat filling new regions with zeros.

.. ocv:function:: void AlignMTB::shiftMat(InputArray src, OutputArray dst, const Point shift)

    :param src: input image

    :param dst: result image

    :param shift: shift value

AlignMTB::computeBitmaps
---------------------------
Computes median threshold and exclude bitmaps of given image.

.. ocv:function:: void AlignMTB::computeBitmaps(InputArray img, OutputArray tb, OutputArray eb)

    :param img: input image

    :param tb: median threshold bitmap

    :param eb: exclude bitmap

createAlignMTB
---------------------------
Creates AlignMTB object

.. ocv:function:: Ptr<AlignMTB> createAlignMTB(int max_bits = 6, int exclude_range = 4, bool cut = true)

    :param max_bits: logarithm to the base 2 of maximal shift in each dimension. Values of 5 and 6 are usually good enough (31 and 63 pixels shift respectively).

    :param exclude_range: range for exclusion bitmap that is constructed to suppress noise around the median value.

    :param cut: if true cuts images, otherwise fills the new regions with zeros.

CalibrateCRF
---------------------------
.. ocv:class:: CalibrateCRF : public Algorithm

The base class for camera response calibration algorithms.

CalibrateCRF::process
---------------------------
Recovers inverse camera response.

.. ocv:function:: void CalibrateCRF::process(InputArrayOfArrays src, OutputArray dst, InputArray times)

    :param src: vector of input images

    :param dst: 256x1 matrix with inverse camera response function

    :param times: vector of exposure time values for each image

CalibrateDebevec
---------------------------
.. ocv:class:: CalibrateDebevec : public CalibrateCRF

Inverse camera response function is extracted for each brightness value by minimizing an objective function as linear system.
Objective function is constructed using pixel values on the same position in all images, extra term is added to make the result smoother.

For more information see [DM97]_.

createCalibrateDebevec
---------------------------
Creates CalibrateDebevec object

.. ocv:function:: createCalibrateDebevec(int samples = 70, float lambda = 10.0f, bool random = false)

    :param samples: number of pixel locations to use

    :param lambda: smoothness term weight. Greater values produce smoother results, but can alter the response.

    :param random: if true sample pixel locations are chosen at random, otherwise the form a rectangular grid.

CalibrateRobertson
---------------------------
.. ocv:class:: CalibrateRobertson : public CalibrateCRF

Inverse camera response function is extracted for each brightness value by minimizing an objective function as linear system.
This algorithm uses all image pixels.

For more information see [RB99]_.

createCalibrateRobertson
---------------------------
Creates CalibrateRobertson object

.. ocv:function:: createCalibrateRobertson(int max_iter = 30, float threshold = 0.01f)

    :param max_iter: maximal number of Gauss-Seidel solver iterations.

    :param threshold: target difference between results of two successive steps of the minimization.

MergeExposures
---------------------------
.. ocv:class:: MergeExposures : public Algorithm

The base class algorithms that can merge exposure sequence to a single image.

MergeExposures::process
---------------------------
Merges images.

.. ocv:function:: void MergeExposures::process(InputArrayOfArrays src, OutputArray dst, InputArray times, InputArray response)

    :param src: vector of input images

    :param dst: result image

    :param times: vector of exposure time values for each image

    :param response: 256x1 matrix with inverse camera response function for each pixel value, it should have the same number of channels as images.

MergeDebevec
---------------------------
.. ocv:class:: MergeDebevec : public MergeExposures

The resulting HDR image is calculated as weighted average of the exposures considering exposure values and camera response.

For more information see [DM97]_.

createMergeDebevec
---------------------------
Creates MergeDebevec object

.. ocv:function:: Ptr<MergeDebevec> createMergeDebevec()

MergeMertens
---------------------------
.. ocv:class:: MergeMertens : public MergeExposures

Pixels are weighted using contrast, saturation and well-exposedness measures, than images are combined using laplacian pyramids.

The resulting image weight is constructed as weighted average of contrast, saturation and well-exposedness measures.

The resulting image doesn't require tonemapping and can be converted to 8-bit image by multiplying by 255, but it's recommended to apply gamma correction and/or linear tonemapping.

For more information see [MK07]_.

MergeMertens::process
---------------------------
Short version of process, that doesn't take extra arguments.

.. ocv:function:: void MergeMertens::process(InputArrayOfArrays src, OutputArray dst)

    :param src: vector of input images

    :param dst: result image

createMergeMertens
---------------------------
Creates MergeMertens object

.. ocv:function:: Ptr<MergeMertens> createMergeMertens(float contrast_weight = 1.0f, float saturation_weight = 1.0f, float exposure_weight = 0.0f)

    :param contrast_weight: contrast measure weight. See :ocv:class:`MergeMertens`.

    :param saturation_weight: saturation measure weight

    :param exposure_weight: well-exposedness measure weight

MergeRobertson
---------------------------
.. ocv:class:: MergeRobertson : public MergeExposures

The resulting HDR image is calculated as weighted average of the exposures considering exposure values and camera response.

For more information see [RB99]_.

createMergeRobertson
---------------------------
Creates MergeRobertson object

.. ocv:function:: Ptr<MergeRobertson> createMergeRobertson()

References
==========

.. [DM03] F. Drago, K. Myszkowski, T. Annen, N. Chiba, "Adaptive Logarithmic Mapping For Displaying High Contrast Scenes", Computer Graphics Forum, 2003, 22, 419 - 426.

.. [FL02] R. Fattal, D. Lischinski, M. Werman, "Gradient Domain High Dynamic Range Compression", Proceedings OF ACM SIGGRAPH, 2002, 249 - 256.

.. [DD02] F. Durand and Julie Dorsey, "Fast Bilateral Filtering for the Display of High-Dynamic-Range Images", ACM Transactions on Graphics, 2002, 21, 3, 257 - 266.

.. [RD05] E. Reinhard, K. Devlin, "Dynamic Range Reduction Inspired by Photoreceptor Physiology", IEEE Transactions on Visualization and Computer Graphics, 2005, 11, 13 - 24.

.. [MM06] R. Mantiuk, K. Myszkowski, H.-P. Seidel, "Perceptual Framework for Contrast Processing of High Dynamic Range Images", ACM Transactions on Applied Perception, 2006, 3, 3, 286 - 308.

.. [GW03] G. Ward, "Fast, Robust Image Registration for Compositing High Dynamic Range Photographs from Handheld Exposures", Journal of Graphics Tools, 2003, 8, 17 - 30.

.. [DM97] P. Debevec, J. Malik, "Recovering High Dynamic Range Radiance Maps from Photographs", Proceedings OF ACM SIGGRAPH, 1997, 369 - 378.

.. [MK07] T. Mertens, J. Kautz, F. Van Reeth, "Exposure Fusion", Proceedings of the 15th Pacific Conference on Computer Graphics and Applications, 2007, 382 - 390.

.. [RB99]  M. Robertson , S. Borman , R. Stevenson , "Dynamic range improvement through multiple exposures ", Proceedings of the Int. Conf. on Image Processing , 1999, 159 - 163.
