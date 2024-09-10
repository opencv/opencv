High level stitching API (Stitcher class) {#tutorial_stitcher}
=========================================

@tableofcontents

@prev_tutorial{tutorial_hdr_imaging}
@next_tutorial{tutorial_background_subtraction}

|    |    |
| -: | :- |
| Original author | Jiri Horner |
| Compatibility | OpenCV >= 3.2 |

Goal
----

In this tutorial you will learn how to:

-   use the high-level stitching API for stitching provided by
    -   @ref cv::Stitcher
-   learn how to use preconfigured Stitcher configurations to stitch images
    using different camera models.

Code
----
@add_toggle_cpp
This tutorial's code is shown in the lines below. You can download it from [here](https://github.com/opencv/opencv/tree/4.x/samples/cpp/stitching.cpp).

Note: The C++ version includes additional options such as image division (--d3) and more detailed error handling, which are not present in the Python example.

@include samples/cpp/stitching.cpp

@end_toggle

@add_toggle_python
This tutorial's code is shown in the lines below. You can download it from [here](https://github.com/opencv/opencv/blob/4.x/samples/python/stitching.py).

Note: The C++ version includes additional options such as image division (--d3) and more detailed error handling, which are not present in the Python example.

@include samples/python/stitching.py

@end_toggle

Explanation
-----------

The most important code part is:

@add_toggle_cpp
@snippet cpp/stitching.cpp stitching
@end_toggle

@add_toggle_python
@snippet python/stitching.py stitching
@end_toggle

A new instance of stitcher is created and the @ref cv::Stitcher::stitch will
do all the hard work.

@ref cv::Stitcher::create can create stitcher in one of the predefined
configurations (argument `mode`). See @ref cv::Stitcher::Mode for details. These
configurations will setup multiple stitcher properties to operate in one of
predefined scenarios. After you create stitcher in one of predefined
configurations you can adjust stitching by setting any of the stitcher
properties.

If you have cuda device @ref cv::Stitcher can be configured to offload certain
operations to GPU. If you prefer this configuration set `try_use_gpu` to true.
OpenCL acceleration will be used transparently based on global OpenCV settings
regardless of this flag.

Stitching might fail for several reasons, you should always check if
everything went good and resulting pano is stored in `pano`. See
@ref cv::Stitcher::Status documentation for possible error codes.

Camera models
-------------

There are currently 2 camera models implemented in stitching pipeline.

- _Homography model_ expecting perspective transformations between images
  implemented in @ref cv::detail::BestOf2NearestMatcher cv::detail::HomographyBasedEstimator
  cv::detail::BundleAdjusterReproj cv::detail::BundleAdjusterRay
- _Affine model_ expecting affine transformation with 6 DOF or 4 DOF implemented in
  @ref cv::detail::AffineBestOf2NearestMatcher cv::detail::AffineBasedEstimator
  cv::detail::BundleAdjusterAffine cv::detail::BundleAdjusterAffinePartial cv::AffineWarper

Homography model is useful for creating photo panoramas captured by camera,
while affine-based model can be used to stitch scans and object captured by
specialized devices.

@note
Certain detailed settings of @ref cv::Stitcher might not make sense. Especially
you should not mix classes implementing affine model and classes implementing
Homography model, as they work with different transformations.

Try it out
----------

If you enabled building samples you can found binary under
`build/bin/cpp-example-stitching`. This example is a console application, run it without
arguments to see help. `opencv_extra` provides some sample data for testing all available
configurations.

to try panorama mode run:
```
./cpp-example-stitching --mode panorama <path to opencv_extra>/testdata/stitching/boat*
```
![](images/boat.jpg)

to try scans mode run (dataset from home-grade scanner):
```
./cpp-example-stitching --mode scans <path to opencv_extra>/testdata/stitching/newspaper*
```
![](images/newspaper.jpg)

or (dataset from professional book scanner):
```
./cpp-example-stitching --mode scans <path to opencv_extra>/testdata/stitching/budapest*
```
![](images/budapest.jpg)

@note
Examples above expects POSIX platform, on windows you have to provide all files names explicitly
(e.g. `boat1.jpg` `boat2.jpg`...) as windows command line does not support `*` expansion.

Stitching detailed (python opencv >4.0.1)
--------

If you want to study internals of the stitching pipeline or you want to experiment with detailed
configuration you can use stitching_detailed source code available in C++ or python

<H4>stitching_detailed</H4>
@add_toggle_cpp
[stitching_detailed.cpp](https://raw.githubusercontent.com/opencv/opencv/4.x/samples/cpp/stitching_detailed.cpp)
@end_toggle

@add_toggle_python
[stitching_detailed.py](https://raw.githubusercontent.com/opencv/opencv/4.x/samples/python/stitching_detailed.py)
@end_toggle

stitching_detailed program uses command line to get stitching parameter. Many parameters exists. Above examples shows some command line parameters possible :

boat5.jpg boat2.jpg boat3.jpg boat4.jpg boat1.jpg boat6.jpg --work_megapix 0.6 --features orb --matcher homography --estimator homography --match_conf 0.3 --conf_thresh 0.3 --ba ray --ba_refine_mask xxxxx --save_graph test.txt --wave_correct no --warp fisheye --blend  multiband --expos_comp no --seam gc_colorgrad

![](images/fisheye.jpg)

Pairwise images are matched using an homography --matcher homography and estimator used for transformation estimation too --estimator homography

Confidence for feature matching step is 0.3 : --match_conf 0.3. You can decrease this value if you have some difficulties to match images

Threshold for two images are from the same panorama confidence is 0. : --conf_thresh 0.3 You can decrease this value if you have some difficulties to match images

Bundle adjustment cost function is ray --ba ray

Refinement mask for bundle adjustment is xxxxx ( --ba_refine_mask xxxxx) where 'x' means refine respective parameter and '_' means don't. Refine one, and has the following format: fx,skew,ppx,aspect,ppy

Save matches graph represented in DOT language to test.txt ( --save_graph test.txt) : Labels description: Nm is number of matches, Ni is number of inliers, C is confidence

![](images/gvedit.jpg)

Perform wave effect correction is no (--wave_correct no)

Warp surface type is fisheye (--warp fisheye)

Blending method is multiband (--blend  multiband)

Exposure compensation method is not used (--expos_comp no)

Seam estimation estimator is  Minimum graph cut-based seam (--seam gc_colorgrad)

you can use those arguments on command line too :

boat5.jpg boat2.jpg boat3.jpg boat4.jpg boat1.jpg boat6.jpg --work_megapix 0.6 --features orb --matcher homography --estimator homography --match_conf 0.3 --conf_thresh 0.3 --ba ray --ba_refine_mask xxxxx --wave_correct horiz --warp compressedPlaneA2B1 --blend multiband --expos_comp channels_blocks --seam gc_colorgrad

You will get :

![](images/compressedPlaneA2B1.jpg)

For images captured using a scanner or a drone ( affine motion) you can use those arguments on command line :

newspaper1.jpg newspaper2.jpg --work_megapix 0.6 --features surf --matcher affine --estimator affine --match_conf 0.3 --conf_thresh 0.3 --ba affine --ba_refine_mask xxxxx --wave_correct no --warp affine

![](images/affinepano.jpg)

You can find  all images in https://github.com/opencv/opencv_extra/tree/4.x/testdata/stitching
