High level stitching API (Stitcher class) {#tutorial_stitcher}
=========================================

Goal
----

In this tutorial you will learn how to:

-   use the high-level stitching API for stitching provided by
    -   @ref cv::Stitcher
-   learn how to use preconfigured Stitcher configurations to stitch images
    using different camera models.

Code
----

This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/master/samples/cpp/samples/cpp/stitching.cpp).

@include samples/cpp/stitching.cpp

Explanation
-----------

The most important code part is:

@snippet cpp/stitching.cpp stitching

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

See also
--------

If you want to study internals of the stitching pipeline or you want to experiment with detailed
configuration see
[stitching_detailed.cpp](https://github.com/opencv/opencv/tree/master/samples/cpp/stitching_detailed.cpp)
in `opencv/samples/cpp` folder.
