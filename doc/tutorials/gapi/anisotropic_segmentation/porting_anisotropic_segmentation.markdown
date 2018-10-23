# Porting anisotropic image segmentation on G-API {#tutorial_gapi_anisotropic_segmentation}

[TOC]

# Introduction {#gapi_anisotropic_intro}

In this tutorial you will learn:
* How an existing algorithm can be transformed to a G-API computation
  (graph);
* How to inspect G-API graphs;
* How G-API can be extended if a particular function is missing;
* How to test accuracy and measure performance.

This tutorial is based on @ref
tutorial_anisotropic_image_segmentation_by_a_gst.

# Quick start: using OpenCV backend {#gapi_anisotropic_start}

Before we start, let's review the [original algorithm](@ref tutorial_anisotropic_image_segmentation_by_a_gst) implementation:

@include cpp/tutorial_code/ImgProc/anisotropic_image_segmentation/anisotropic_image_segmentation.cpp

## Examining calcGST() {#gapi_anisotropic_calcgst}

The function calcGST() is clearly an image processing pipeline:
* It is just a sequence of operations over a number of cv::Mat;
* No logic (conditionals) and loops involved in the code;
* All functions operate on 2D images (like cv::Sobel, cv::multiply,
cv::boxFilter, cv::sqrt, etc).

Considering the above, calcGST() is a great candidate to start
with. In the original code, its prototype is defined like this:

@snippet cpp/tutorial_code/ImgProc/anisotropic_image_segmentation/anisotropic_image_segmentation.cpp calcGST_proto

With G-API, we can define it as follows:

@snippet cpp/tutorial_code/gapi/porting_anisotropic_image_segmentation/porting_anisotropic_image_segmentation_gapi.cpp calcGST_proto

It is important to understand that the new G-API based version of
calcGST() will just produce a compute graph, in contrast to its
original version. This is a principial difference - G-API based
functions like this are used to construct graphs, not to process the
actual data.

Let's start implementing calcGST() with calculation of \f$J\f$
matrix. This is how the original code looks like:

@snippet cpp/tutorial_code/ImgProc/anisotropic_image_segmentation/anisotropic_image_segmentation.cpp calcJ_header

Here we need to declare output objects for every new operation (see
img as a result for cv::Mat::convertTo, imgDiffX and others as results for
cv::Sobel and cv::multiply).

The G-API analogue is listed below:

@snippet cpp/tutorial_code/gapi/porting_anisotropic_image_segmentation/porting_anisotropic_image_segmentation_gapi.cpp calcGST_header

This snippet demonstrates the following syntactic difference between
G-API and traditional OpenCV:
* All standard G-API functions are by default placed in "cv::gapi"
namespace;
* G-API operations _return_ its results -- there's no need to pass
extra "output" parameters to the functions.

G-API standard kernels are trying to follow OpenCV API conventions
whenever possible -- so cv::gapi::sobel takes the same arguments as
cv::Sobel, cv::gapi::mul follows cv::multiply, and so on (except
having the return value).

The rest of calcGST() function can be implemented the same
way trivially. Below is its full source code:

@snippet cpp/tutorial_code/gapi/porting_anisotropic_image_segmentation/porting_anisotropic_image_segmentation_gapi.cpp calcGST

## Running G-API graph {#gapi_anisotropic_running}

After calcGST() is defined in G-API language, we can construct a graph
based on it and finally run it -- pass input image and obtain
result. Before we do it, let's have a look how original code looked
like:

@snippet cpp/tutorial_code/ImgProc/anisotropic_image_segmentation/anisotropic_image_segmentation.cpp main_extra

G-API-based functions like calcGST() can't be applied to input data
directly, since it is a _construction_ code, not the _processing_ code.
In order to _run_ computations, a special object of class
cv::GComputation needs to be created. This object wraps our G-API code
(which is a composition of G-API data and operations) into a callable
object, similar to C++11 std::function<>.

cv::GComputation class has a number of constructors which can be used
to define a graph. Generally, the user needs to pass graph boundaries
-- _input_ and _output_ objects, on which a GComputation is
defined. Then G-API analyzes the call flow from _outputs_ to _inputs_
and reconstructs the graph with operations in-between the specified
boundaries. This may sound complex, however in fact the code looks
like this:

@snippet cpp/tutorial_code/gapi/porting_anisotropic_image_segmentation/porting_anisotropic_image_segmentation_gapi.cpp main
