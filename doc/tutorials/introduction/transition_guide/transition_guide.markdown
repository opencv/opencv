Transition guide {#tutorial_transition_guide}
================

@prev_tutorial{tutorial_documentation}
@next_tutorial{tutorial_cross_referencing}


@tableofcontents

Changes overview {#tutorial_transition_overview}
================
This document is intended to software developers who want to migrate their code to OpenCV 3.0.

OpenCV 3.0 introduced many new algorithms and features comparing to version 2.4. Some modules have been rewritten, some have been reorganized. Although most of the algorithms from 2.4 are still present, the interfaces can differ.

This section describes most notable changes in general, all details and examples of transition actions are in the next part of the document.

##### Contrib repository
<https://github.com/opencv/opencv_contrib>

This is a place for all new, experimental and non-free algorithms. It does not receive so much attention from the support team comparing to main repository, but the community makes an effort to keep it in a good shape.

To build OpenCV with _contrib_ repository, add the following option to your cmake command:
@code{.sh}
-DOPENCV_EXTRA_MODULES_PATH=<path-to-opencv_contrib>/modules
@endcode

##### Headers layout
In 2.4 all headers are located in corresponding module subfolder (_opencv2/\<module\>/\<module\>.hpp_), in 3.0 there are top-level module headers containing the most of the module functionality: _opencv2/\<module\>.hpp_ and all C-style API definitions have been moved to separate headers (for example opencv2/core/core_c.h).

##### Algorithm interfaces
General algorithm usage pattern has changed: now it must be created on heap wrapped in smart pointer cv::Ptr. Version 2.4 allowed both stack and heap allocations, directly or via smart pointer.

_get_ and _set_ methods have been removed from the cv::Algorithm class along with _CV_INIT_ALGORITHM_ macro. In 3.0 all properties have been converted to the pairs of _getProperty/setProperty_ pure virtual methods. As a result it is __not__ possible to create and use cv::Algorithm instance by name (using generic _Algorithm::create(String)_ method), one should call corresponding factory method explicitly.

##### Changed modules
-   _ml_ module has been rewritten
-   _highgui_ module has been split into parts: _imgcodecs_, _videoio_ and _highgui_ itself
-   _features2d_ module have been reorganized (some feature detectors has been moved to _opencv_contrib/xfeatures2d_ module)
-   _legacy_, _nonfree_ modules have been removed. Some algorithms have been moved to different locations and some have been completely rewritten or removed
-   CUDA API has been updated (_gpu_ module -> several _cuda_ modules, namespace _gpu_ -> namespace _cuda_)
-   OpenCL API has changed (_ocl_ module has been removed, separate _ocl::_ implementations -> Transparent API)
-   Some other methods and classes have been relocated

Transition hints {#tutorial_transition_hints}
================
This section describes concrete actions with examples.

Prepare 2.4 {#tutorial_transition_hints_24}
-----------
Some changes made in the latest 2.4.11 OpenCV version allow you to prepare current codebase to migration:

- cv::makePtr function is now available
- _opencv2/\<module\>.hpp_ headers have been created

New headers layout {#tutorial_transition_hints_headers}
------------------
__Note:__
Changes intended to ease the migration have been made in OpenCV 3.0, thus the following instructions are not necessary, but recommended.

1. Replace inclusions of old module headers:
@code{.cpp}
// old header
#include "opencv2/<module>/<module>.hpp"
// new header
#include "opencv2/<module>.hpp"
@endcode

2. If your code is using C API (`cv*` functions, `Cv*` structures or `CV_*` enumerations), include corresponding `*_c.h` headers. Although it is recommended to use C++ API, most of C-functions are still accessible in separate header files (opencv2/core/core_c.h, opencv2/core/types_c.h, opencv2/imgproc/imgproc_c.h, etc.).

Modern way to use algorithm {#tutorial_transition_algorithm}
---------------------------
1.  Algorithm instances must be created with cv::makePtr function or corresponding static factory method if available:
    @code{.cpp}
    // good ways
    Ptr<SomeAlgo> algo = makePtr<SomeAlgo>(...);
    Ptr<SomeAlgo> algo = SomeAlgo::create(...);
    @endcode
    Other ways are deprecated:
    @code{.cpp}
    // bad ways
    Ptr<SomeAlgo> algo = new SomeAlgo(...);
    SomeAlgo * algo = new SomeAlgo(...);
    SomeAlgo algo(...);
    Ptr<SomeAlgo> algo = Algorithm::create<SomeAlgo>("name");
    @endcode

2.  Algorithm properties should be accessed via corresponding virtual methods, _getSomeProperty/setSomeProperty_, generic _get/set_ methods have been removed:
    @code{.cpp}
    // good way
    double clipLimit = clahe->getClipLimit();
    clahe->setClipLimit(clipLimit);
    // bad way
    double clipLimit = clahe->getDouble("clipLimit");
    clahe->set("clipLimit", clipLimit);
    clahe->setDouble("clipLimit", clipLimit);
    @endcode


3.  Remove `initModule_<moduleName>()` calls

Machine learning module {#tutorial_transition_hints_ml}
-----------------------
Since this module has been rewritten, it will take some effort to adapt your software to it. All algorithms are located in separate _ml_ namespace along with their base class _StatModel_. Separate _SomeAlgoParams_ classes have been replaced with a sets of corresponding _getProperty/setProperty_ methods.

The following table illustrates correspondence between 2.4 and 3.0 machine learning classes.

|       2.4 | 3.0       |
| --------- | --------- |
| CvStatModel | cv::ml::StatModel |
| CvNormalBayesClassifier | cv::ml::NormalBayesClassifier |
| CvKNearest | cv::ml::KNearest |
| CvSVM | cv::ml::SVM |
| CvDTree | cv::ml::DTrees |
| CvBoost | cv::ml::Boost |
| CvGBTrees | _Not implemented_ |
| CvRTrees | cv::ml::RTrees |
| CvERTrees | _Not implemented_ |
| EM | cv::ml::EM |
| CvANN_MLP | cv::ml::ANN_MLP |
| _Not implemented_ | cv::ml::LogisticRegression |
| CvMLData | cv::ml::TrainData |

Although rewritten _ml_ algorithms in 3.0 allow you to load old trained models from _xml/yml_ file, deviations in prediction process are possible.

The following code snippets from the `points_classifier.cpp` example illustrate differences in model training process:
@code{.cpp}
using namespace cv;
// ======== version 2.4 ========
Mat trainSamples, trainClasses;
prepare_train_data( trainSamples, trainClasses );
CvBoost  boost;
Mat var_types( 1, trainSamples.cols + 1, CV_8UC1, Scalar(CV_VAR_ORDERED) );
var_types.at<uchar>( trainSamples.cols ) = CV_VAR_CATEGORICAL;
CvBoostParams  params( CvBoost::DISCRETE, // boost_type
                       100, // weak_count
                       0.95, // weight_trim_rate
                       2, // max_depth
                       false, //use_surrogates
                       0 // priors
                     );
boost.train( trainSamples, CV_ROW_SAMPLE, trainClasses, Mat(), Mat(), var_types, Mat(), params );

// ======== version 3.0 ========
Ptr<Boost> boost = Boost::create();
boost->setBoostType(Boost::DISCRETE);
boost->setWeakCount(100);
boost->setWeightTrimRate(0.95);
boost->setMaxDepth(2);
boost->setUseSurrogates(false);
boost->setPriors(Mat());
boost->train(prepare_train_data()); // 'prepare_train_data' returns an instance of ml::TrainData class
@endcode

Features detect {#tutorial_transition_hints_features}
---------------
Some algorithms (FREAK, BRIEF, SIFT, SURF) has been moved to _opencv_contrib_ repository, to _xfeatures2d_ module, _xfeatures2d_ namespace. Their interface has been also changed (inherit from `cv::Feature2D` base class).

List of _xfeatures2d_ module classes:

- cv::xfeatures2d::BriefDescriptorExtractor - Class for computing BRIEF descriptors (2.4 location: _features2d_)
- cv::xfeatures2d::FREAK - Class implementing the FREAK (Fast Retina Keypoint) keypoint descriptor (2.4 location: _features2d_)
- cv::xfeatures2d::StarDetector - The class implements the  CenSurE detector (2.4 location: _features2d_)
- cv::xfeatures2d::SIFT - Class for extracting keypoints and computing descriptors using the Scale Invariant Feature Transform (SIFT) algorithm (2.4 location: _nonfree_)
- cv::xfeatures2d::SURF - Class for extracting Speeded Up Robust Features from an image (2.4 location: _nonfree_)

Following steps are needed:
1. Add _opencv_contrib_ to compilation process
2. Include `opencv2/xfeatures2d.h` header
3. Use namespace `xfeatures2d`
4. Replace `operator()` calls with `detect`, `compute` or `detectAndCompute` if needed

Some classes now use general methods `detect`, `compute` or `detectAndCompute` provided by `Feature2D` base class instead of custom `operator()`

Following code snippets illustrate the difference (from `video_homography.cpp` example):
@code{.cpp}
using namespace cv;
// ====== 2.4 =======
#include "opencv2/features2d/features2d.hpp"
BriefDescriptorExtractor brief(32);
GridAdaptedFeatureDetector detector(new FastFeatureDetector(10, true), DESIRED_FTRS, 4, 4);
// ...
detector.detect(gray, query_kpts); //Find interest points
brief.compute(gray, query_kpts, query_desc); //Compute brief descriptors at each keypoint location
// ====== 3.0 =======
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
using namespace cv::xfeatures2d;
Ptr<BriefDescriptorExtractor> brief = BriefDescriptorExtractor::create(32);
Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(10, true);
// ...
detector->detect(gray, query_kpts); //Find interest points
brief->compute(gray, query_kpts, query_desc); //Compute brief descriptors at each keypoint location
@endcode

OpenCL {#tutorial_transition_hints_opencl}
------
All specialized `ocl` implementations has been hidden behind general C++ algorithm interface. Now the function execution path can be selected dynamically at runtime: CPU or OpenCL; this mechanism is also called "Transparent API".

New class cv::UMat is intended to hide data exchange with OpenCL device in a convenient way.

Following example illustrate API modifications (from [OpenCV site](http://opencv.org/platforms/opencl.html)):

-   OpenCL-aware code OpenCV-2.x
@code{.cpp}
// initialization
VideoCapture vcap(...);
ocl::OclCascadeClassifier fd("haar_ff.xml");
ocl::oclMat frame, frameGray;
Mat frameCpu;
vector<Rect> faces;
for(;;){
    // processing loop
    vcap >> frameCpu;
    frame = frameCpu;
    ocl::cvtColor(frame, frameGray, BGR2GRAY);
    ocl::equalizeHist(frameGray, frameGray);
    fd.detectMultiScale(frameGray, faces, ...);
    // draw rectangles …
    // show image …
}
@endcode
-   OpenCL-aware code OpenCV-3.x
@code{.cpp}
// initialization
VideoCapture vcap(...);
CascadeClassifier fd("haar_ff.xml");
UMat frame, frameGray; // the only change from plain CPU version
vector<Rect> faces;
for(;;){
    // processing loop
    vcap >> frame;
    cvtColor(frame, frameGray, BGR2GRAY);
    equalizeHist(frameGray, frameGray);
    fd.detectMultiScale(frameGray, faces, ...);
    // draw rectangles …
    // show image …
}
@endcode

CUDA {#tutorial_transition_hints_cuda}
----
_cuda_ module has been split into several smaller pieces:
- _cuda_ - @ref cuda
- _cudaarithm_ - @ref cudaarithm
- _cudabgsegm_ - @ref cudabgsegm
- _cudacodec_ - @ref cudacodec
- _cudafeatures2d_ - @ref cudafeatures2d
- _cudafilters_ - @ref cudafilters
- _cudaimgproc_ - @ref cudaimgproc
- _cudalegacy_ - @ref cudalegacy
- _cudaoptflow_ - @ref cudaoptflow
- _cudastereo_ - @ref cudastereo
- _cudawarping_ - @ref cudawarping
- _cudev_ - @ref cudev

`gpu` namespace has been removed, use cv::cuda namespace instead. Many classes has also been renamed, for example:
- `gpu::FAST_GPU` -> cv::cuda::FastFeatureDetector
- `gpu::createBoxFilter_GPU` -> cv::cuda::createBoxFilter

Documentation format {#tutorial_transition_docs}
--------------------
Documentation has been converted to Doxygen format. You can find updated documentation writing guide in _Tutorials_ section of _OpenCV_ reference documentation (@ref tutorial_documentation).

Support both versions {#tutorial_transition_both}
---------------------
In some cases it is possible to support both versions of OpenCV.

### Source code

To check library major version in your application source code, the following method should be used:
@code{.cpp}
#include "opencv2/core/version.hpp"
#if CV_MAJOR_VERSION == 2
// do opencv 2 code
#elif CV_MAJOR_VERSION == 3
// do opencv 3 code
#endif
@endcode

@note Do not use __CV_VERSION_MAJOR__, it has different meaning for 2.4 and 3.x branches!

### Build system

It is possible to link different modules or enable/disable some of the features in your application by checking library version in the build system. Standard cmake or pkg-config variables can be used for this:
- `OpenCV_VERSION` for cmake will contain full version: "2.4.11" or "3.0.0" for example
- `OpenCV_VERSION_MAJOR` for cmake will contain only major version number: 2 or 3
- pkg-config file has standard field `Version`

Example:
@code{.cmake}
if(OpenCV_VERSION VERSION_LESS "3.0")
# use 2.4 modules
else()
# use 3.x modules
endif()
@endcode
