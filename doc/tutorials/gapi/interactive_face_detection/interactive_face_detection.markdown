# Face analytics pipeline with G-API {#tutorial_gapi_interactive_face_detection}

@next_tutorial{tutorial_gapi_anisotropic_segmentation}

[TOC]

# Overview {#gapi_ifd_intro}

In this tutorial you will learn:
* How to integrate Deep Learning inference in a G-API graph;
* How to run a G-API graph on a video stream and obtain data from it.

# Prerequisites {#gapi_ifd_prereq}

This sample requires:
- PC with GNU/Linux or Microsoft Windows (Apple macOS is supported but
  was not tested);
- OpenCV 4.2 or later built with Intel® Distribution of [OpenVINO™
  Toolkit](https://docs.openvinotoolkit.org/) (building with [Intel®
  TBB](https://www.threadingbuildingblocks.org/intel-tbb-tutorial) is
  a plus);
- The following topologies from OpenVINO™ Toolkit [Open Model
  Zoo](https://github.com/opencv/open_model_zoo):
  - `face-detection-adas-0001`;
  - `age-gender-recognition-retail-0013`;
  - `emotions-recognition-retail-0003`.

# Introduction: why G-API {#gapi_ifd_why}

Many computer vision algorithms run on a video stream rather than on
individual images. Stream processing usually consists of multiple
steps -- like decode, preprocessing, detection, tracking,
classification (on detected objects), and visualization -- forming a
*video processing pipeline*. Moreover, many these steps of such
pipeline can run in parallel -- modern platforms have different
hardware blocks on the same chip like decoders and GPUs, and extra
accelerators can be plugged in as extensions, like Intel® Movidius™
Neural Compute Stick for deep learning offload.

Given all this manifold of options and a variety in video analytics
algorithms, managing such pipelines effectively quickly becomes a
problem. For sure it can be done manually, but this approach doesn't
scale: if a change is required in the algorithm (e.g. a new pipeline
step is added), or if it is ported on a new platform with different
capabilities, the whole pipeline needs to be re-optimized.

Starting with version 4.2, OpenCV offers a solution to this
problem. OpenCV G-API now can manage Deep Learning inference (a
cornerstone of any modern analytics pipeline) with a traditional
Computer Vision as well as video capturing/decoding, all in a single
pipeline. G-API takes care of pipelining itself -- so if the algorithm
or platform changes, the execution model adapts to it automatically.

# Pipeline overview {#gapi_ifd_overview}

Our sample application is based on ["Interactive Face Detection"] demo
from OpenVINO™ Toolkit Open Model Zoo. A simplified pipeline consists
of the following steps:
1. Image acquisition and decode;
2. Detection with preprocessing;
3. Classification with preprocessing for every detected object with
   two networks;
4. Visualization.

\dot
digraph pipeline {
  node [shape=record fontname=Helvetica fontsize=10 style=filled color="#4c7aa4" fillcolor="#5b9bd5" fontcolor="white"];
  edge [color="#62a8e7"];
  splines=ortho;

  rankdir = LR;
  subgraph cluster_0 {
    color=invis;
    capture [label="Capture\nDecode"];
    resize [label="Resize\nConvert"];
    detect [label="Detect faces"];
    capture -> resize -> detect
  }

  subgraph cluster_1 {
    graph[style=dashed];

    subgraph cluster_2 {
      color=invis;
      temp_4 [style=invis shape=point width=0];
      postproc_1 [label="Crop\nResize\nConvert"];
      age_gender [label="Classify\nAge/gender"];
      postproc_1 -> age_gender [constraint=true]
      temp_4 -> postproc_1 [constraint=none]
    }

    subgraph cluster_3 {
      color=invis;
      postproc_2 [label="Crop\nResize\nConvert"];
      emo [label="Classify\nEmotions"];
      postproc_2 -> emo [constraint=true]
    }
    label="(for each face)";
  }

  temp_1 [style=invis shape=point width=0];
  temp_2 [style=invis shape=point width=0];
  detect -> temp_1 [arrowhead=none]
  temp_1 -> postproc_1

  capture -> {temp_4, temp_2} [arrowhead=none constraint=false]
  temp_2 -> postproc_2

  temp_1 -> temp_2 [arrowhead=none constraint=false]

  temp_3 [style=invis shape=point width=0];
  show [label="Visualize\nDisplay"];

  {age_gender, emo} -> temp_3 [arrowhead=none]
  temp_3 -> show
}
\enddot

# Constructing a pipeline {#gapi_ifd_constructing}

Constructing a G-API graph for a video streaming case does not differ
much from a [regular usage](@ref gapi_example) of G-API -- it is still
about defining graph *data* (with cv::GMat, cv::GScalar, and
cv::GArray) and *operations* over it. Inference also becomes an
operation in the graph, but is defined in a little bit different way.

## Declaring Deep Learning topologies {#gapi_ifd_declaring_nets}

In contrast with traditional CV functions (see [core] and [imgproc])
where G-API declares distinct operations for every function, inference
in G-API is a single generic operation cv::gapi::infer<>. As usual, it
is just an interface and it can be implemented in a number of ways under
the hood. In OpenCV 4.2, only OpenVINO™ Inference Engine-based backend
is available, and OpenCV's own DNN module-based backend is to come.

cv::gapi::infer<> is _parametrized_ by the details of a topology we are
going to execute. Like operations, topologies in G-API are strongly
typed and are defined with a special macro G_API_NET():

@snippet cpp/tutorial_code/gapi/age_gender_emotion_recognition/age_gender_emotion_recognition.cpp G_API_NET

Similar to how operations are defined with G_API_OP(), network
description requires three parameters:
1. A type name. Every defined topology is declared as a distinct C++
   type which is used further in the program -- see below;
2. A `std::function<>`-like API signature. G-API traits networks as
   regular "functions" which take and return data. Here network
   `Faces` (a detector) takes a cv::GMat and returns a cv::GMat, while
   network `AgeGender` is known to provide two outputs (age and gender
   blobs, respectively) -- so its has a `std::tuple<>` as a return
   type.
3. A topology name -- can be any non-empty string, G-API is using
   these names to distinguish networks inside. Names should be unique
   in the scope of a single graph.

## Building a GComputation {#gapi_ifd_gcomputation}

Now the above pipeline is expressed in G-API like this:

@snippet cpp/tutorial_code/gapi/age_gender_emotion_recognition/age_gender_emotion_recognition.cpp GComputation

Every pipeline starts with declaring empty data objects -- which act
as inputs to the pipeline. Then we call a generic cv::gapi::infer<>
specialized to `Faces` detection network. cv::gapi::infer<> inherits its
signature from its template parameter -- and in this case it expects
one input cv::GMat and produces one output cv::GMat.

In this sample we use a pre-trained SSD-based network and its output
needs to be parsed to an array of detections (object regions of
interest, ROIs). It is done by a custom operation `custom::PostProc`,
which returns an array of rectangles (of type `cv::GArray<cv::Rect>`)
back to the pipeline. This operation also filters out results by a
confidence threshold -- and these details are hidden in the kernel
itself. Still, at the moment of graph construction we operate with
interfaces only and don't need actual kernels to express the pipeline
-- so the implementation of this post-processing will be listed later.

After detection result output is parsed to an array of objects, we can run
classification on any of those. G-API doesn't support syntax for
in-graph loops like `for_each()` yet, but instead cv::gapi::infer<>
comes with a special list-oriented overload.

User can call cv::gapi::infer<> with a cv::GArray as the first
argument, so then G-API assumes it needs to run the associated network
on every rectangle from the given list of the given frame (second
argument). Result of such operation is also a list -- a  cv::GArray of
cv::GMat.

Since `AgeGender` network itself produces two outputs, it's output
type for a list-based version of cv::gapi::infer is a tuple of
arrays. We use `std::tie()` to decompose this input into two distinct
objects.

`Emotions` network produces a single output so its list-based
inference's return type is `cv::GArray<cv::GMat>`.

# Configuring the pipeline {#gapi_ifd_configuration}

G-API strictly separates construction from configuration -- with the
idea to keep algorithm code itself platform-neutral. In the above
listings we only declared our operations and expressed the overall
data flow, but didn't even mention that we use OpenVINO™. We only
described *what* we do, but not *how* we do it. Keeping these two
aspects clearly separated is the design goal for G-API.

Platform-specific details arise when the pipeline is *compiled* --
i.e. is turned from a declarative to an executable form. The way *how*
to run stuff is specified via compilation arguments, and new
inference/streaming features are no exception from this rule.

G-API is built on backends which implement interfaces (see
[Architecture] and [Kernels] for details) -- thus cv::gapi::infer<> is
a function which can be implemented by different backends. In OpenCV
4.2, only OpenVINO™ Inference Engine backend for inference is
available. Every inference backend in G-API has to provide a special
parameterizable structure to express *backend-specific* neural network
parameters -- and in this case, it is cv::gapi::ie::Params:

@snippet cpp/tutorial_code/gapi/age_gender_emotion_recognition/age_gender_emotion_recognition.cpp Param_Cfg

Here we define three parameter objects: `det_net`, `age_net`, and
`emo_net`. Every object is a cv::gapi::ie::Params structure
parametrization for each particular network we use. On a compilation
stage, G-API automatically matches network parameters with their
cv::gapi::infer<> calls in graph using this information.

Regardless of the topology, every parameter structure is constructed
with three string arguments -- specific to the OpenVINO™ Inference
Engine:
1. Path to the topology's intermediate representation (.xml file);
2. Path to the topology's model weights (.bin file);
3. Device where to run -- "CPU", "GPU", and others -- based on your
OpenVINO™ Toolkit installation.
These arguments are taken from the command-line parser.

Once networks are defined and custom kernels are implemented, the
pipeline is compiled for streaming:

@snippet cpp/tutorial_code/gapi/age_gender_emotion_recognition/age_gender_emotion_recognition.cpp Compile

cv::GComputation::compileStreaming() triggers a special video-oriented
form of graph compilation where G-API is trying to optimize
throughput. Result of this compilation is an object of special type
cv::GStreamingCompiled -- in constract to a traditional callable
cv::GCompiled, these objects are closer to media players in their
semantics.

@note There is no need to pass metadata arguments describing the
format of the input video stream in
cv::GComputation::compileStreaming() -- G-API figures automatically
what are the formats of the input vector and adjusts the pipeline to
these formats on-the-fly. User still can pass metadata there as with
regular cv::GComputation::compile() in order to fix the pipeline to
the specific input format.

# Running the pipeline  {#gapi_ifd_running}

Pipelining optimization is based on processing multiple input video
frames simultaneously, running different steps of the pipeline in
parallel. This is why it works best when the framework takes full
control over the video stream.

The idea behind streaming API is that user specifies an *input source*
to the pipeline and then G-API manages its execution automatically
until the source ends or user interrupts the execution. G-API pulls
new image data from the source and passes it to the pipeline for
processing.

Streaming sources are represented by the interface
cv::gapi::wip::IStreamSource. Objects implementing this interface may
be passed to `GStreamingCompiled` as regular inputs via `cv::gin()`
helper function. In OpenCV 4.2, only one streaming source is allowed
per pipeline -- this requirement will be relaxed in the future.

OpenCV comes with a great class cv::VideoCapture and by default G-API
ships with a stream source class based on it --
cv::gapi::wip::GCaptureSource. Users can implement their own
streaming sources e.g. using [VAAPI] or other Media or Networking
APIs.

Sample application specifies the input source as follows:

@snippet cpp/tutorial_code/gapi/age_gender_emotion_recognition/age_gender_emotion_recognition.cpp Source

Please note that a GComputation may still have multiple inputs like
cv::GMat, cv::GScalar, or cv::GArray objects. User can pass their
respective host-side types (cv::Mat, cv::Scalar, std::vector<>) in the
input vector as well, but in Streaming mode these objects will create
"endless" constant streams. Mixing a real video source stream and a
const data stream is allowed.

Running a pipeline is easy -- just call
cv::GStreamingCompiled::start() and fetch your data with blocking
cv::GStreamingCompiled::pull() or non-blocking
cv::GStreamingCompiled::try_pull(); repeat until the stream ends:

@snippet cpp/tutorial_code/gapi/age_gender_emotion_recognition/age_gender_emotion_recognition.cpp Run

The above code may look complex but in fact it handles two modes --
with and without graphical user interface (GUI):
- When a sample is running in a "headless" mode (`--pure` option is
  set), this code simply pulls data from the pipeline with the
  blocking `pull()` until it ends. This is the most performant mode of
  execution.
- When results are also displayed on the screen, the Window System
  needs to take some time to refresh the window contents and handle
  GUI events. In this case, the demo pulls data with a non-blocking
  `try_pull()` until there is no more data available (but it does not
  mark end of the stream -- just means new data is not ready yet), and
  only then displays the latest obtained result and refreshes the
  screen. Reducing the time spent in GUI with this trick increases the
  overall performance a little bit.

# Comparison with serial mode {#gapi_ifd_comparison}

The sample can also run in a serial mode for a reference and
benchmarking purposes.  In this case, a regular
cv::GComputation::compile() is used and a regular single-frame
cv::GCompiled object is produced; the pipelining optimization is not
applied within G-API; it is the user responsibility to acquire image
frames from cv::VideoCapture object and pass those to G-API.

@snippet cpp/tutorial_code/gapi/age_gender_emotion_recognition/age_gender_emotion_recognition.cpp Run_Serial

On a test machine (Intel® Core™ i5-6600), with OpenCV built with
[Intel® TBB]
support, detector network assigned to CPU, and classifiers to iGPU,
the pipelined sample outperformes the serial one by the factor of
1.36x (thus adding +36% in overall throughput).

# Conclusion {#gapi_ifd_conclusion}

G-API introduces a technological way to build and optimize hybrid
pipelines. Switching to a new execution model does not require changes
in the algorithm code expressed with G-API -- only the way how graph
is triggered differs.

# Listing: post-processing kernel {#gapi_ifd_pp}

G-API gives an easy way to plug custom code into the pipeline even if
it is running in a streaming mode and processing tensor
data. Inference results are represented by multi-dimensional cv::Mat
objects so accessing those is as easy as with a regular DNN module.

The OpenCV-based SSD post-processing kernel is defined and implemented in this
sample as follows:

@snippet cpp/tutorial_code/gapi/age_gender_emotion_recognition/age_gender_emotion_recognition.cpp Postproc

["Interactive Face Detection"]: https://github.com/opencv/open_model_zoo/tree/master/demos/interactive_face_detection_demo
[core]: @ref gapi_core
[imgproc]: @ref gapi_imgproc
[Architecture]: @ref gapi_hld
[Kernels]: @ref gapi_kernel_api
[VAAPI]: https://01.org/vaapi
