# Why Graph API? {#gapi_purposes}

# Motivation behind G-API {#gapi_intro_why}

G-API module brings graph-based model of execution to OpenCV. This
chapter briefly describes how this new model can help software
developers in two aspects: optimizing and porting image processing
algorithms.

## Optimizing with Graph API {#gapi_intro_opt}

Traditionally OpenCV provided a lot of stand-alone image processing
functions  (see modules `core` and `imgproc`). Many of that functions
are well-optimized (e.g. vectorized for specific CPUs, parallel, etc)
but still the out-of-box optimization scope has been limited to a
single function only -- optimizing the whole algorithm built atop of that
functions was a responsibility of a programmer.

OpenCV 3.0 introduced _Transparent API_ (or _T-API_) which allowed to
offload OpenCV function calls transparently to OpenCL devices and save
on Host/Device data transfers with cv::UMat -- and it was a great step
forward. However, T-API is a dynamic API -- user code still remains
unconstrained and OpenCL kernels are enqueued in arbitrary order, thus
eliminating further pipeline-level optimization potential.

G-API brings implicit graph model to OpenCV 4.0. Graph model captures
all operations and its data dependencies in a pipeline and so provides
G-API framework with extra information to do pipeline-level
optimizations.

The cornerstone of graph-based optimizations is _Tiling_. Tiling
allows to break the processing into smaller parts and reorganize
operations to enable data parallelism, improve data locality, and save
memory footprint. Data locality is an especially important aspect of
software optimization due to diffent costs of memory access on modern
computer architectures -- the more data is reused in the first level
cache, the more efficient pipeline is.

Definitely the aforementioned techinques can be applied manually --
but it requires extra skills and knowledge of the target platform and
the algorithm implementation changes irrevocably -- becoming more
specific, less flexible, and harder to extend and maintain.

G-API takes this responsibility and complexity from user and does the
majority of the work by itself, keeping the algorithm code clean from
device or optimization details. This approach has its own limitations,
though, as graph model is a _constrained_ model and not every
algorithm can be represented as a graph, so the G-API scope is limited
only to regular image processing -- various filters, arithmentic,
binary operations, and well-defined geometrical transformations.

## Porting with Graph API {#gapi_intro_port}

The essense of G-API is declaring a sequence of operations to run, and
then executing that sequence. G-API is a constrained API, so it puts a
number of limitations on which operations can form a pipeline and
which data these operations may exchange each other.

This formalization in fact helps to make an algorithm portable. G-API
clearly separates operation _interfaces_ from its _implementations_.

One operation (_kernel_) may have multiple implementations even for a
single device (e.g., OpenCV-based "reference" implementation and a
tiled optimized implementation, both running on CPU). Graphs (or
_Computations_ in G-API terms) are built only using operation
interfaces, not implementations -- thus the same graph can be executed
on different devices (and, of course, using different optimization
techniques) with little-to-no changes in the graph itself.

G-API supports plugins (_Backends_) which aggreate logic and
intelligence on what is the best way to execute on a particular
platform. Once a pipeline is built with G-API, it can be parametrized
to use either of the backends (or a combination of it) and so a graph
can be ported easily to a new platform.

@sa @ref gapi_hld
