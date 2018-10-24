# Kernel API {#gapi_kernel_api}

[TOC]

# G-API Kernel API

The core idea behind G-API is portability -- a pipeline built with
G-API must be portable (or at least able to be portable). It means
that either it works out-of-the box when compiled for new platform,
_or_ G-API provides necessary tools to make it running there, with
little-to-no changes in the algorithm itself.

This idea can be achieved by separating kernel interface from its
implementation. Once a pipeline is built using kernel interfaces, it
becomes implementation-neutral -- the implementation details
(i.e. which kernels to use) are passed on a separate stage (graph
compilation).

Kernel-implementation hierarchy may look like:

![Kernel API/implementation hierarchy example](pics/kernel_hierarchy.png)

A pipeline itself then can be expressed only in terms of `A`, `B`, and
so on, and choosing which implementation to use in execution becomes
an external parameter.

# Defining a kernel {#gapi_defining_kernel}

G-API provides a macro to define a new kernel interface --
G_TYPED_KERNEL():

@snippet modules/gapi/samples/kernel_api_snippets.cpp filter2d_api

This macro is a shortcut to a new type definition. It takes three
arguments to register a new type, and requires type body to be present
(see [below](@ref gapi_kernel_supp_info)). The macro arguments are:
1. Kernel interface name -- also serves as a name of new type defined
   with this macro;
2. Kernel signature -- an `std::function<>`-like signature which defines
   API of the kernel;
3. Kernel's unique name -- used to identify kernel when its type
   informattion is stripped within the system.

Kernel declaration may be seen as function declaration -- in both cases
a new entity must be used then according to the way it was defined.

Kernel signature defines kernel's usage syntax --  which parameters
it takes during graph construction. Implementations can also use this
signature to derive it into backend-specific callback signatures (see
next chapter).

Kernel may accept values of any type, and G-API _dynamic_ types are
handled in a special way. All other types are opaque to G-API and
passed to kernel in `outMeta()` or in execution callbacks as-is.

Kernel's return value can _only_ be of G-API dynamic type -- cv::GMat,
cv::GScalar, or cv::GArray<T>. If an operation has more than one output,
it should be wrapped into an `std::tuple<>` (which can contain only
mentioned G-API types). Arbitrary-output-number operations are not
supported.

Once a kernel is defined, it can be used in pipelines with special,
G-API-supplied method "::on()". This method has the same signature as
defined in kernel, so this code:

@snippet modules/gapi/samples/kernel_api_snippets.cpp filter2d_on

is a perfectly legal construction. This example has some verbosity,
though, so usually a kernel declaration comes with a C++ function
wrapper ("factory method") which enables optional parameters, more
compact syntax, Doxygen comments, etc:

@snippet modules/gapi/samples/kernel_api_snippets.cpp filter2d_wrap

so now it can be used like:

@snippet modules/gapi/samples/kernel_api_snippets.cpp filter2d_wrap_call

# Extra information {#gapi_kernel_supp_info}

In the current version, kernel declaration body (everything within the
curly braces) must contain a static function `outMeta()`. This function
establishes a functional dependency between operation's input and
output metadata.

_Metadata_ is an information about data kernel operates on. Since
non-G-API types are opaque to G-API, G-API cares only about `G*` data
descriptors (i.e. dimensions and format of cv::GMat, etc).

`outMeta()` is also an example of how kernel's signature can be
transformed into a derived callback -- note that in this example,
`outMeta()` signature exactly follows the kernel signature (defined
within the macro) but is different -- where kernel expects cv::GMat,
`outMeta()` takes and returns cv::GMatDesc (a G-API structure metadata
for cv::GMat).

The point of `outMeta()` is to propagate metadata information within
computation from inputs to outputs and infer metadata of internal
(intermediate, temporary) data objects. This information is required
for further pipeline optimizations, memory allocation, and other
operations done by G-API framework during graph compilation.

<!-- TODO add examples -->

# Implementing a kernel {#gapi_kernel_implementing}

Once a kernel is declared, its interface can be used to implement
versions of this kernel in different backends. This concept is
naturally projected from object-oriented programming
"Interface/Implementation" idiom: an interface can be implemented
multiple times, and different implementations of a kernel should be
substitutable with each other without breaking the algorithm
(pipeline) logic (Liskov Substitution Principle).

Every backend defines its own way to implement a kernel interface.
This way is regular, though -- whatever plugin is, its kernel
implementation must be "derived" from a kernel interface type.

Kernel implementation are then organized into _kernel
packages_. Kernel packages are passed to cv::GComputation::compile()
as compile arguments, with some hints to G-API on how to select proper
kernels (see more on this in "Heterogeneity"[TBD]).

For example, the aforementioned `Filter2D` is implemented in
"reference" CPU (OpenCV) plugin this way (*NOTE* -- this is a
simplified form with improper border handling):

@snippet modules/gapi/samples/kernel_api_snippets.cpp filter2d_ocv

Note how CPU (OpenCV) plugin has transformed the original kernel
signature:
- Input cv::GMat has been substituted with cv::Mat, holding actual input
  data for the underlying OpenCV function call;
- Output cv::GMat has been transformed into extra output parameter, thus
  `GCPUFilter2D::run()` takes one argument more than the original
  kernel signature.

The basic intuition for kernel developer here is _not to care_ where
that cv::Mat objects come from instead of the original cv::GMat -- and
just follow the signature conventions defined by the plugin. G-API
will call this method during execution and supply all the necessary
information (and forward the original opaque data as-is).

# Compound kernels

Sometimes kernel is a single thing only on API level. It is convenient
for users, but on a particular  implementation side it would be better to
have multiple kernels (a subgraph) doing the thing instead. An example
is goodFeaturesToTrack() -- while in OpenCV backend it may remain a
single kernel, with Fluid it becomes compound -- Fluid can handle Harris
response calculation but can't do sparse non-maxima suppression and
point extraction to an STL vector:

<!-- PIC -->

A compound kernel _implementation_ can be defined using a generic
macro GAPI_COMPOUND_KERNEL():

@snippet modules/gapi/samples/kernel_api_snippets.cpp compound

<!-- TODO: ADD on how Compound kernels may simplify dispatching -->
<!-- TODO: Add details on when expand() is called! -->

It is important to distinguish a compound kernel from G-API high-order
function, i.e. a C++ function which looks like a kernel but in fact
generates a subgraph. The core difference is that a compound kernel is
an _implementation detail_ and a kernel implementation may be either
compound or not (depending on backend capabilities), while a
high-order function is a "macro" in terms of G-API and so cannot act as
an interface which then needs to be implemented by a backend.
