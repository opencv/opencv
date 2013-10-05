opencv-raw
==========

Raw Haskell bindings to OpenCV >= 2.0


Compilation / Installation
------------

Compiling this package might be a little tricky at the moment since I've only had the chance to test it on my machine
so far. First, you need to generate C wrappers for the version of OpenCV on your machine -- this repo holds the wrappers
for OpenCV 2.4.6 only. You can generate these wrappers (and the corresponding Haskell bindings) via:

    ./setup.sh <path to opencv headers>
      
e.g.

    ./setup.sh /usr/local/include
      
You want the include directory that contains the opencv2 directory at the top level.

If that goes off without a hitch, compilation *should* be as easy as:

    cabal build --with-gcc=<c++ compiler>
      
Cabal doesn't have a way of specifying that the library in use is a C++ library and not a C library, so it tries to use
gcc for everything. But of course, you can't compile this project without a C++ compiler handy. On my machine the
following is sufficient:

    cabal build --with-gcc=g++
      
The build currently doesn't work with GHC 7.6 and clang++ because hsc2hs doesn't insert a cast in a generated c file.
This issue has been fixed in GHC HEAD, so this issue should disappear with the release of 7.8. Similarly, the repl command 
doesn't work with this library and GHC 7.6 due to a longstanding bug with ghci and dynamically linked libraries. This has also 
been fixed in HEAD.

Check [revelation](https://github.com/arjuncomar/revelation) for a library built on these raw bindings.

C Wrappers
----------

The C wrappers are generated in genc.py via a parsed header representation provided by
hdr_parser.py, borrowed from the official opencv repository and slightly tweaked. It is
this parser that actually makes this project possible, though a more complete approach
(perhaps via gcc-xml) to actually parse the C++ syntax tree and transform it into
C wrappers would probably be better. 

These wrappers are not meant to be an API to the programmer, 
but rather hooks for other languages like Haskell that can call C via an FFI but cannot call
C++ directly due to name mangling. For example, C++ objects are passed around as raw pointers, 
and memory management is not handled by these wrappers. The intention is that the binding platform 
will hook malloc and free to do memory management correctly -- the Haskell runtime provided by GHC 
does a wonderful job of this. Moreover, copying is kept to a minimum as much as possible, and 
objects are exposed primarily via their methods. 

The transformation from C++ method to C wrapper function is as follows:

    /RetType/ /Classname/::/methodCall/(/args.../) -> /RetType/ cv_/Classname/_/methodCall/(/Classname/* self, args...)
  
Constructors are provided similarly as:

    Classname* cv_create_/Classname/(/args.../)
  
And #define constants are transformed as follows:

    cv::/module/::/CONSTANT/ -> CV_/MODULE/_/CONSTANT/0
  
The 0 is appended because some (but not all) of these constants are exported with the same name by OpenCV already,
and I wanted to avoid the warnings. A more thorough approach would track these constants and drop the ones
that are already defined this way in the headers.

The automatically generated portions of these wrappers are provided in opencv_generated.hpp and
opencv_generated.cpp for headers and source respectively. Some major classes are not parsed
correctly by hdr_parser.py so these types are manually wrapped and show up in explicitly named
headers and source files (e.g. mat.hpp/mat.cpp). 

I hope to get these wrapped functions into the OpenCV library proper because I imagine many people would find
them useful. Most languages can interoperate with C, but very few can interoperate directly with C++. 

Classes are not converted to structs and instead the wrapper deals with them as opaque pointers. Again, the
intention is not to create a fully featured C API. Instead, the API is made more pleasant and usable from
the Haskell side. When making bindings available through these wrappers, a similar approach should be employed.

Haskell Bindings
----------------

The low level bindings are provided via bindings-dsl macros that call the C functions directly. The C -> Haskell
translation is as follows:

    Ret* cv_wrapped_function(Arg1* arg1, Arg2 arg2, int arg3...) -> 
    c'cv_wrapped_function :: Ptr C'Arg1 -> C'Arg2 -> CInt ... -> IO (Ptr Ret)
    
Similar to the C API, these functions are not meant to be used directly. They're all stuck in IO and provide a rather
inconvenient API. The programming model feels a little worse than it would trying to use those wrapped functions in C.
Instead, a higher level API is exposed by this module (under active development). Function pointers are also available
as 
    p'cv_wrapped_function
    
Types are exposed primarily as opaque pointers that you can interact with via the provided functions. Classes are transformed
into types as:

    /Classname/ -> C'/Classname/
    
Beyond that, there's not much to say. The same caveats apply as with the C bindings. They are not intended for real use.

