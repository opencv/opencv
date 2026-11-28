#error This is a shadow header file, which is not intended for processing by any compiler. \
       Only bindings parser should handle this file.

namespace cv
{

//! cv::theRNG() returns a reference to the thread-local generator, which can not be represented
//! in Python. The binding is implemented on top of cv::theRNGPtr() instead, see cv_theRNG()
//! in modules/core/misc/python/pyopencv_core.hpp
CV_WRAP_PHANTOM(Ptr<RNG> theRNG());

} // namespace cv
