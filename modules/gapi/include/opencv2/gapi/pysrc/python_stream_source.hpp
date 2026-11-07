#ifndef OPENCV_GAPI_PYSRC_PYTHONSTREAMSOURCE_HPP
#define OPENCV_GAPI_PYSRC_PYTHONSTREAMSOURCE_HPP
#include <opencv2/gapi/streaming/source.hpp>
#include <opencv2/core.hpp>

namespace cv {
namespace gapi {
namespace wip {

/**
 * @brief Creates a G-API IStreamSource that delegates to a Python-defined source.
 *
 * This factory function wraps a Python object (for example, an instance of a class
 * implementing a `pull()` and a `descr_of()` method) into a `cv::gapi::wip::IStreamSource`,
 * enabling it to be used within a G-API computation graph. The OpenCV Python bindings
 * automatically convert the PyObject into a `cv::Ptr<IStreamSource>`.
 *
 * @param src
 * A `cv::Ptr<IStreamSource>` that internally holds the original Python object.
 *
 * @return
 * A `cv::Ptr<IStreamSource>` that wraps the provided Python object. On each frame pull,
 * G-API will:
 *   - Acquire the Python GIL
 *   - Call the Python object’s `pull()` method
 *   - Convert the resulting NumPy array to a `cv::Mat`
 *   - Pass the `cv::Mat` into the G-API pipeline
 *
 * @note
 * In Python, you can use the returned `make_py_src` as follows:
 *
 * @code{.py}
 * class MyClass:
 *     def __init__(self):
 *         # Initialize your source
 *     def pull(self):
 *         # Return the next frame as a numpy.ndarray or None for end-of-stream
 *     def descr_of(self):
 *         # Return a numpy.ndarray that describes the format of the frames
 *
 * # Create a G-API source from a Python class
 * py_src = cv.gapi.wip.make_py_src(MyClass())
 *
 * # Define a simple graph: input → copy → output
 * g_in = cv.GMat()
 * g_out = cv.gapi.copy(g_in)
 * graph = cv.GComputation(g_in, g_out)
 *
 * # Compile the pipeline for streaming and assign the source
 * pipeline = graph.compileStreaming()
 * pipeline.setSource([py_src])
 * pipeline.start()
 * @endcode
 */

CV_EXPORTS_W cv::Ptr<cv::gapi::wip::IStreamSource>
make_py_src(const cv::Ptr<cv::gapi::wip::IStreamSource>& src);


} // namespace wip
} // namespace gapi
} // namespace cv


#endif // OPENCV_GAPI_PYSRC_PYTHONSTREAMSOURCE_HPP
