#ifndef OPENCV_DNN_NPY_BLOB_HPP
#define OPENCV_DNN_NPY_BLOB_HPP

#include <opencv2/core.hpp>
#include <string>

namespace cv {
namespace dnn {

/** @brief Reads a NumPy array (*.npy) from file and returns it as a cv::Mat.
 *  @param path Path to the .npy file.
 *  @return cv::Mat containing the loaded array.
 *
 *  The function supports the following NumPy data types:
 *  - <f4 (float32)
 *  - <i4 (int32)
 *  - <i8 (int64)
 *
 *  Note: Only C-style ordering (fortran_order=False) is supported.
 */
CV_EXPORTS_W Mat blobFromNPY(const String& path);

}} // namespace cv::dnn

#endif // OPENCV_DNN_NPY_BLOB_HPP