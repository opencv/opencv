#ifndef OPENCV_HIP_KERNELS_HPP
#define OPENCV_HIP_KERNELS_HPP

#include <opencv2/core.hpp>
#include "hip_dispatcher.hpp"

namespace cv {
namespace hip {

/**
 * @brief GPU-accelerated Gaussian Blur
 * 
 * Processes image data on GPU with automatic CPU fallback for small inputs.
 * Optimal for images > 1MB with standard kernels (3x3 to 31x31).
 * 
 * @param src Input image (CV_8U or CV_32F, 1-4 channels)
 * @param dst Output image (same size and type as src)
 * @param ksize Gaussian kernel size (odd, 3-31)
 * @param sigma Gaussian standard deviation
 * @param borderType Border handling mode
 */
CV_EXPORTS void gaussianBlur_gpu(InputArray src, OutputArray dst, Size ksize, 
                                  double sigma_x = 0.0, double sigma_y = 0.0,
                                  int borderType = BORDER_REFLECT_101);

/**
 * @brief GPU-accelerated image resize
 * 
 * Uses HIP Threads for parallel resampling. Efficient for large images.
 * Supports nearest, linear, and cubic interpolation.
 * 
 * @param src Input image
 * @param dst Output image  
 * @param dsize Output size
 * @param fx Horizontal scale factor
 * @param fy Vertical scale factor
 * @param interpolation Interpolation method (INTER_NEAREST, INTER_LINEAR, INTER_CUBIC)
 */
CV_EXPORTS void resize_gpu(InputArray src, OutputArray dst, Size dsize,
                            double fx = 0, double fy = 0,
                            int interpolation = INTER_LINEAR);

/**
 * @brief GPU-accelerated color space conversion
 * 
 * Parallelizes per-pixel color transformation using HIP Threads.
 * Supports RGB↔BGR, RGB↔HSV, RGB↔YCrCb, etc.
 * 
 * @param src Input image
 * @param dst Output image
 * @param code Color conversion code (COLOR_BGR2RGB, etc.)
 * @param dstCn Number of output channels (0 = automatic)
 */
CV_EXPORTS void cvtColor_gpu(InputArray src, OutputArray dst, int code, int dstCn = 0);

/**
 * @brief GPU-accelerated brightness/contrast adjustment
 * 
 * Simple per-pixel linear transformation: out = alpha*in + beta
 * Useful for demonstrating GPU advantages on image-wide operations.
 * 
 * @param src Input image
 * @param dst Output image
 * @param alpha Multiplication factor for brightness
 * @param beta Offset factor for contrast
 */
CV_EXPORTS void adjustBrightnessContrast_gpu(InputArray src, OutputArray dst, 
                                              double alpha = 1.0, double beta = 0.0);

/**
 * @brief GPU-accelerated bilateral filter
 * 
 * Edge-preserving filter using spatial and range Gaussian kernels.
 * More complex operation that better justifies GPU computation overhead.
 * 
 * @param src Input image
 * @param dst Output image
 * @param d Diameter of each pixel neighborhood
 * @param sigmaColor Filter sigma in the color space
 * @param sigmaSpace Filter sigma in the coordinate space
 * @param borderType Border handling mode
 */
CV_EXPORTS void bilateralFilter_gpu(InputArray src, OutputArray dst, int d,
                                     double sigmaColor, double sigmaSpace,
                                     int borderType = BORDER_REFLECT_101);

/**
 * @brief GPU-accelerated morphological operations
 * 
 * Erosion and dilation using structuring elements.
 * Efficient parallel reduction on GPU.
 * 
 * @param src Input image
 * @param dst Output image
 * @param kernel Morphological kernel/structuring element
 * @param op Operation: MORPH_ERODE or MORPH_DILATE
 * @param iterations Number of times to apply operation
 * @param borderType Border handling mode
 */
CV_EXPORTS void morphOp_gpu(InputArray src, OutputArray dst, InputArray kernel,
                             int op, int iterations = 1, int borderType = BORDER_REFLECT_101);

/**
 * @brief GPU-accelerated Canny edge detection
 * 
 * Multi-stage edge detection with GPU-parallel gradient computation.
 * Combines Gaussian blur, gradient calculation, and hysteresis thresholding.
 * 
 * @param src Input image (must be 8-bit grayscale)
 * @param dst Output image (8-bit binary)
 * @param threshold1 Lower threshold for hysteresis
 * @param threshold2 Upper threshold for hysteresis
 * @param apertureSize Sobel kernel size (3, 5, or 7)
 * @param L2gradient Use L2 norm for gradient magnitude
 */
CV_EXPORTS void Canny_gpu(InputArray src, OutputArray dst, double threshold1, double threshold2,
                           int apertureSize = 3, bool L2gradient = false);

/**
 * @brief GPU-accelerated histogram computation
 * 
 * Parallel histogram calculation using atomic operations.
 * Useful for image analysis and equalization preprocessing.
 * 
 * @param src Input image
 * @param hist Output histogram
 * @param histSize Number of bins
 * @param ranges Range boundaries [min, max]
 * @param uniform Uniform bin spacing
 */
CV_EXPORTS void calcHist_gpu(InputArray src, std::vector<int>& hist, int histSize,
                              const std::vector<cv::Range>& ranges, bool uniform = true);

} // namespace hip
} // namespace cv

#endif // OPENCV_HIP_KERNELS_HPP
