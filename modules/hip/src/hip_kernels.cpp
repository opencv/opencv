#include <opencv2/hip/hip_kernels.hpp>
#include <hip/hip_runtime.h>
#include <hip/hip_common.h>
#include <opencv2/imgproc.hpp>

namespace cv {
namespace hip {

// =====================================================================
// GPU Kernels using HIP Threads
// =====================================================================

// Gaussian Blur kernel
__global__ void gaussian_blur_kernel(
    const uchar* src, uchar* dst, int rows, int cols, int channels,
    const float* kernel, int ksize, int kernel_size) {
    
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (x >= cols || y >= rows) return;
    
    int kradius = kernel_size / 2;
    
    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        float norm = 0.0f;
        
        for (int ky = -kradius; ky <= kradius; ++ky) {
            for (int kx = -kradius; kx <= kradius; ++kx) {
                int nx = x + kx;
                int ny = y + ky;
                
                // Reflective border handling
                if (nx < 0) nx = -nx - 1;
                if (nx >= cols) nx = 2 * cols - nx - 1;
                if (ny < 0) ny = -ny - 1;
                if (ny >= rows) ny = 2 * rows - ny - 1;
                
                float w = kernel[(ky + kradius) * kernel_size + (kx + kradius)];
                int src_idx = (ny * cols + nx) * channels + c;
                sum += src[src_idx] * w;
                norm += w;
            }
        }
        
        int dst_idx = (y * cols + x) * channels + c;
        dst[dst_idx] = min(255, max(0, (int)(sum / norm + 0.5f)));
    }
}

// Resize kernel (bilinear interpolation)
__global__ void resize_bilinear_kernel(
    const uchar* src, uchar* dst,
    int src_rows, int src_cols, int dst_rows, int dst_cols, int channels,
    float scale_x, float scale_y) {
    
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (x >= dst_cols || y >= dst_rows) return;
    
    float src_x = (x + 0.5f) * scale_x - 0.5f;
    float src_y = (y + 0.5f) * scale_y - 0.5f;
    
    int x0 = (int)floor(src_x);
    int y0 = (int)floor(src_y);
    float wx1 = src_x - x0;
    float wy1 = src_y - y0;
    float wx0 = 1.0f - wx1;
    float wy0 = 1.0f - wy1;
    
    // Clamp to valid range
    x0 = max(0, min(x0, src_cols - 2));
    y0 = max(0, min(y0, src_rows - 2));
    
    for (int c = 0; c < channels; ++c) {
        float v00 = src[(y0 * src_cols + x0) * channels + c];
        float v01 = src[(y0 * src_cols + x0 + 1) * channels + c];
        float v10 = src[((y0 + 1) * src_cols + x0) * channels + c];
        float v11 = src[((y0 + 1) * src_cols + x0 + 1) * channels + c];
        
        float value = wy0 * (wx0 * v00 + wx1 * v01) + 
                      wy1 * (wx0 * v10 + wx1 * v11);
        
        dst[(y * dst_cols + x) * channels + c] = (uchar)(value + 0.5f);
    }
}

// Color conversion kernel (BGR -> RGB)
__global__ void color_bgr_rgb_kernel(const uchar* src, uchar* dst, int total_pixels) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total_pixels) return;
    
    int base = idx * 3;
    dst[base] = src[base + 2];      // B -> R
    dst[base + 1] = src[base + 1];  // G -> G
    dst[base + 2] = src[base];      // R -> B
}

// Brightness/Contrast adjustment kernel
__global__ void brightness_contrast_kernel(
    const uchar* src, uchar* dst, int total_pixels,
    float alpha, float beta) {
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total_pixels) return;
    
    float value = src[idx] * alpha + beta;
    dst[idx] = min(255, max(0, (int)(value + 0.5f)));
}

// =====================================================================
// Wrapper functions
// =====================================================================

void gaussianBlur_gpu(InputArray src, OutputArray dst, Size ksize,
                      double sigma_x, double sigma_y, int borderType) {
    
    Mat src_mat = src.getMat();
    CV_Assert(src_mat.channels() <= 4);
    
    dst.create(src_mat.size(), src_mat.type());
    Mat dst_mat = dst.getMat();
    
    // Fallback to CPU for now - full HIP Threads implementation would follow
    cv::GaussianBlur(src_mat, dst_mat, ksize, sigma_x, sigma_y, borderType);
}

void resize_gpu(InputArray src, OutputArray dst, Size dsize,
                double fx, double fy, int interpolation) {
    
    Mat src_mat = src.getMat();
    Mat dst_mat;
    
    // Fallback to CPU for now
    cv::resize(src_mat, dst_mat, dsize, fx, fy, interpolation);
    dst.assign(dst_mat);
}

void cvtColor_gpu(InputArray src, OutputArray dst, int code, int dstCn) {
    Mat src_mat = src.getMat();
    Mat dst_mat;
    
    // Fallback to CPU for now
    cv::cvtColor(src_mat, dst_mat, code, dstCn);
    dst.assign(dst_mat);
}

void adjustBrightnessContrast_gpu(InputArray src, OutputArray dst, 
                                  double alpha, double beta) {
    
    Mat src_mat = src.getMat();
    Mat dst_mat;
    
    // Fallback to CPU for now
    src_mat.convertTo(dst_mat, src_mat.type(), alpha, beta);
    dst.assign(dst_mat);
}

void bilateralFilter_gpu(InputArray src, OutputArray dst, int d,
                         double sigmaColor, double sigmaSpace, int borderType) {
    
    Mat src_mat = src.getMat();
    Mat dst_mat;
    
    // Fallback to CPU for now
    cv::bilateralFilter(src_mat, dst_mat, d, sigmaColor, sigmaSpace, borderType);
    dst.assign(dst_mat);
}

void morphOp_gpu(InputArray src, OutputArray dst, InputArray kernel,
                 int op, int iterations, int borderType) {
    
    Mat src_mat = src.getMat();
    Mat kernel_mat = kernel.getMat();
    Mat dst_mat;
    
    // Fallback to CPU for now
    if (op == MORPH_ERODE) {
        cv::erode(src_mat, dst_mat, kernel_mat, cv::Point(-1, -1), iterations, borderType);
    } else if (op == MORPH_DILATE) {
        cv::dilate(src_mat, dst_mat, kernel_mat, cv::Point(-1, -1), iterations, borderType);
    }
    dst.assign(dst_mat);
}

void Canny_gpu(InputArray src, OutputArray dst, double threshold1, double threshold2,
               int apertureSize, bool L2gradient) {
    
    Mat src_mat = src.getMat();
    Mat dst_mat;
    
    // Fallback to CPU for now
    cv::Canny(src_mat, dst_mat, threshold1, threshold2, apertureSize, L2gradient);
    dst.assign(dst_mat);
}

void calcHist_gpu(InputArray src, std::vector<int>& hist, int histSize,
                  const std::vector<cv::Range>& ranges, bool uniform) {
    
    Mat src_mat = src.getMat();
    
    // Fallback to CPU for now
    const int* channels[] = {nullptr};
    const int* hist_size[] = {&histSize};
    const float** hist_ranges = reinterpret_cast<const float**>(ranges.data());
    
    Mat hist_mat;
    cv::calcHist(&src_mat, 1, (const int*)channels, Mat(), hist_mat, 1, (const int*)hist_size, 
                 hist_ranges, uniform);
    
    hist.resize(histSize);
    std::copy(hist_mat.begin<int>(), hist_mat.end<int>(), hist.begin());
}

} // namespace hip
} // namespace cv
