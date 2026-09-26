// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "depth_to_3d.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv
{

/** If the input image is of type CV_16UC1 (like the Kinect one), the image is converted to floats, divided
 * by 1000 to get a depth in meters, and the values 0 are converted to std::numeric_limits<float>::quiet_NaN()
 * Otherwise, the image is simply converted to floats
 * @param in_in the depth image (if given as short int CV_U, it is assumed to be the depth in millimeters
 *              (as done with the Microsoft Kinect), it is assumed in meters)
 * @param depth the desired output depth (floats or double)
 * @param out_out The rescaled float depth image
 */
void rescaleDepth(InputArray in_in, int type, OutputArray out_out, double depth_factor)
{
    cv::Mat in = in_in.getMat();
    CV_Assert(in.type() == CV_64FC1 || in.type() == CV_32FC1 || in.type() == CV_16UC1 || in.type() == CV_16SC1);
    CV_Assert(type == CV_64FC1 || type == CV_32FC1);

    int in_depth = in.depth();

    out_out.create(in.size(), type);
    cv::Mat out = out_out.getMat();
    if (in_depth == CV_16U)
    {
      in.convertTo(out, type, 1 / depth_factor); //convert to float so that it is in meters
      cv::Mat valid_mask = in == std::numeric_limits<ushort>::min(); // Should we do std::numeric_limits<ushort>::max() too ?
      out.setTo(std::numeric_limits<float>::quiet_NaN(), valid_mask); //set a$
    }
    if (in_depth == CV_16S)
    {
      in.convertTo(out, type, 1 / depth_factor); //convert to float so tha$
      cv::Mat valid_mask = (in == std::numeric_limits<short>::min()) | (in == std::numeric_limits<short>::max()); // Should we do std::numeric_limits<ushort>::max() too ?
      out.setTo(std::numeric_limits<float>::quiet_NaN(), valid_mask); //set a$
    }
    if ((in_depth == CV_32F) || (in_depth == CV_64F))
      in.convertTo(out, type);
}


/**
 * @param K
 * @param depth the depth image
 * @param mask the mask of the points to consider (can be empty)
 * @param points3d the resulting 3d points, a 3-channel matrix
 */
static void depthTo3d_from_uvz(const cv::Mat& in_K, const cv::Mat& u_mat, const cv::Mat& v_mat, const cv::Mat& z_mat,
                               cv::Mat& points3d)
{
    CV_Assert((u_mat.size() == z_mat.size()) && (v_mat.size() == z_mat.size()));
    if (u_mat.empty())
        return;
    CV_Assert((u_mat.type() == z_mat.type()) && (v_mat.type() == z_mat.type()));

    //grab camera params
    cv::Mat_<float> K;

    if (in_K.depth() == CV_32F)
        K = in_K;
    else
        in_K.convertTo(K, CV_32F);

    float fx = K(0, 0);
    float fy = K(1, 1);
    float s = K(0, 1);
    float cx = K(0, 2);
    float cy = K(1, 2);

    std::vector<cv::Mat> coordinates(4);

    coordinates[0] = (u_mat - cx) / fx;

    if (s != 0)
        coordinates[0] = coordinates[0] + (-(s / fy) * v_mat + cy * s / fy) / fx;

    coordinates[0] = coordinates[0].mul(z_mat);
    coordinates[1] = (v_mat - cy).mul(z_mat) * (1. / fy);
    coordinates[2] = z_mat;
    coordinates[3] = Mat(u_mat.size(), CV_32F, Scalar(0));
    cv::merge(coordinates, points3d);
}

/**
 * @param K
 * @param depth the depth image
 * @param mask the mask of the points to consider (can be empty)
 * @param points3d the resulting 3d points
 */
static void depthTo3dMask(const cv::Mat& depth, const cv::Mat& K, const cv::Mat& mask, cv::Mat& points3d)
{
    // Create 3D points in one go.
    cv::Mat_<float> u_mat, v_mat, z_mat;

    cv::Mat_<uchar> uchar_mask = mask;

    if (mask.depth() != (CV_8U))
        mask.convertTo(uchar_mask, CV_8U);

    // Figure out the interesting indices
    size_t n_points;

    if (depth.depth() == CV_16U)
        n_points = convertDepthToFloat<ushort>(depth, mask, 1.0f / 1000.0f, u_mat, v_mat, z_mat);
    else if (depth.depth() == CV_16S)
        n_points = convertDepthToFloat<short>(depth, mask, 1.0f / 1000.0f, u_mat, v_mat, z_mat);
    else
    {
        CV_Assert(depth.type() == CV_32F);
        n_points = convertDepthToFloat<float>(depth, mask, 1.0f, u_mat, v_mat, z_mat);
    }

    if (n_points == 0)
        return;

    u_mat.resize(n_points);
    v_mat.resize(n_points);
    z_mat.resize(n_points);

    depthTo3d_from_uvz(K, u_mat, v_mat, z_mat, points3d);
    points3d = points3d.reshape(4, 1);
}

/**
 * @param K
 * @param depth the depth image
 * @param points3d the resulting 3d points
 */
#if (CV_SIMD || CV_SIMD_SCALABLE)
// Interleaved 4-channel store: on RVV the portable v_store_interleave lowers to
// 4x strided stores (vsse), which are much slower than scalar sequential stores
// on in-order cores - use the native vsseg4 segmented store (one instruction
// writes the AoS block) there. Other backends keep the portable path.
template<typename T, typename VT>
static inline void storePoints4(Vec<T, 4>* p, const VT& X, const VT& Y,
                                const VT& Z, const VT& W, int vl)
{
#if defined(__riscv_vector)
    if constexpr (sizeof(T) == 4)
        __riscv_vsseg4e32_v_f32m1x4(reinterpret_cast<float*>(p),
                                    __riscv_vcreate_v_f32m1x4(X, Y, Z, W), vl);
    else
        __riscv_vsseg4e64_v_f64m1x4(reinterpret_cast<double*>(p),
                                    __riscv_vcreate_v_f64m1x4(X, Y, Z, W), vl);
#else
    v_store_interleave(reinterpret_cast<T*>(p), X, Y, Z, W);
#endif
    (void)vl;
}

/** One row of the dense no-mask path, universal-intrinsics vectorized.
 * Per output element: p0 = x_cache*z, p1 = y*z, p2 = z, p3 = 0 - a single IEEE
 * multiply each (no accumulation, no reassociation), so the vector body is
 * bit-identical to the scalar reference for every input, NaN payloads included
 * where the hardware preserves them the same way for scalar and vector mul. */
template<typename T>
static void depthTo3dRowVec(const T* x_cache, const T* z, T y_val, Vec<T, 4>* point, int n)
{
    typedef decltype(vx_load(static_cast<const T*>(nullptr))) VT;
    const int vl = VTraits<VT>::vlanes();
    const VT yv = v_setall_<VT>(y_val);
    const VT zero = v_setzero_<VT>();
    int i = 0;
    for (; i + vl <= n; i += vl)
    {
        VT zv = vx_load(z + i);
        VT xv = vx_load(x_cache + i);
        storePoints4<T>(point + i, v_mul(xv, zv), v_mul(yv, zv), zv, zero, vl);
    }
    for (; i < n; ++i)
    {
        T zz = z[i];
        point[i][0] = x_cache[i] * zz;
        point[i][1] = y_val * zz;
        point[i][2] = zz;
        point[i][3] = 0;
    }
}
#endif

template<typename T>
void depthTo3dNoMask(const cv::Mat& in_depth, const cv::Mat_<T>& K, cv::Mat& points3d)
{
    const T inv_fx = T(1) / K(0, 0);
    const T inv_fy = T(1) / K(1, 1);
    const T ox = K(0, 2);
    const T oy = K(1, 2);

    // Build z
    cv::Mat_<T> z_mat;
    if (z_mat.depth() == in_depth.depth())
        z_mat = in_depth;
    else
        rescaleDepthTemplated<T>(in_depth, z_mat);

    // Pre-compute some constants
    cv::Mat_<T> x_cache(1, in_depth.cols), y_cache(in_depth.rows, 1);
    T* x_cache_ptr = x_cache[0], * y_cache_ptr = y_cache[0];
    for (int x = 0; x < in_depth.cols; ++x, ++x_cache_ptr)
        *x_cache_ptr = (x - ox) * inv_fx;
    for (int y = 0; y < in_depth.rows; ++y, ++y_cache_ptr)
        *y_cache_ptr = (y - oy) * inv_fy;

    y_cache_ptr = y_cache[0];
    for (int y = 0; y < in_depth.rows; ++y, ++y_cache_ptr)
    {
        cv::Vec<T, 4>* point = points3d.ptr<cv::Vec<T, 4> >(y);
        const T* depth = z_mat[y];
#if (CV_SIMD || CV_SIMD_SCALABLE)
        depthTo3dRowVec(x_cache[0], depth, *y_cache_ptr, point, in_depth.cols);
#else
        const T* x_cache_ptr_end = x_cache[0] + in_depth.cols;
        for (x_cache_ptr = x_cache[0]; x_cache_ptr != x_cache_ptr_end; ++x_cache_ptr, ++point, ++depth)
        {
            T z = *depth;
            (*point)[0] = (*x_cache_ptr) * z;
            (*point)[1] = (*y_cache_ptr) * z;
            (*point)[2] = z;
            (*point)[3] = 0;
        }
#endif
    }
}

///////////////////////////////////////////////////////////////////////////////

/**
 * @param K
 * @param depth the depth image
 * @param u_mat the list of x coordinates
 * @param v_mat the list of matching y coordinates
 * @param points3d the resulting 3d points
 */
void depthTo3dSparse(InputArray depth_in, InputArray K_in, InputArray points_in, OutputArray points3d_out)
{
    // Make sure we use foat types
    cv::Mat points = points_in.getMat();
    cv::Mat depth = depth_in.getMat();

    cv::Mat points_float;
    if (points.depth() != CV_32F)
        points.convertTo(points_float, CV_32FC2);
    else
        points_float = points;

    // Fill the depth matrix
    cv::Mat_<float> z_mat;

    if (depth.depth() == CV_16U)
        convertDepthToFloat<ushort>(depth, 1.0f / 1000.0f, points_float, z_mat);
    else if (depth.depth() == CV_16U)
        convertDepthToFloat<short>(depth, 1.0f / 1000.0f, points_float, z_mat);
    else
    {
        CV_Assert(depth.type() == CV_32F);
        convertDepthToFloat<float>(depth, 1.0f, points_float, z_mat);
    }

    std::vector<cv::Mat> channels(2);
    cv::split(points_float, channels);

    points3d_out.create(channels[0].rows, channels[0].cols, CV_32FC4);
    cv::Mat points3d = points3d_out.getMat();
    depthTo3d_from_uvz(K_in.getMat(), channels[0], channels[1], z_mat, points3d);
}

/**
 * @param depth the depth image (if given as short int CV_U, it is assumed to be the depth in millimeters
 *              (as done with the Microsoft Kinect), otherwise, if given as CV_32F, it is assumed in meters)
 * @param K The calibration matrix
 * @param points3d the resulting 3d points. They are of depth the same as `depth` if it is CV_32F or CV_64F, and the
 *        depth of `K` if `depth` is of depth CV_U
 * @param mask the mask of the points to consider (can be empty)
 */
void depthTo3d(InputArray depth_in, InputArray K_in, OutputArray points3d_out, InputArray mask_in)
{
    cv::Mat depth = depth_in.getMat();
    cv::Mat K = K_in.getMat();
    cv::Mat mask = mask_in.getMat();
    CV_Assert(K.cols == 3 && K.rows == 3 && (K.depth() == CV_64F || K.depth() == CV_32F));
    CV_Assert(depth.type() == CV_64FC1 || depth.type() == CV_32FC1 || depth.type() == CV_16UC1 || depth.type() == CV_16SC1);
    CV_Assert(mask.empty() || mask.channels() == 1);

    cv::Mat K_new;
    K.convertTo(K_new, depth.depth() == CV_64F ? CV_64F : CV_32F); // issue #1021

    // Create 3D points in one go.
    if (!mask.empty())
    {
        cv::Mat points3d;
        depthTo3dMask(depth, K_new, mask, points3d);
        points3d_out.create(points3d.size(), CV_MAKETYPE(K_new.depth(), 4));
        points3d.copyTo(points3d_out.getMat());
    }
    else
    {
        points3d_out.create(depth.size(), CV_MAKETYPE(K_new.depth(), 4));
        cv::Mat points3d = points3d_out.getMat();
        if (K_new.depth() == CV_64F)
            depthTo3dNoMask<double>(depth, K_new, points3d);
        else
            depthTo3dNoMask<float>(depth, K_new, points3d);
    }
}

}
