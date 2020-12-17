// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_WillowGarage.md file found in this module's directory

#include "precomp.hpp"

namespace cv
{
namespace rgbd
{
  /** Just compute the norm of a vector
   * @param vec a vector of size 3 and any type T
   * @return
   */
  template<typename T>
  T
  inline
  norm_vec(const Vec<T, 3> &vec)
  {
    return std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
  }

  /** Given 3d points, compute their distance to the origin
   * @param points
   * @return
   */
  template<typename T>
  Mat_<T>
  computeRadius(const Mat &points)
  {
    typedef Vec<T, 3> PointT;

    // Compute the
    Size size(points.cols, points.rows);
    Mat_<T> r(size);
    if (points.isContinuous())
      size = Size(points.cols * points.rows, 1);
    for (int y = 0; y < size.height; ++y)
    {
      const PointT* point = points.ptr < PointT > (y), *point_end = points.ptr < PointT > (y) + size.width;
      T * row = r[y];
      for (; point != point_end; ++point, ++row)
        *row = norm_vec(*point);
    }

    return r;
  }

  // Compute theta and phi according to equation 3 of
  // ``Fast and Accurate Computation of Surface Normals from Range Images``
  // by H. Badino, D. Huber, Y. Park and T. Kanade
  template<typename T>
  void
  computeThetaPhi(int rows, int cols, const Matx<T, 3, 3>& K, Mat &cos_theta, Mat &sin_theta,
                  Mat &cos_phi, Mat &sin_phi)
  {
    // Create some bogus coordinates
    Mat depth_image = K(0, 0) * Mat_ < T > ::ones(rows, cols);
    Mat points3d;
    depthTo3d(depth_image, Mat(K), points3d);

    typedef Vec<T, 3> Vec3T;

    cos_theta = Mat_ < T > (rows, cols);
    sin_theta = Mat_ < T > (rows, cols);
    cos_phi = Mat_ < T > (rows, cols);
    sin_phi = Mat_ < T > (rows, cols);
    Mat r = computeRadius<T>(points3d);
    for (int y = 0; y < rows; ++y)
    {
      T *row_cos_theta = cos_theta.ptr < T > (y), *row_sin_theta = sin_theta.ptr < T > (y);
      T *row_cos_phi = cos_phi.ptr < T > (y), *row_sin_phi = sin_phi.ptr < T > (y);
      const Vec3T * row_points = points3d.ptr < Vec3T > (y), *row_points_end = points3d.ptr < Vec3T
          > (y) + points3d.cols;
      const T * row_r = r.ptr < T > (y);
      for (; row_points < row_points_end;
          ++row_cos_theta, ++row_sin_theta, ++row_cos_phi, ++row_sin_phi, ++row_points, ++row_r)
      {
        // In the paper, z goes away from the camera, y goes down, x goes right
        // OpenCV has the same conventions
        // Theta goes from z to x (and actually goes from -pi/2 to pi/2, phi goes from z to y
        float theta = (float)std::atan2(row_points->val[0], row_points->val[2]);
        *row_cos_theta = std::cos(theta);
        *row_sin_theta = std::sin(theta);
        float phi = (float)std::asin(row_points->val[1] / (*row_r));
        *row_cos_phi = std::cos(phi);
        *row_sin_phi = std::sin(phi);
      }
    }
  }

  /** Modify normals to make sure they point towards the camera
   * @param normals
   */
  template<typename T>
  inline
  void
  signNormal(const Vec<T, 3> & normal_in, Vec<T, 3> & normal_out)
  {
    Vec<T, 3> res;
    if (normal_in[2] > 0)
      res = -normal_in / norm_vec(normal_in);
    else
      res = normal_in / norm_vec(normal_in);

    normal_out[0] = res[0];
    normal_out[1] = res[1];
    normal_out[2] = res[2];
  }
  /** Modify normals to make sure they point towards the camera
   * @param normals
   */
  template<typename T>
  inline
  void
  signNormal(T a, T b, T c, Vec<T, 3> & normal)
  {
    T norm = 1 / std::sqrt(a * a + b * b + c * c);
    if (c > 0)
    {
      normal[0] = -a * norm;
      normal[1] = -b * norm;
      normal[2] = -c * norm;
    }
    else
    {
      normal[0] = a * norm;
      normal[1] = b * norm;
      normal[2] = c * norm;
    }
  }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  class RgbdNormalsImpl
  {
  public:
    RgbdNormalsImpl(int rows, int cols, int window_size, int depth, const Mat &K,
                    RgbdNormals::RGBD_NORMALS_METHOD method)
        :
          rows_(rows),
          cols_(cols),
          depth_(depth),
          window_size_(window_size),
          method_(method)
    {
      K.convertTo(K_, depth);
      K.copyTo(K_ori_);
    }

    virtual
    ~RgbdNormalsImpl()
    {
    }

    virtual void
    cache()=0;

    bool
    validate(int rows, int cols, int depth, const Mat &K_ori, int window_size, int method) const
    {
      if ((K_ori.cols != K_ori_.cols) || (K_ori.rows != K_ori_.rows) || (K_ori.type() != K_ori_.type()))
        return false;
      bool K_test = !(countNonZero(K_ori != K_ori_));
      return (rows == rows_) && (cols == cols_) && (window_size == window_size_) && (depth == depth_) && (K_test)
             && (method == method_);
    }
  protected:
    int rows_, cols_, depth_;
    Mat K_, K_ori_;
    int window_size_;
    RgbdNormals::RGBD_NORMALS_METHOD method_;
  };

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /** Given a set of 3d points in a depth image, compute the normals at each point
   * using the FALS method described in
   * ``Fast and Accurate Computation of Surface Normals from Range Images``
   * by H. Badino, D. Huber, Y. Park and T. Kanade
   */
  template<typename T>
  class FALS: public RgbdNormalsImpl
  {
  public:
    typedef Matx<T, 3, 3> Mat33T;
    typedef Vec<T, 9> Vec9T;
    typedef Vec<T, 3> Vec3T;

    FALS(int rows, int cols, int window_size, int depth, const Mat &K, RgbdNormals::RGBD_NORMALS_METHOD method)
        :
          RgbdNormalsImpl(rows, cols, window_size, depth, K, method)
    {
    }
    ~FALS()
    {
    }

    /** Compute cached data
     */
    virtual void
    cache() CV_OVERRIDE
    {
      // Compute theta and phi according to equation 3
      Mat cos_theta, sin_theta, cos_phi, sin_phi;
      computeThetaPhi<T>(rows_, cols_, K_, cos_theta, sin_theta, cos_phi, sin_phi);

      // Compute all the v_i for every points
      std::vector<Mat> channels(3);
      channels[0] = sin_theta.mul(cos_phi);
      channels[1] = sin_phi;
      channels[2] = cos_theta.mul(cos_phi);
      merge(channels, V_);

      // Compute M
      Mat_<Vec9T> M(rows_, cols_);
      Mat33T VVt;
      const Vec3T * vec = V_[0];
      Vec9T * M_ptr = M[0], *M_ptr_end = M_ptr + rows_ * cols_;
      for (; M_ptr != M_ptr_end; ++vec, ++M_ptr)
      {
        VVt = (*vec) * vec->t();
        *M_ptr = Vec9T(VVt.val);
      }

      boxFilter(M, M, M.depth(), Size(window_size_, window_size_), Point(-1, -1), false);

      // Compute M's inverse
      Mat33T M_inv;
      M_inv_.create(rows_, cols_);
      Vec9T * M_inv_ptr = M_inv_[0];
      for (M_ptr = &M(0); M_ptr != M_ptr_end; ++M_inv_ptr, ++M_ptr)
      {
        // We have a semi-definite matrix
        invert(Mat33T(M_ptr->val), M_inv, DECOMP_CHOLESKY);
        *M_inv_ptr = Vec9T(M_inv.val);
      }
    }

    /** Compute the normals
     * @param r
     * @return
     */
    virtual void
    compute(const Mat&, const Mat &r, Mat & normals) const
    {
      // Compute B
      Mat_<Vec3T> B(rows_, cols_);

      const T* row_r = r.ptr < T > (0), *row_r_end = row_r + rows_ * cols_;
      const Vec3T *row_V = V_[0];
      Vec3T *row_B = B[0];
      for (; row_r != row_r_end; ++row_r, ++row_B, ++row_V)
      {
          Vec3T val = (*row_V) / (*row_r);
          if(cvIsInf(val[0]) || cvIsNaN(val[0]) ||
             cvIsInf(val[1]) || cvIsNaN(val[1]) ||
             cvIsInf(val[2]) || cvIsNaN(val[2]))
              *row_B = Vec3T();
          else
              *row_B = val;
      }

      // Apply a box filter to B
      boxFilter(B, B, B.depth(), Size(window_size_, window_size_), Point(-1, -1), false);

      // compute the Minv*B products
      row_r = r.ptr < T > (0);
      const Vec3T * B_vec = B[0];
      const Mat33T * M_inv = reinterpret_cast<const Mat33T *>(M_inv_.ptr(0));
      Vec3T *normal = normals.ptr<Vec3T>(0);
      for (; row_r != row_r_end; ++row_r, ++B_vec, ++normal, ++M_inv)
        if (cvIsNaN(*row_r))
        {
          (*normal)[0] = *row_r;
          (*normal)[1] = *row_r;
          (*normal)[2] = *row_r;
        }
        else
        {
            Mat33T Mr = *M_inv;
            Vec3T Br = *B_vec;
            Vec3T MBr(Mr(0, 0) * Br[0] + Mr(0, 1)*Br[1] + Mr(0, 2)*Br[2],
                      Mr(1, 0) * Br[0] + Mr(1, 1)*Br[1] + Mr(1, 2)*Br[2],
                      Mr(2, 0) * Br[0] + Mr(2, 1)*Br[1] + Mr(2, 2)*Br[2]);
           signNormal(MBr, *normal);
        }
    }

  private:
    Mat_<Vec3T> V_;
    Mat_<Vec9T> M_inv_;
  };

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** Function that multiplies K_inv by a vector. It is just meant to speed up the product as we know
 * that K_inv is upper triangular and K_inv(2,2)=1
 * @param K_inv
 * @param a
 * @param b
 * @param c
 * @param res
 */
template<typename T, typename U>
void
multiply_by_K_inv(const Matx<T, 3, 3> & K_inv, U a, U b, U c, Vec<T, 3> &res)
{
  res[0] = (T)(K_inv(0, 0) * a + K_inv(0, 1) * b + K_inv(0, 2) * c);
  res[1] = (T)(K_inv(1, 1) * b + K_inv(1, 2) * c);
  res[2] = (T)c;
}

  /** Given a depth image, compute the normals as detailed in the LINEMOD paper
   * ``Gradient Response Maps for Real-Time Detection of Texture-Less Objects``
   * by S. Hinterstoisser, C. Cagniart, S. Ilic, P. Sturm, N. Navab, P. Fua, and V. Lepetit
   */
  template<typename T>
  class LINEMOD: public RgbdNormalsImpl
  {
  public:
    typedef Vec<T, 3> Vec3T;
    typedef Matx<T, 3, 3> Mat33T;

    LINEMOD(int rows, int cols, int window_size, int depth, const Mat &K,
            RgbdNormals::RGBD_NORMALS_METHOD method)
        :
          RgbdNormalsImpl(rows, cols, window_size, depth, K, method)
    {
    }

    /** Compute cached data
     */
    virtual void
    cache() CV_OVERRIDE
    {
    }

    /** Compute the normals
     * @param r
     * @param normals the output normals
     */
    void
    compute(const Mat& depth_in, Mat & normals) const
    {
      switch (depth_in.depth())
      {
        case CV_16U:
        {
          const Mat_<unsigned short> &depth(depth_in);
          computeImpl<unsigned short, long>(depth, normals);
          break;
        }
        case CV_32F:
        {
          const Mat_<float> &depth(depth_in);
          computeImpl<float, float>(depth, normals);
          break;
        }
        case CV_64F:
        {
          const Mat_<double> &depth(depth_in);
          computeImpl<double, double>(depth, normals);
          break;
        }
      }
    }

  private:
    /** Compute the normals
     * @param r
     * @return
     */
    template<typename DepthDepth, typename ContainerDepth>
    Mat
    computeImpl(const Mat_<DepthDepth> &depth, Mat & normals) const
    {
      const int r = 5; // used to be 7
      const int sample_step = r;
      const int square_size = ((2 * r / sample_step) + 1);
      long offsets[square_size * square_size];
      long offsets_x[square_size * square_size];
      long offsets_y[square_size * square_size];
      long offsets_x_x[square_size * square_size];
      long offsets_x_y[square_size * square_size];
      long offsets_y_y[square_size * square_size];
      for (int j = -r, index = 0; j <= r; j += sample_step)
        for (int i = -r; i <= r; i += sample_step, ++index)
        {
          offsets_x[index] = i;
          offsets_y[index] = j;
          offsets_x_x[index] = i*i;
          offsets_x_y[index] = i*j;
          offsets_y_y[index] = j*j;
          offsets[index] = j * cols_ + i;
        }

      // Define K_inv by hand, just for higher accuracy
      Mat33T K_inv = Matx<T, 3, 3>::eye(), K;
      K_.copyTo(K);
      K_inv(0, 0) = 1.0f / K(0, 0);
      K_inv(0, 1) = -K(0, 1) / (K(0, 0) * K(1, 1));
      K_inv(0, 2) = (K(0, 1) * K(1, 2) - K(0, 2) * K(1, 1)) / (K(0, 0) * K(1, 1));
      K_inv(1, 1) = 1 / K(1, 1);
      K_inv(1, 2) = -K(1, 2) / K(1, 1);

      Vec3T X1_minus_X, X2_minus_X;

      ContainerDepth difference_threshold = 50;
      normals.setTo(std::numeric_limits<DepthDepth>::quiet_NaN());
      for (int y = r; y < rows_ - r - 1; ++y)
      {
        const DepthDepth * p_line = reinterpret_cast<const DepthDepth*>(depth.ptr(y, r));
        Vec3T *normal = normals.ptr<Vec3T>(y, r);

        for (int x = r; x < cols_ - r - 1; ++x)
        {
          DepthDepth d = p_line[0];

          // accum
          long A[4];
          A[0] = A[1] = A[2] = A[3] = 0;
          ContainerDepth b[2];
          b[0] = b[1] = 0;
          for (unsigned int i = 0; i < square_size * square_size; ++i) {
            // We need to cast to ContainerDepth in case we have unsigned DepthDepth
            ContainerDepth delta = ContainerDepth(p_line[offsets[i]]) - ContainerDepth(d);
            if (std::abs(delta) > difference_threshold)
               continue;

             A[0] += offsets_x_x[i];
             A[1] += offsets_x_y[i];
             A[3] += offsets_y_y[i];
             b[0] += offsets_x[i] * delta;
             b[1] += offsets_y[i] * delta;
          }

          // solve for the optimal gradient D of equation (8)
          long det = A[0] * A[3] - A[1] * A[1];
          // We should divide the following two by det, but instead, we multiply
          // X1_minus_X and X2_minus_X by det (which does not matter as we normalize the normals)
          // Therefore, no division is done: this is only for speedup
          ContainerDepth dx = (A[3] * b[0] - A[1] * b[1]);
          ContainerDepth dy = (-A[1] * b[0] + A[0] * b[1]);

          // Compute the dot product
          //Vec3T X = K_inv * Vec3T(x, y, 1) * depth(y, x);
          //Vec3T X1 = K_inv * Vec3T(x + 1, y, 1) * (depth(y, x) + dx);
          //Vec3T X2 = K_inv * Vec3T(x, y + 1, 1) * (depth(y, x) + dy);
          //Vec3T nor = (X1 - X).cross(X2 - X);
          multiply_by_K_inv(K_inv, d * det + (x + 1) * dx, y * dx, dx, X1_minus_X);
          multiply_by_K_inv(K_inv, x * dy, d * det + (y + 1) * dy, dy, X2_minus_X);
          Vec3T nor = X1_minus_X.cross(X2_minus_X);
          signNormal(nor, *normal);

          ++p_line;
          ++normal;
        }
      }

      return normals;
    }
  };

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /** Given a set of 3d points in a depth image, compute the normals at each point
   * using the SRI method described in
   * ``Fast and Accurate Computation of Surface Normals from Range Images``
   * by H. Badino, D. Huber, Y. Park and T. Kanade
   */
  template<typename T>
  class SRI: public RgbdNormalsImpl
  {
  public:
    typedef Matx<T, 3, 3> Mat33T;
    typedef Vec<T, 9> Vec9T;
    typedef Vec<T, 3> Vec3T;

    SRI(int rows, int cols, int window_size, int depth, const Mat &K, RgbdNormals::RGBD_NORMALS_METHOD method)
        :
          RgbdNormalsImpl(rows, cols, window_size, depth, K, method),
          phi_step_(0),
          theta_step_(0)
    {
    }

    /** Compute cached data
     */
    virtual void
    cache() CV_OVERRIDE
    {
      Mat_<T> cos_theta, sin_theta, cos_phi, sin_phi;
      computeThetaPhi<T>(rows_, cols_, K_, cos_theta, sin_theta, cos_phi, sin_phi);

      // Create the derivative kernels
      getDerivKernels(kx_dx_, ky_dx_, 1, 0, window_size_, true, depth_);
      getDerivKernels(kx_dy_, ky_dy_, 0, 1, window_size_, true, depth_);

      // Get the mapping function for SRI
      float min_theta = (float)std::asin(sin_theta(0, 0)), max_theta = (float)std::asin(sin_theta(0, cols_ - 1));
      float min_phi = (float)std::asin(sin_phi(0, cols_/2-1)), max_phi = (float)std::asin(sin_phi(rows_ - 1, cols_/2-1));

      std::vector<Point3f> points3d(cols_ * rows_);
      R_hat_.create(rows_, cols_);
      phi_step_ = float(max_phi - min_phi) / (rows_ - 1);
      theta_step_ = float(max_theta - min_theta) / (cols_ - 1);
      for (int phi_int = 0, k = 0; phi_int < rows_; ++phi_int)
      {
        float phi = min_phi + phi_int * phi_step_;
        for (int theta_int = 0; theta_int < cols_; ++theta_int, ++k)
        {
          float theta = min_theta + theta_int * theta_step_;
          // Store the 3d point to project it later
          points3d[k] = Point3f(std::sin(theta) * std::cos(phi), std::sin(phi), std::cos(theta) * std::cos(phi));

          // Cache the rotation matrix and negate it
          Mat_<T> mat =
              (Mat_ < T > (3, 3) << 0, 1, 0, 0, 0, 1, 1, 0, 0) * ((Mat_ < T > (3, 3) << std::cos(theta), -std::sin(
                  theta), 0, std::sin(theta), std::cos(theta), 0, 0, 0, 1))
              * ((Mat_ < T > (3, 3) << std::cos(phi), 0, -std::sin(phi), 0, 1, 0, std::sin(phi), 0, std::cos(phi)));
          for (unsigned char i = 0; i < 3; ++i)
            mat(i, 1) = mat(i, 1) / std::cos(phi);
          // The second part of the matrix is never explained in the paper ... but look at the wikipedia normal article
          mat(0, 0) = mat(0, 0) - 2 * std::cos(phi) * std::sin(theta);
          mat(1, 0) = mat(1, 0) - 2 * std::sin(phi);
          mat(2, 0) = mat(2, 0) - 2 * std::cos(phi) * std::cos(theta);

          R_hat_(phi_int, theta_int) = Vec9T((T*) (mat.data));
        }
      }

      map_.create(rows_, cols_);
      projectPoints(points3d, Mat(3,1,CV_32FC1,Scalar::all(0.0f)), Mat(3,1,CV_32FC1,Scalar::all(0.0f)), K_, Mat(), map_);
      map_ = map_.reshape(2, rows_);
      convertMaps(map_, Mat(), xy_, fxy_, CV_16SC2);

      //map for converting from Spherical coordinate space to Euclidean space
      euclideanMap_.create(rows_,cols_);
      float invFx = (float)(1.0f/K_.at<T>(0,0)), cx = (float)K_.at<T>(0,2);
      double invFy = 1.0f/K_.at<T>(1,1), cy = K_.at<T>(1,2);
      for (int i = 0; i < rows_; i++)
      {
          float y = (float)((i - cy)*invFy);
          for (int j = 0; j < cols_; j++)
          {
              float x = (j - cx)*invFx;
              float theta = std::atan(x);
              float phi = std::asin(y/std::sqrt(x*x+y*y+1.0f));

              euclideanMap_(i,j) = Vec2f((theta-min_theta)/theta_step_,(phi-min_phi)/phi_step_);
          }
      }
      //convert map to 2 maps in short format for increasing speed in remap function
      convertMaps(euclideanMap_, Mat(), invxy_, invfxy_, CV_16SC2);

      // Update the kernels: the steps are due to the fact that derivatives will be computed on a grid where
      // the step is not 1. Only need to do it on one dimension as it computes derivatives in only one direction
      kx_dx_ /= theta_step_;
      ky_dy_ /= phi_step_;
    }

    /** Compute the normals
     * @param r
     * @return
     */
    virtual void
    compute(const Mat& points3d, const Mat &r, Mat & normals) const
    {
      const Mat_<T>& r_T(r);
      const Mat_<Vec3T> &points3d_T(points3d);
      compute(points3d_T, r_T, normals);
    }

    /** Compute the normals
     * @param r
     * @return
     */
    void
    compute(const Mat_<Vec3T> &, const Mat_<T> &r_non_interp, Mat & normals_out) const
    {
      // Interpolate the radial image to make derivatives meaningful
      Mat_<T> r;
      // higher quality remapping does not help here
      remap(r_non_interp, r, xy_, fxy_, INTER_LINEAR);

      // Compute the derivatives with respect to theta and phi
      // TODO add bilateral filtering (as done in kinfu)
      Mat_<T> r_theta, r_phi;
      sepFilter2D(r, r_theta, r.depth(), kx_dx_, ky_dx_);
      //current OpenCV version sometimes corrupts r matrix after second call of sepFilter2D
      //it depends on resolution, be careful
      sepFilter2D(r, r_phi, r.depth(), kx_dy_, ky_dy_);

      // Fill the result matrix
      Mat_<Vec3T> normals(rows_, cols_);

      const T* r_theta_ptr = r_theta[0], *r_theta_ptr_end = r_theta_ptr + rows_ * cols_;
      const T* r_phi_ptr = r_phi[0];
      const Mat33T * R = reinterpret_cast<const Mat33T *>(R_hat_[0]);
      const T* r_ptr = r[0];
      Vec3T * normal = normals[0];
      for (; r_theta_ptr != r_theta_ptr_end; ++r_theta_ptr, ++r_phi_ptr, ++R, ++r_ptr, ++normal)
      {
        if (cvIsNaN(*r_ptr))
        {
          (*normal)[0] = *r_ptr;
          (*normal)[1] = *r_ptr;
          (*normal)[2] = *r_ptr;
        }
        else
        {
          T r_theta_over_r = (*r_theta_ptr) / (*r_ptr);
          T r_phi_over_r = (*r_phi_ptr) / (*r_ptr);
          // R(1,1) is 0
          signNormal((*R)(0, 0) + (*R)(0, 1) * r_theta_over_r + (*R)(0, 2) * r_phi_over_r,
                     (*R)(1, 0) + (*R)(1, 2) * r_phi_over_r,
                     (*R)(2, 0) + (*R)(2, 1) * r_theta_over_r + (*R)(2, 2) * r_phi_over_r, *normal);
        }
      }

      remap(normals, normals_out, invxy_, invfxy_, INTER_LINEAR);
      normal = normals_out.ptr<Vec3T>(0);
      Vec3T * normal_end = normal + rows_ * cols_;
      for (; normal != normal_end; ++normal)
        signNormal((*normal)[0], (*normal)[1], (*normal)[2], *normal);
    }
  private:
    /** Stores R */
    Mat_<Vec9T> R_hat_;
    float phi_step_, theta_step_;

    /** Derivative kernels */
    Mat kx_dx_, ky_dx_, kx_dy_, ky_dy_;
    /** mapping function to get an SRI image */
    Mat_<Vec2f> map_;
    Mat xy_, fxy_;

    Mat_<Vec2f> euclideanMap_;
    Mat invxy_, invfxy_;
  };

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /** Default constructor of the Algorithm class that computes normals
   */
  RgbdNormals::RgbdNormals(int rows, int cols, int depth, InputArray K_in, int window_size, int method_in)
      :
        rows_(rows),
        cols_(cols),
        depth_(depth),
        K_(K_in.getMat()),
        window_size_(window_size),
        method_(method_in),
        rgbd_normals_impl_(0)
  {
    CV_Assert(depth == CV_32F || depth == CV_64F);
    CV_Assert(K_.cols == 3 && K_.rows == 3);
  }

  // Just to remove a warning
  void delete_normals_impl(void *rgbd_normals_impl_, int method_, int depth);
  void delete_normals_impl(void *rgbd_normals_impl_, int method_, int depth)
  {
    if (rgbd_normals_impl_ == 0)
      return;
    switch (method_)
    {
      case RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD:
      {
        if (depth == CV_32F)
          delete reinterpret_cast<const LINEMOD<float> *>(rgbd_normals_impl_);
        else
          delete reinterpret_cast<const LINEMOD<double> *>(rgbd_normals_impl_);
        break;
      }
      case RgbdNormals::RGBD_NORMALS_METHOD_SRI:
      {
        if (depth == CV_32F)
          delete reinterpret_cast<const SRI<float> *>(rgbd_normals_impl_);
        else
          delete reinterpret_cast<const SRI<double> *>(rgbd_normals_impl_);
        break;
      }
      case (RgbdNormals::RGBD_NORMALS_METHOD_FALS):
      {
        if (depth == CV_32F)
          delete reinterpret_cast<const FALS<float> *>(rgbd_normals_impl_);
        else
          delete reinterpret_cast<const FALS<double> *>(rgbd_normals_impl_);
        break;
      }
    }
  }

  /** Destructor
   */
  RgbdNormals::~RgbdNormals()
  {
    delete_normals_impl(rgbd_normals_impl_, method_, depth_);
  }

  void
  RgbdNormals::initialize_normals_impl(int rows, int cols, int depth, const Mat & K, int window_size,
                                       int method_in) const
  {
    CV_Assert(rows > 0 && cols > 0 && (depth == CV_32F || depth == CV_64F));
    CV_Assert(window_size == 1 || window_size == 3 || window_size == 5 || window_size == 7);
    CV_Assert(K_.cols == 3 && K.rows == 3 && (K.depth() == CV_32F || K.depth() == CV_64F));
    CV_Assert(
        method_in == RGBD_NORMALS_METHOD_FALS || method_in == RGBD_NORMALS_METHOD_LINEMOD
        || method_in == RGBD_NORMALS_METHOD_SRI);
    switch (method_in)
    {
      case (RGBD_NORMALS_METHOD_FALS):
      {
        if (depth == CV_32F)
          rgbd_normals_impl_ = new FALS<float>(rows, cols, window_size, depth, K, RGBD_NORMALS_METHOD_FALS);
        else
          rgbd_normals_impl_ = new FALS<double>(rows, cols, window_size, depth, K, RGBD_NORMALS_METHOD_FALS);
        break;
      }
      case (RGBD_NORMALS_METHOD_LINEMOD):
      {
        if (depth == CV_32F)
          rgbd_normals_impl_ = new LINEMOD<float>(rows, cols, window_size, depth, K, RGBD_NORMALS_METHOD_LINEMOD);
        else
          rgbd_normals_impl_ = new LINEMOD<double>(rows, cols, window_size, depth, K, RGBD_NORMALS_METHOD_LINEMOD);
        break;
      }
      case RGBD_NORMALS_METHOD_SRI:
      {
        if (depth == CV_32F)
          rgbd_normals_impl_ = new SRI<float>(rows, cols, window_size, depth, K, RGBD_NORMALS_METHOD_SRI);
        else
          rgbd_normals_impl_ = new SRI<double>(rows, cols, window_size, depth, K, RGBD_NORMALS_METHOD_SRI);
        break;
      }
    }

    reinterpret_cast<RgbdNormalsImpl *>(rgbd_normals_impl_)->cache();
  }

  /** Initializes some data that is cached for later computation
   * If that function is not called, it will be called the first time normals are computed
   */
  void
  RgbdNormals::initialize() const
  {
    if (rgbd_normals_impl_ == 0)
      initialize_normals_impl(rows_, cols_, depth_, K_, window_size_, method_);
    else if (!reinterpret_cast<RgbdNormalsImpl *>(rgbd_normals_impl_)->validate(rows_, cols_, depth_, K_, window_size_,
                                                                                method_)) {
      delete_normals_impl(rgbd_normals_impl_, method_, depth_);
      initialize_normals_impl(rows_, cols_, depth_, K_, window_size_, method_);
    }
  }

  /** Given a set of 3d points in a depth image, compute the normals at each point
   * @param points3d_in depth a float depth image. Or it can be rows x cols x 3 is they are 3d points
   * @param normals a rows x cols x 3 matrix
   */
  void
  RgbdNormals::operator()(InputArray points3d_in, OutputArray normals_out) const
  {
    Mat points3d_ori = points3d_in.getMat();

    CV_Assert(points3d_ori.dims == 2);
    // Either we have 3d points or a depth image
    switch (method_)
    {
      case (RGBD_NORMALS_METHOD_FALS):
      {
        CV_Assert(points3d_ori.channels() == 3);
        CV_Assert(points3d_ori.depth() == CV_32F || points3d_ori.depth() == CV_64F);
        break;
      }
      case RGBD_NORMALS_METHOD_LINEMOD:
      {
        CV_Assert(
            ((points3d_ori.channels() == 3) && (points3d_ori.depth() == CV_32F || points3d_ori.depth() == CV_64F)) || ((points3d_ori.channels() == 1) && (points3d_ori.depth() == CV_16U || points3d_ori.depth() == CV_32F || points3d_ori.depth() == CV_64F)));
        break;
      }
      case RGBD_NORMALS_METHOD_SRI:
      {
        CV_Assert( ((points3d_ori.channels() == 3) && (points3d_ori.depth() == CV_32F || points3d_ori.depth() == CV_64F)));
        break;
      }
    }

    // Initialize the pimpl
    initialize();

    // Precompute something for RGBD_NORMALS_METHOD_SRI and RGBD_NORMALS_METHOD_FALS
    Mat points3d, radius;
    if ((method_ == RGBD_NORMALS_METHOD_SRI) || (method_ == RGBD_NORMALS_METHOD_FALS))
    {
      // Make the points have the right depth
      if (points3d_ori.depth() == depth_)
        points3d = points3d_ori;
      else
        points3d_ori.convertTo(points3d, depth_);

      // Compute the distance to the points
      if (depth_ == CV_32F)
        radius = computeRadius<float>(points3d);
      else
        radius = computeRadius<double>(points3d);
    }

    // Get the normals
    normals_out.create(points3d_ori.size(), CV_MAKETYPE(depth_, 3));
    if (points3d_in.empty())
      return;

    Mat normals = normals_out.getMat();
    switch (method_)
    {
      case (RGBD_NORMALS_METHOD_FALS):
      {
        if (depth_ == CV_32F)
          reinterpret_cast<const FALS<float> *>(rgbd_normals_impl_)->compute(points3d, radius, normals);
        else
          reinterpret_cast<const FALS<double> *>(rgbd_normals_impl_)->compute(points3d, radius, normals);
        break;
      }
      case RGBD_NORMALS_METHOD_LINEMOD:
      {
        // Only focus on the depth image for LINEMOD
        Mat depth;
        if (points3d_ori.channels() == 3)
        {
          std::vector<Mat> channels;
          split(points3d, channels);
          depth = channels[2];
        }
        else
          depth = points3d_ori;

        if (depth_ == CV_32F)
          reinterpret_cast<const LINEMOD<float> *>(rgbd_normals_impl_)->compute(depth, normals);
        else
          reinterpret_cast<const LINEMOD<double> *>(rgbd_normals_impl_)->compute(depth, normals);
        break;
      }
      case RGBD_NORMALS_METHOD_SRI:
      {
        if (depth_ == CV_32F)
          reinterpret_cast<const SRI<float> *>(rgbd_normals_impl_)->compute(points3d, radius, normals);
        else
          reinterpret_cast<const SRI<double> *>(rgbd_normals_impl_)->compute(points3d, radius, normals);
        break;
      }
    }
  }
}
}

