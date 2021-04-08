// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_WillowGarage.md file found in this module's directory

#include "precomp.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cv
{
namespace rgbd
{
  class DepthCleanerImpl
  {
  public:
    DepthCleanerImpl(int window_size, int depth, DepthCleaner::DEPTH_CLEANER_METHOD method)
        :
          depth_(depth),
          window_size_(window_size),
          method_(method)
    {
    }

    virtual
    ~DepthCleanerImpl()
    {
    }

    virtual void
    cache()=0;

    bool
    validate(int depth, int window_size, int method) const
    {
      return (window_size == window_size_) && (depth == depth_) && (method == method_);
    }
  protected:
    int depth_;
    int window_size_;
    DepthCleaner::DEPTH_CLEANER_METHOD method_;
  };

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /** Given a depth image, compute the normals as detailed in the LINEMOD paper
   * ``Gradient Response Maps for Real-Time Detection of Texture-Less Objects``
   * by S. Hinterstoisser, C. Cagniart, S. Ilic, P. Sturm, N. Navab, P. Fua, and V. Lepetit
   */
  template<typename T>
  class NIL: public DepthCleanerImpl
  {
  public:
    typedef Vec<T, 3> Vec3T;
    typedef Matx<T, 3, 3> Mat33T;

    NIL(int window_size, int depth, DepthCleaner::DEPTH_CLEANER_METHOD method)
        :
          DepthCleanerImpl(window_size, depth, method)
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
     * @return
     */
    void
    compute(const Mat& depth_in, Mat& depth_out) const
    {
      switch (depth_in.depth())
      {
        case CV_16U:
        {
          const Mat_<unsigned short> &depth(depth_in);
          Mat depth_out_tmp;
          computeImpl<unsigned short, float>(depth, depth_out_tmp, 0.001f);
          depth_out_tmp.convertTo(depth_out, CV_16U);
          break;
        }
        case CV_32F:
        {
          const Mat_<float> &depth(depth_in);
          computeImpl<float, float>(depth, depth_out, 1);
          break;
        }
        case CV_64F:
        {
          const Mat_<double> &depth(depth_in);
          computeImpl<double, double>(depth, depth_out, 1);
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
    void
    computeImpl(const Mat_<DepthDepth> &depth_in, Mat & depth_out, ContainerDepth scale) const
    {
      const ContainerDepth theta_mean = (float)(30. * CV_PI / 180);
      int rows = depth_in.rows;
      int cols = depth_in.cols;

      // Precompute some data
      const ContainerDepth sigma_L = (float)(0.8 + 0.035 * theta_mean / (CV_PI / 2 - theta_mean));
      Mat_<ContainerDepth> sigma_z(rows, cols);
      for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
          sigma_z(y, x) = (float)(0.0012 + 0.0019 * (depth_in(y, x) * scale - 0.4) * (depth_in(y, x) * scale - 0.4));

      ContainerDepth difference_threshold = 10;
      Mat_<ContainerDepth> Dw_sum = Mat_<ContainerDepth>::zeros(rows, cols), w_sum =
          Mat_<ContainerDepth>::zeros(rows, cols);
      for (int y = 0; y < rows - 1; ++y)
      {
        // Every pixel has had the contribution of previous pixels (in a row-major way)
        for (int x = 1; x < cols - 1; ++x)
        {
          for (int j = 0; j <= 1; ++j)
            for (int i = -1; i <= 1; ++i)
            {
              if ((j == 0) && (i == -1))
                continue;
              ContainerDepth delta_u = sqrt(
                  ContainerDepth(j) * ContainerDepth(j) + ContainerDepth(i) * ContainerDepth(i));
              ContainerDepth delta_z;
              if (depth_in(y, x) > depth_in(y + j, x + i))
                delta_z = (float)(depth_in(y, x) - depth_in(y + j, x + i));
              else
                delta_z = (float)(depth_in(y + j, x + i) - depth_in(y, x));
              if (delta_z < difference_threshold)
              {
                delta_z *= scale;
                ContainerDepth w = exp(
                    -delta_u * delta_u / 2 / sigma_L / sigma_L - delta_z * delta_z / 2 / sigma_z(y, x) / sigma_z(y, x));
                w_sum(y, x) += w;
                Dw_sum(y, x) += depth_in(y + j, x + i) * w;
                if ((j != 0) || (i != 0))
                {
                  w = exp(
                      -delta_u * delta_u / 2 / sigma_L / sigma_L - delta_z * delta_z / 2 / sigma_z(y + j, x + i)
                                                                   / sigma_z(y + j, x + i));
                  w_sum(y + j, x + i) += w;
                  Dw_sum(y + j, x + i) += depth_in(y, x) * w;
                }
              }
            }
        }
      }
      Mat(Dw_sum / w_sum).copyTo(depth_out);
    }
  };

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /** Default constructor of the Algorithm class that computes normals
   */
  DepthCleaner::DepthCleaner(int depth, int window_size, int method_in)
      :
        depth_(depth),
        window_size_(window_size),
        method_(method_in),
        depth_cleaner_impl_(0)
  {
    CV_Assert(depth == CV_16U || depth == CV_32F || depth == CV_64F);
  }

  /** Destructor
   */
  DepthCleaner::~DepthCleaner()
  {
    if (depth_cleaner_impl_ == 0)
      return;
    switch (method_)
    {
      case DEPTH_CLEANER_NIL:
      {
        switch (depth_)
        {
          case CV_16U:
            delete reinterpret_cast<const NIL<unsigned short> *>(depth_cleaner_impl_);
            break;
          case CV_32F:
            delete reinterpret_cast<const NIL<float> *>(depth_cleaner_impl_);
            break;
          case CV_64F:
            delete reinterpret_cast<const NIL<double> *>(depth_cleaner_impl_);
            break;
        }
        break;
      }
    }
  }

  void
  DepthCleaner::initialize_cleaner_impl() const
  {
    CV_Assert(depth_ == CV_16U || depth_ == CV_32F || depth_ == CV_64F);
    CV_Assert(window_size_ == 1 || window_size_ == 3 || window_size_ == 5 || window_size_ == 7);
    CV_Assert( method_ == DEPTH_CLEANER_NIL);
    switch (method_)
    {
      case (DEPTH_CLEANER_NIL):
      {
        switch (depth_)
        {
          case CV_16U:
            depth_cleaner_impl_ = new NIL<unsigned short>(window_size_, depth_, DEPTH_CLEANER_NIL);
            break;
          case CV_32F:
            depth_cleaner_impl_ = new NIL<float>(window_size_, depth_, DEPTH_CLEANER_NIL);
            break;
          case CV_64F:
            depth_cleaner_impl_ = new NIL<double>(window_size_, depth_, DEPTH_CLEANER_NIL);
            break;
        }
        break;
      }
    }

    reinterpret_cast<DepthCleanerImpl *>(depth_cleaner_impl_)->cache();
  }

  /** Initializes some data that is cached for later computation
   * If that function is not called, it will be called the first time normals are computed
   */
  void
  DepthCleaner::initialize() const
  {
    if (depth_cleaner_impl_ == 0)
      initialize_cleaner_impl();
    else if (!reinterpret_cast<DepthCleanerImpl *>(depth_cleaner_impl_)->validate(depth_, window_size_, method_))
      initialize_cleaner_impl();
  }

  /** Given a set of 3d points in a depth image, compute the normals at each point
   * using the SRI method described in
   * ``Fast and Accurate Computation of Surface Normals from Range Images``
   * by H. Badino, D. Huber, Y. Park and T. Kanade
   * @param depth depth a float depth image. Or it can be rows x cols x 3 is they are 3d points
   * @param window_size the window size on which to compute the derivatives
   * @return normals a rows x cols x 3 matrix
   */
  void
  DepthCleaner::operator()(InputArray depth_in_array, OutputArray depth_out_array) const
  {
    Mat depth_in = depth_in_array.getMat();
    CV_Assert(depth_in.dims == 2);
    CV_Assert(depth_in.channels() == 1);

    depth_out_array.create(depth_in.size(), depth_);
    Mat depth_out = depth_out_array.getMat();

    // Initialize the pimpl
    initialize();

    // Clean the depth
    switch (method_)
    {
      case (DEPTH_CLEANER_NIL):
      {
        switch (depth_)
        {
          case CV_16U:
            reinterpret_cast<const NIL<unsigned short> *>(depth_cleaner_impl_)->compute(depth_in, depth_out);
            break;
          case CV_32F:
            reinterpret_cast<const NIL<float> *>(depth_cleaner_impl_)->compute(depth_in, depth_out);
            break;
          case CV_64F:
            reinterpret_cast<const NIL<double> *>(depth_cleaner_impl_)->compute(depth_in, depth_out);
            break;
        }
        break;
      }
    }
  }
}
}
