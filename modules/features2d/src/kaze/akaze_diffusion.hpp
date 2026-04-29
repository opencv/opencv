// Internal header: shared nonlinear diffusion utilities used by both
// AKAZEFeatures.cpp and KAZEFeatures.cpp.
// Not part of the public API.

#ifndef __OPENCV_FEATURES_2D_AKAZE_DIFFUSION_HPP__
#define __OPENCV_FEATURES_2D_AKAZE_DIFFUSION_HPP__

#include "../precomp.hpp"
#include "nldiffusion_functions.h"
#ifdef HAVE_OPENCL
#include "opencl_kernels_features2d.hpp"
#endif

namespace cv
{

static inline void
nld_step_scalar_one_lane(const Mat& Lt, const Mat& Lf, Mat& Lstep, float step_size, int row_begin, int row_end)
{
  CV_INSTRUMENT_REGION();

  Lstep.create(Lt.size(), Lt.type());
  const int cols = Lt.cols - 2;
  int row = row_begin;

  const float *lt_a, *lt_c, *lt_b;
  const float *lf_a, *lf_c, *lf_b;
  float *dst;
  float step_r = 0.f;

  if (row == 0) {
    lt_c = Lt.ptr<float>(0) + 1;
    lf_c = Lf.ptr<float>(0) + 1;
    lt_b = Lt.ptr<float>(1) + 1;
    lf_b = Lf.ptr<float>(1) + 1;

    dst = Lstep.ptr<float>(0);
    dst[0] = 0.0f;
    ++dst;

    for (int j = 0; j < cols; j++) {
      step_r = (lf_c[j] + lf_c[j + 1])*(lt_c[j + 1] - lt_c[j]) +
               (lf_c[j] + lf_c[j - 1])*(lt_c[j - 1] - lt_c[j]) +
               (lf_c[j] + lf_b[j    ])*(lt_b[j    ] - lt_c[j]);
      dst[j] = step_r * step_size;
    }

    dst[cols] = 0.0f;
    ++row;
  }

  int middle_end = std::min(Lt.rows - 1, row_end);
  for (; row < middle_end; ++row)
  {
    lt_a = Lt.ptr<float>(row - 1);
    lf_a = Lf.ptr<float>(row - 1);
    lt_c = Lt.ptr<float>(row    );
    lf_c = Lf.ptr<float>(row    );
    lt_b = Lt.ptr<float>(row + 1);
    lf_b = Lf.ptr<float>(row + 1);
    dst = Lstep.ptr<float>(row);

    step_r = (lf_c[0] + lf_c[1])*(lt_c[1] - lt_c[0]) +
             (lf_c[0] + lf_b[0])*(lt_b[0] - lt_c[0]) +
             (lf_c[0] + lf_a[0])*(lt_a[0] - lt_c[0]);
    dst[0] = step_r * step_size;

    lt_a++; lt_c++; lt_b++;
    lf_a++; lf_c++; lf_b++;
    dst++;

    for (int j = 0; j < cols; j++)
    {
      step_r = (lf_c[j] + lf_c[j + 1])*(lt_c[j + 1] - lt_c[j]) +
               (lf_c[j] + lf_c[j - 1])*(lt_c[j - 1] - lt_c[j]) +
               (lf_c[j] + lf_b[j    ])*(lt_b[j    ] - lt_c[j]) +
               (lf_c[j] + lf_a[j    ])*(lt_a[j    ] - lt_c[j]);
      dst[j] = step_r * step_size;
    }

    step_r = (lf_c[cols] + lf_c[cols - 1])*(lt_c[cols - 1] - lt_c[cols]) +
             (lf_c[cols] + lf_b[cols    ])*(lt_b[cols    ] - lt_c[cols]) +
             (lf_c[cols] + lf_a[cols    ])*(lt_a[cols    ] - lt_c[cols]);
    dst[cols] = step_r * step_size;
  }

  if (row_end == Lt.rows) {
    lt_a = Lt.ptr<float>(row - 1) + 1;
    lf_a = Lf.ptr<float>(row - 1) + 1;
    lt_c = Lt.ptr<float>(row    ) + 1;
    lf_c = Lf.ptr<float>(row    ) + 1;

    dst = Lstep.ptr<float>(row);
    dst[0] = 0.0f;
    ++dst;

    for (int j = 0; j < cols; j++) {
      step_r = (lf_c[j] + lf_c[j + 1])*(lt_c[j + 1] - lt_c[j]) +
               (lf_c[j] + lf_c[j - 1])*(lt_c[j - 1] - lt_c[j]) +
               (lf_c[j] + lf_a[j    ])*(lt_a[j    ] - lt_c[j]);
      dst[j] = step_r * step_size;
    }

    dst[cols] = 0.0f;
  }
}

class NonLinearScalarDiffusionStep : public ParallelLoopBody
{
public:
  NonLinearScalarDiffusionStep(const Mat& Lt, const Mat& Lf, Mat& Lstep, float step_size)
    : Lt_(&Lt), Lf_(&Lf), Lstep_(&Lstep), step_size_(step_size)
  {}

  void operator()(const Range& range) const CV_OVERRIDE
  {
    nld_step_scalar_one_lane(*Lt_, *Lf_, *Lstep_, step_size_, range.start, range.end);
  }

private:
  const Mat* Lt_;
  const Mat* Lf_;
  Mat* Lstep_;
  float step_size_;
};

#ifdef HAVE_OPENCL
static inline bool
ocl_non_linear_diffusion_step(InputArray Lt_, InputArray Lf_, OutputArray Lstep_, float step_size)
{
  if(!Lt_.isContinuous())
    return false;

  UMat Lt = Lt_.getUMat();
  UMat Lf = Lf_.getUMat();
  UMat Lstep = Lstep_.getUMat();

  size_t globalSize[] = {(size_t)Lt.cols, (size_t)Lt.rows};

  ocl::Kernel ker("AKAZE_nld_step_scalar", ocl::features2d::akaze_oclsrc);
  if( ker.empty() )
    return false;

  return ker.args(
    ocl::KernelArg::ReadOnly(Lt),
    ocl::KernelArg::PtrReadOnly(Lf),
    ocl::KernelArg::PtrWriteOnly(Lstep),
    step_size).run(2, globalSize, 0, false);
}

static inline bool
ocl_pm_g2(InputArray Lx_, InputArray Ly_, OutputArray Lflow_, float kcontrast)
{
  UMat Lx = Lx_.getUMat();
  UMat Ly = Ly_.getUMat();
  UMat Lflow = Lflow_.getUMat();

  int total = Lx.rows * Lx.cols;
  size_t globalSize[] = {(size_t)total};

  ocl::Kernel ker("AKAZE_pm_g2", ocl::features2d::akaze_oclsrc);
  if( ker.empty() )
    return false;

  return ker.args(
    ocl::KernelArg::PtrReadOnly(Lx),
    ocl::KernelArg::PtrReadOnly(Ly),
    ocl::KernelArg::PtrWriteOnly(Lflow),
    kcontrast, total).run(1, globalSize, 0, false);
}
#endif // HAVE_OPENCL

static inline void
non_linear_diffusion_step(InputArray Lt_, InputArray Lf_, OutputArray Lstep_, float step_size)
{
  CV_INSTRUMENT_REGION();

  Lstep_.create(Lt_.size(), Lt_.type());

  CV_OCL_RUN(Lt_.isUMat() && Lf_.isUMat() && Lstep_.isUMat(),
    ocl_non_linear_diffusion_step(Lt_, Lf_, Lstep_, step_size));

  Mat Lt = Lt_.getMat();
  Mat Lf = Lf_.getMat();
  Mat Lstep = Lstep_.getMat();
  parallel_for_(Range(0, Lt.rows), NonLinearScalarDiffusionStep(Lt, Lf, Lstep, step_size));
}

static inline void
compute_diffusivity(InputArray Lx, InputArray Ly, OutputArray Lflow, float kcontrast, KAZE::DiffusivityType diffusivity)
{
  CV_INSTRUMENT_REGION();

  Lflow.create(Lx.size(), Lx.type());

  switch (diffusivity) {
    case KAZE::DIFF_PM_G1:
      pm_g1(Lx, Ly, Lflow, kcontrast);
    break;
    case KAZE::DIFF_PM_G2:
      CV_OCL_RUN(Lx.isUMat() && Ly.isUMat() && Lflow.isUMat(), ocl_pm_g2(Lx, Ly, Lflow, kcontrast));
      pm_g2(Lx, Ly, Lflow, kcontrast);
    break;
    case KAZE::DIFF_WEICKERT:
      weickert_diffusivity(Lx, Ly, Lflow, kcontrast);
    break;
    case KAZE::DIFF_CHARBONNIER:
      charbonnier_diffusivity(Lx, Ly, Lflow, kcontrast);
    break;
    default:
      CV_Error_(Error::StsError, ("Diffusivity is not supported: %d", static_cast<int>(diffusivity)));
    break;
  }
}

} // namespace cv

#endif // __OPENCV_FEATURES_2D_AKAZE_DIFFUSION_HPP__
