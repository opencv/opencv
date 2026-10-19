// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_CALIB3D_RIGHT_SVD_HPP
#define OPENCV_CALIB3D_RIGHT_SVD_HPP

#include "opencv2/core.hpp"
#include <cmath>
#include <limits>

namespace cv { namespace internal {

/**
 * @brief Checks whether the normal matrix @f$A^\top A@f$ resolves the right
 * singular vectors of @f$A@f$ that belong to smaller eigenvalues.
 *
 * Forming @f$A^\top A@f$ squares the condition number. Eigenvectors below the
 * eigenvalue @f$\lambda@f$ are perturbed by roughly
 * @f$\epsilon\lambda_{\max}/\lambda@f$. Requiring
 * @f$\lambda > \sqrt{\epsilon}\lambda_{\max}@f$ bounds this perturbation by
 * @f$\sqrt{\epsilon}@f$.
 *
 * @param eigenvalue Smallest eigenvalue that must remain separated from the
 * eigenvalues of the solution subspace.
 * @param largestEigenvalue Largest eigenvalue of the normal matrix.
 */
inline bool isNormalMatrixResolved(double eigenvalue, double largestEigenvalue)
{
    return eigenvalue > largestEigenvalue * std::sqrt(std::numeric_limits<double>::epsilon());
}

/**
 * @brief Computes the singular values and right singular vectors of a matrix
 * without forming its normal matrix.
 *
 * The complete right singular basis is retained for underdetermined systems.
 */
inline void SVDecompRight(const Mat& source, OutputArray values, OutputArray right)
{
    const int flags = source.rows < source.cols ? SVD::FULL_UV : 0;
    SVD::compute(source, values, noArray(), right, flags);
}

}}
#endif
