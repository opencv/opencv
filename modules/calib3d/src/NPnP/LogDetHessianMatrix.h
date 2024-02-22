/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#ifndef PNP_USING_EIGEN_LIBRARY_LOGDETHESSIANMATRIX_H
#define PNP_USING_EIGEN_LIBRARY_LOGDETHESSIANMATRIX_H

#include "../Utils_Npnp/Definitions.h"

namespace NPnP
{

    typedef ColMatrix<M_MATRIX_DIM, M_MATRIX_DIM> inv_mat_15_15_type;

    class log_det_hessian_matrix_functor
    {
        const inv_mat_15_15_type &inv_mat;

    public:
        explicit log_det_hessian_matrix_functor(const inv_mat_15_15_type &invmat)
            : inv_mat(invmat) {}

        double operator()(int row, int col) const
        {
            if (row <= M_MATRIX_DIM || col <= M_MATRIX_DIM)
                return 0;
            row -= M_MATRIX_DIM;
            col -= M_MATRIX_DIM;
            int row1 = row / M_MATRIX_DIM;
            int col1 = row % M_MATRIX_DIM;
            int row2 = col / M_MATRIX_DIM;
            int col2 = col % M_MATRIX_DIM;
            return inv_mat(row1, col2) * inv_mat(col1, row2);
        }
    };
    Eigen::CwiseNullaryOp<log_det_hessian_matrix_functor, Eigen::MatrixXd>
    make_log_det_hessian_matrix(const inv_mat_15_15_type &inv_mat);
} // namespace NPnP

#endif // PNP_USING_EIGEN_LIBRARY_LOGDETHESSIANMATRIX_H
